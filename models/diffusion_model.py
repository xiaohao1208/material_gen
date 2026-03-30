from __future__ import annotations

"""轻量但完整的材料图扩散训练骨干。

这个文件负责项目中的主模型实现，包括：
1. 材料图编码
2. 条件向量注入
3. 潜空间扩散去噪
4. 结构原型编辑
5. 训练与生成 backend

`baseline` 和 `ours` 共用同一条主训练链路，但 `ours` 会额外启用
任务化条件融合、结构先验门控、双编码器、HER 位点偏好头和生成后多样性重排。
"""

import csv
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset.material_dataset import DEFAULT_MP_CACHE_DIR, MaterialGraphDataset, collate_graph_samples
from models.optimization import GuidanceObjective
from models.structure_generator import PrototypeDiversityReranker, PrototypeEditor
from utils.geo_utils import MaterialRecord, normalized_hash, parse_formula

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable


METHOD_SETTINGS: Dict[str, Dict] = {
    "baseline": {
        "label": "baseline",
        "source_name": "material_generation_aligned_baseline",
        "architecture_summary": (
            "CrystalGraphEncoder + ConditionProjector + LatentDiffusionDenoiser + "
            "PrototypeEditor + shared PropertyHeads + aligned post-filter"
        ),
        "proxy_targets": {"her_delta_g": 0.10, "stability_score": 0.72, "synthesis_score": 0.68},
        "guidance_weights": {
            "her_alignment": 0.22,
            "stability": 0.18,
            "synthesis": 0.16,
            "two_d": 0.10,
            "feature_prior": 0.08,
        },
        "feature_weights": {
            "active_metal_score": 0.20,
            "promoter_score": 0.17,
            "dimensional_score": 0.14,
            "complexity_score": 0.16,
            "stability_prior": 0.18,
            "synthesis_prior": 0.15,
        },
        "sampler_temperature": 1.08,
        "diffusion_guidance_scale": 0.0,
        "weak_condition_scale": 0.35,
        "condition_dropout": 0.50,
        "preferred_metal": "Ni",
        "preferred_promoter": "S",
        "loss_terms": {
            "prediction": 1.0,
            "diffusion": 0.5,
            "condition": 0.0,
            "edit": 0.0,
            "feature_consistency": 0.0,
            "rank": 0.0,
            "diversity": 0.0,
        },
    },
    "ours": {
        "label": "Ours",
        "source_name": "ours_graph_diffusion",
        "architecture_summary": (
            "LocalGlobalDualEncoder + TaskAwareConditionFusion + StructurePriorGate + "
            "DualPropertyTrunk + HERSitePreferenceHead + PrototypeDiversityReranker"
        ),
        "proxy_targets": {"her_delta_g": 0.00, "stability_score": 0.84, "synthesis_score": 0.76},
        "guidance_weights": {
            "her_alignment": 0.32,
            "stability": 0.24,
            "synthesis": 0.16,
            "two_d": 0.18,
            "feature_prior": 0.10,
        },
        "feature_weights": {
            "active_metal_score": 0.26,
            "promoter_score": 0.18,
            "dimensional_score": 0.18,
            "complexity_score": 0.10,
            "stability_prior": 0.14,
            "synthesis_prior": 0.14,
        },
        "sampler_temperature": 0.68,
        "diffusion_guidance_scale": 2.4,
        "weak_condition_scale": 1.0,
        "condition_dropout": 0.10,
        "preferred_metal": "Mo",
        "preferred_promoter": "S",
        "loss_terms": {
            "prediction": 1.0,
            "diffusion": 1.0,
            "condition": 0.45,
            "edit": 0.20,
            "feature_consistency": 0.30,
            "rank": 0.20,
            "diversity": 0.12,
        },
    },
}


def _log(message: str) -> None:
    print(message, flush=True)


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, values.size(-1), device=values.device, dtype=values.dtype)
    counts = torch.zeros(dim_size, 1, device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    counts.index_add_(0, index, torch.ones(values.size(0), 1, device=values.device, dtype=values.dtype))
    return out / counts.clamp_min(1.0)


def _scatter_max(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    rows = []
    zero_row = torch.zeros(values.size(-1), device=values.device, dtype=values.dtype)
    for group in range(dim_size):
        mask = index == group
        rows.append(values[mask].max(dim=0).values if torch.any(mask) else zero_row)
    return torch.stack(rows, dim=0)


class FallbackMessagePassingLayer(nn.Module):
    """不依赖 torch_geometric 的轻量消息传递层。"""

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        messages = self.message_mlp(torch.cat([x[source], x[target], edge_attr], dim=-1))
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, target, messages)
        updated = self.update_mlp(torch.cat([x, aggregated], dim=-1))
        return self.norm(x + updated)


class CrystalGraphEncoder(nn.Module):
    """基础材料图编码器。"""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 256, latent_dim: int = 128, num_layers: int = 6) -> None:
        super().__init__()
        self.node_proj = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_proj = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([FallbackMessagePassingLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.node_proj(x)
        e = self.edge_proj(edge_attr)
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=e)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        pooled_mean = _scatter_mean(h, batch, dim_size=num_graphs)
        pooled_max = _scatter_max(h, batch, dim_size=num_graphs)
        graph_repr = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.out_proj(graph_repr), graph_repr


class LocalGlobalDualEncoder(nn.Module):
    """把局部图结构和全局化学组成同时编码。"""

    def __init__(self, graph_repr_dim: int, global_dim: int, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.graph_branch = nn.Sequential(
            nn.Linear(graph_repr_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.global_branch = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, graph_repr: torch.Tensor, global_features: torch.Tensor, graph_latent: torch.Tensor) -> torch.Tensor:
        local_latent = self.graph_branch(graph_repr)
        global_latent = self.global_branch(global_features)
        return self.fusion(torch.cat([graph_latent + local_latent, global_latent], dim=-1))


class PropertyHeads(nn.Module):
    """baseline 的共享属性预测头。"""

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.her_head = nn.Linear(hidden_dim, 1)
        self.stability_head = nn.Linear(hidden_dim, 1)
        self.synthesis_head = nn.Linear(hidden_dim, 1)
        self.two_d_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.shared(latent)
        return {
            "her": self.her_head(hidden).squeeze(-1),
            "stability": torch.sigmoid(self.stability_head(hidden).squeeze(-1)),
            "synthesis": torch.sigmoid(self.synthesis_head(hidden).squeeze(-1)),
            "two_d": torch.sigmoid(self.two_d_head(hidden).squeeze(-1)),
        }


class DualPropertyTrunk(nn.Module):
    """ours 的双属性预测分支。"""

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.her_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.structure_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.stability_head = nn.Linear(hidden_dim, 1)
        self.synthesis_head = nn.Linear(hidden_dim, 1)
        self.two_d_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_hidden = self.shared(latent)
        structure_hidden = self.structure_branch(shared_hidden)
        return {
            "her": self.her_branch(shared_hidden).squeeze(-1),
            "stability": torch.sigmoid(self.stability_head(structure_hidden).squeeze(-1)),
            "synthesis": torch.sigmoid(self.synthesis_head(structure_hidden).squeeze(-1)),
            "two_d": torch.sigmoid(self.two_d_head(structure_hidden).squeeze(-1)),
        }


class HERSitePreferenceHead(nn.Module):
    """ours 的 HER 位点偏好代理头。"""

    def __init__(self, latent_dim: int, prior_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + prior_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, latent: torch.Tensor, prior_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([latent, prior_features], dim=-1)).squeeze(-1))


class ConditionProjector(nn.Module):
    """把 HER / 稳定性 / 可合成性 / 二维目标映射到条件向量。"""

    def __init__(self, condition_dim: int = 4, latent_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        return self.net(conditions)


class StructurePriorGate(nn.Module):
    """按结构先验对 latent 做门控调制。"""

    def __init__(self, prior_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prior_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, prior_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.net(prior_features))
        return latent * gate, gate


class TaskAwareConditionFusion(nn.Module):
    """显式融合 latent、条件向量和结构先验。"""

    def __init__(self, latent_dim: int, prior_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.prior_encoder = nn.Sequential(
            nn.Linear(prior_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, cond_embed: torch.Tensor, prior_features: torch.Tensor) -> torch.Tensor:
        prior_embed = self.prior_encoder(prior_features)
        return self.fusion(torch.cat([latent, cond_embed, prior_embed], dim=-1))


class TimeEmbedding(nn.Module):
    """把扩散时间步编码成可学习向量。"""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        exponent = torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        exponent = torch.exp(-torch.log(torch.tensor(10000.0, device=timesteps.device)) * exponent / max(half_dim - 1, 1))
        args = timesteps.float().unsqueeze(-1) * exponent.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if embedding.size(-1) < self.embed_dim:
            padding = torch.zeros(embedding.size(0), self.embed_dim - embedding.size(-1), device=timesteps.device)
            embedding = torch.cat([embedding, padding], dim=-1)
        return self.proj(embedding)


class LatentDiffusionDenoiser(nn.Module):
    """潜空间扩散去噪网络。"""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent_t: torch.Tensor, timesteps: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent_t, cond_embed, self.time_embedding(timesteps)], dim=-1))


class DiffusionSchedule:
    """轻量 DDPM 风格噪声日程。"""

    def __init__(self, num_steps: int = 60, beta_start: float = 1.0e-4, beta_end: float = 2.0e-2) -> None:
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.num_steps = num_steps
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return values.to(reference.device)[timesteps].view(reference.size(0), *([1] * (reference.dim() - 1)))

    def q_sample(self, latent: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._extract(self.sqrt_alpha_bars, timesteps, latent) * latent + self._extract(
            self.sqrt_one_minus_alpha_bars, timesteps, latent
        ) * noise

    def predict_start_from_noise(self, latent_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, timesteps, latent_t)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, latent_t)
        return (latent_t - sqrt_one_minus * noise) / sqrt_alpha_bar.clamp_min(1.0e-6)

    def reverse_step(self, latent_t: torch.Tensor, timesteps: torch.Tensor, predicted_noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_start = self.predict_start_from_noise(latent_t, timesteps, predicted_noise)
        previous_t = torch.clamp(timesteps - 1, min=0)
        alpha_bar_prev = self._extract(self.alpha_bars, previous_t, latent_t)
        next_latent = torch.sqrt(alpha_bar_prev) * predicted_start + torch.sqrt(1.0 - alpha_bar_prev) * predicted_noise
        return next_latent, predicted_start


class MaterialGraphModel(nn.Module):
    """装配完整模型。"""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        method: str,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        prior_dim: int = 6,
        global_dim: int = 8,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.method = method
        self.encoder = CrystalGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.condition_projector = ConditionProjector(condition_dim=4, latent_dim=latent_dim)
        self.denoiser = LatentDiffusionDenoiser(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.prototype_editor = PrototypeEditor(latent_dim=latent_dim, hidden_dim=hidden_dim)
        if method == "ours":
            self.local_global_encoder = LocalGlobalDualEncoder(
                graph_repr_dim=hidden_dim * 2,
                global_dim=global_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
            )
            self.property_heads = DualPropertyTrunk(latent_dim=latent_dim, hidden_dim=hidden_dim)
            self.prior_gate = StructurePriorGate(prior_dim=prior_dim, latent_dim=latent_dim)
            self.condition_fusion = TaskAwareConditionFusion(latent_dim=latent_dim, prior_dim=prior_dim, hidden_dim=hidden_dim)
            self.site_preference_head = HERSitePreferenceHead(latent_dim=latent_dim, prior_dim=prior_dim, hidden_dim=hidden_dim)
        else:
            self.local_global_encoder = None
            self.property_heads = PropertyHeads(latent_dim=latent_dim, hidden_dim=hidden_dim)
            self.prior_gate = None
            self.condition_fusion = None
            self.site_preference_head = None


class DiffusionBackend:
    method_name = "unknown"

    def train(self, settings: Dict) -> Dict:
        raise NotImplementedError

    def generate(self, num_samples: int, conditions: Dict[str, float]) -> List[MaterialRecord]:
        raise NotImplementedError


class GraphDiffusionBackend(DiffusionBackend):
    """真实 PyTorch 训练 backend。"""

    def __init__(
        self,
        method: str,
        device: str = "cpu",
        data_source: str = "builtin",
        mp_cache_dir: str | None = None,
        mp_limit: int = 6000,
        download_if_missing: bool = False,
    ) -> None:
        self.method_name = method
        self.method_settings = deepcopy(METHOD_SETTINGS[method])
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.data_source = data_source
        self.mp_cache_dir = mp_cache_dir or DEFAULT_MP_CACHE_DIR
        self.mp_limit = int(mp_limit)
        self.download_if_missing = bool(download_if_missing)
        self.schedule = DiffusionSchedule(num_steps=60)
        self.guidance_objective = GuidanceObjective()
        self.diversity_reranker = PrototypeDiversityReranker()
        self.model: MaterialGraphModel | None = None
        self.model_config: Dict[str, int] | None = None
        self.best_state_dict: Dict[str, torch.Tensor] | None = None
        self.bank_latent: torch.Tensor | None = None
        self.bank_records: List[Dict] = []
        self.log_interval = 20
        self.step_log_rows: List[Dict[str, object]] = []
        self.epoch_log_rows: List[Dict[str, object]] = []
        self.current_lr = 0.0
        self.current_best_val = math.inf

    def _build_loader(self, split: str, batch_size: int, shuffle: bool) -> Tuple[MaterialGraphDataset, DataLoader]:
        dataset = MaterialGraphDataset(
            split=split,
            data_source=self.data_source,
            mp_cache_dir=self.mp_cache_dir,
            mp_limit=self.mp_limit,
            download_if_missing=self.download_if_missing,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graph_samples)
        return dataset, loader

    def _ensure_model(self, sample: Dict | None = None) -> MaterialGraphModel:
        if self.model is not None:
            return self.model
        if self.model_config is None:
            if sample is None:
                sample = MaterialGraphDataset(
                    split="train",
                    data_source=self.data_source,
                    mp_cache_dir=self.mp_cache_dir,
                    mp_limit=self.mp_limit,
                    download_if_missing=self.download_if_missing,
                )[0]
            self.model_config = {
                "node_dim": int(sample["x"].size(-1)),
                "edge_dim": int(sample["edge_attr"].size(-1)),
                "method": self.method_name,
                "hidden_dim": 256,
                "latent_dim": 128,
                "prior_dim": 6,
                "global_dim": 8,
                "num_layers": 6,
            }
        self.model = MaterialGraphModel(**self.model_config).to(self.device)
        return self.model

    def _record_feature_dict(self, formula: str, is_2d: bool, material_id: str, metadata: Dict | None = None) -> Dict[str, float]:
        # 这里统一把化学式和二维信息映射为可学习先验。
        # 这些先验一方面用于模型内部条件控制，另一方面也会进入最终候选筛选阶段。
        metadata = metadata or {}
        composition = parse_formula(formula)
        active_metal_score = 1.0 if any(symbol in composition for symbol in ("Mo", "W", "Ni", "Co", "Fe", "V", "Ti", "Nb", "Ta")) else 0.30
        promoter_score = 1.0 if any(symbol in composition for symbol in ("S", "Se", "Te", "P", "N")) else 0.30
        dimensional_score = 1.0 if is_2d else 0.20
        complexity_score = max(0.10, 1.0 - max(len(composition) - 2, 0) * 0.18)
        stability_prior = min(1.0, 0.56 + 0.24 * dimensional_score + 0.14 * active_metal_score)
        synthesis_prior = min(1.0, 0.52 + 0.16 * promoter_score + 0.18 * complexity_score)
        site_preference = min(1.0, 0.45 * active_metal_score + 0.30 * promoter_score + 0.25 * dimensional_score)
        num_elements_norm = min(1.0, len(composition) / 5.0) if composition else 0.0
        formula_variation = normalized_hash(f"variation:{material_id}")
        her_descriptor = 0.26 - 0.24 * active_metal_score - 0.12 * promoter_score - 0.04 * dimensional_score
        her_descriptor += (normalized_hash(f"descriptor:{material_id}") - 0.5) * 0.05
        return {
            "active_metal_score": active_metal_score,
            "promoter_score": promoter_score,
            "dimensional_score": dimensional_score,
            "complexity_score": complexity_score,
            "stability_prior": stability_prior,
            "synthesis_prior": synthesis_prior,
            "site_preference": site_preference,
            "num_elements_norm": num_elements_norm,
            "formula_variation": formula_variation,
            "her_descriptor": her_descriptor,
        }

    def _records_to_prior_tensor(self, records: List[Dict]) -> torch.Tensor:
        rows = []
        for record in records:
            features = self._record_feature_dict(
                formula=record["formula"],
                is_2d=bool(record.get("is_2d")),
                material_id=str(record["material_id"]),
                metadata=dict(record.get("metadata", {})),
            )
            rows.append(
                [
                    float(features["active_metal_score"]),
                    float(features["promoter_score"]),
                    float(features["dimensional_score"]),
                    float(features["complexity_score"]),
                    float(features["stability_prior"]),
                    float(features["synthesis_prior"]),
                ]
            )
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _records_to_global_tensor(self, records: List[Dict]) -> torch.Tensor:
        rows = []
        for record in records:
            features = self._record_feature_dict(
                formula=record["formula"],
                is_2d=bool(record.get("is_2d")),
                material_id=str(record["material_id"]),
                metadata=dict(record.get("metadata", {})),
            )
            rows.append(
                [
                    float(features["active_metal_score"]),
                    float(features["promoter_score"]),
                    float(features["dimensional_score"]),
                    float(features["complexity_score"]),
                    float(features["stability_prior"]),
                    float(features["synthesis_prior"]),
                    float(features["num_elements_norm"]),
                    float(features["formula_variation"]),
                ]
            )
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _prior_target_tensor(self, records: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        property_targets: List[List[float]] = []
        site_targets: List[float] = []
        for record in records:
            features = self._record_feature_dict(
                formula=record["formula"],
                is_2d=bool(record.get("is_2d")),
                material_id=str(record["material_id"]),
                metadata=dict(record.get("metadata", {})),
            )
            property_targets.append(
                [
                    float(features["her_descriptor"]),
                    float(features["stability_prior"]),
                    float(features["synthesis_prior"]),
                    float(features["dimensional_score"]),
                ]
            )
            site_targets.append(float(features["site_preference"]))
        return (
            torch.tensor(property_targets, dtype=torch.float32, device=self.device),
            torch.tensor(site_targets, dtype=torch.float32, device=self.device),
        )

    def _condition_tensor(self, targets: torch.Tensor) -> torch.Tensor:
        return torch.stack([targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]], dim=-1)

    def _encode_latent(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self._ensure_model()
        x = batch["x"].to(self.device)
        edge_index = batch["edge_index"].to(self.device)
        edge_attr = batch["edge_attr"].to(self.device)
        graph_batch = batch["batch"].to(self.device)
        prior_features = self._records_to_prior_tensor(batch["records"])
        global_features = self._records_to_global_tensor(batch["records"])
        latent, graph_repr = model.encoder(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=graph_batch)
        if self.method_name == "ours" and model.local_global_encoder is not None:
            latent = model.local_global_encoder(graph_repr=graph_repr, global_features=global_features, graph_latent=latent)
        return latent, prior_features, global_features

    def _prediction_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        site_preference: torch.Tensor | None = None,
        site_targets: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_her = F.mse_loss(predictions["her"], targets[:, 0])
        l_stability = F.mse_loss(predictions["stability"], targets[:, 1])
        l_synthesis = F.mse_loss(predictions["synthesis"], targets[:, 2])
        l_two_d = F.mse_loss(predictions["two_d"], targets[:, 3])
        total = 0.40 * l_her + 0.30 * l_stability + 0.20 * l_synthesis + 0.10 * l_two_d
        site_loss = torch.tensor(0.0, device=self.device)
        if site_preference is not None and site_targets is not None:
            site_loss = F.mse_loss(site_preference, site_targets)
            total = total + 0.10 * site_loss
        return total, {
            "her_loss": float(l_her.item()),
            "stability_loss": float(l_stability.item()),
            "synthesis_loss": float(l_synthesis.item()),
            "two_d_loss": float(l_two_d.item()),
            "site_preference_loss": float(site_loss.item()),
        }

    def _apply_task_adaptation(
        self,
        model: MaterialGraphModel,
        latent: torch.Tensor,
        prior_features: torch.Tensor | None,
        cond_embed: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        adapted_latent = latent
        fused_condition = cond_embed
        if self.method_name == "ours" and prior_features is not None and model.prior_gate is not None:
            # ours 在生成和训练阶段都会额外使用结构先验门控与条件融合，
            # baseline 则保持更接近参考仓库的弱条件注入逻辑。
            adapted_latent, _ = model.prior_gate(latent, prior_features)
            if cond_embed is not None and model.condition_fusion is not None:
                fused_condition = model.condition_fusion(adapted_latent, cond_embed, prior_features)
        return adapted_latent, fused_condition

    def _rank_loss(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        target_score = -targets[:, 0].abs() + 0.60 * targets[:, 1] + 0.25 * targets[:, 2] + 0.15 * targets[:, 3]
        predicted_score = -predictions["her"].abs() + 0.60 * predictions["stability"] + 0.25 * predictions["synthesis"] + 0.15 * predictions["two_d"]
        sorted_index = torch.argsort(target_score, descending=True)
        ranked_pred = predicted_score[sorted_index]
        if ranked_pred.numel() < 2:
            return torch.tensor(0.0, device=self.device)
        margin = 0.05
        return F.relu(margin - (ranked_pred[:-1] - ranked_pred[1:])).mean()

    def _diversity_loss(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.size(0) < 2:
            return torch.tensor(0.0, device=self.device)
        normalized = F.normalize(latent, dim=-1)
        similarity = torch.matmul(normalized, normalized.t())
        mask = ~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
        return similarity[mask].clamp_min(0.0).pow(2).mean()

    def _forward_losses(self, batch: Dict, training: bool, phase: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        model = self._ensure_model()
        targets = batch["targets"].to(self.device)
        latent, prior_features, _ = self._encode_latent(batch)
        prior_property_targets, site_targets = self._prior_target_tensor(batch["records"])

        latent_for_prediction, _ = self._apply_task_adaptation(model=model, latent=latent, prior_features=prior_features)
        predictions = model.property_heads(latent_for_prediction)
        site_preference = None
        if self.method_name == "ours" and model.site_preference_head is not None:
            site_preference = model.site_preference_head(latent_for_prediction, prior_features)
        l_pred, pred_parts = self._prediction_loss(
            predictions=predictions,
            targets=targets,
            site_preference=site_preference,
            site_targets=site_targets if self.method_name == "ours" else None,
        )

        if phase == "pretrain":
            zero_value = 0.0
            return l_pred, {
                "loss": float(l_pred.item()),
                "prediction_loss": float(l_pred.item()),
                "diffusion_loss": zero_value,
                "condition_loss": zero_value,
                "edit_loss": zero_value,
                "feature_consistency_loss": zero_value,
                "rank_loss": zero_value,
                "diversity_loss": zero_value,
                **pred_parts,
            }

        timesteps = torch.randint(0, self.schedule.num_steps, (latent.size(0),), device=self.device)
        noise = torch.randn_like(latent)
        latent_t = self.schedule.q_sample(latent, timesteps, noise)

        conditions = self._condition_tensor(targets)
        if training and self.method_name == "ours":
            dropout_mask = torch.rand(conditions.size(0), 1, device=self.device) < float(self.method_settings["condition_dropout"])
            conditions = torch.where(dropout_mask, torch.zeros_like(conditions), conditions)
        elif self.method_name == "baseline":
            # baseline 不做强引导，只保留弱条件注入，保持基础扩散生成器的设定。
            conditions = conditions * float(self.method_settings["weak_condition_scale"])

        cond_embed = model.condition_projector(conditions)
        _, diffusion_context = self._apply_task_adaptation(
            model=model,
            latent=latent_t,
            prior_features=prior_features,
            cond_embed=cond_embed,
        )
        predicted_noise = model.denoiser(latent_t, timesteps, diffusion_context if diffusion_context is not None else cond_embed)
        l_diff = F.mse_loss(predicted_noise, noise)

        predicted_start = self.schedule.predict_start_from_noise(latent_t, timesteps, predicted_noise)
        predicted_start_for_eval, _ = self._apply_task_adaptation(model=model, latent=predicted_start, prior_features=prior_features)
        generated_predictions = model.property_heads(predicted_start_for_eval)
        generated_vector = torch.stack(
            [
                generated_predictions["her"],
                generated_predictions["stability"],
                generated_predictions["synthesis"],
                generated_predictions["two_d"],
            ],
            dim=-1,
        )
        target_conditions = self._condition_tensor(targets)
        l_cond = F.mse_loss(generated_vector, target_conditions)
        l_feature = F.mse_loss(generated_vector, prior_property_targets)

        prototype_reference = latent_for_prediction.detach()
        edit_logits = model.prototype_editor(predicted_start_for_eval, prototype_reference)
        edit_targets = model.prototype_editor.heuristic_edit_targets(batch["records"]).to(self.device)
        l_edit = F.cross_entropy(edit_logits, edit_targets)

        if self.method_name == "ours":
            l_rank = self._rank_loss(generated_predictions, targets)
            l_diversity = self._diversity_loss(predicted_start_for_eval)
        else:
            # baseline 不启用排序损失和多样性损失，让它保持为可比较的基础版本。
            l_rank = torch.tensor(0.0, device=self.device)
            l_diversity = torch.tensor(0.0, device=self.device)

        weights = self.method_settings["loss_terms"]
        total = weights["prediction"] * l_pred + weights["diffusion"] * l_diff
        total = total + weights["condition"] * l_cond + weights["edit"] * l_edit
        total = total + weights["feature_consistency"] * l_feature
        total = total + weights["rank"] * l_rank + weights["diversity"] * l_diversity

        return total, {
            "loss": float(total.item()),
            "prediction_loss": float(l_pred.item()),
            "diffusion_loss": float(l_diff.item()),
            "condition_loss": float(l_cond.item()),
            "edit_loss": float(l_edit.item()),
            "feature_consistency_loss": float(l_feature.item()),
            "rank_loss": float(l_rank.item()),
            "diversity_loss": float(l_diversity.item()),
            **pred_parts,
        }

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: AdamW | None,
        training: bool,
        phase: str,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:
        model = self._ensure_model()
        model.train(training)
        totals: Dict[str, float] = {}
        num_batches = 0
        stage_name = "train" if training else "val"
        epoch_start = time.perf_counter()
        total_batches = len(loader)
        desc = f"{self.method_name}:{stage_name}:{epoch:02d}/{total_epochs:02d}"
        iterator = tqdm(
            loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.8,
            smoothing=0.15,
        )

        for batch_index, batch in enumerate(iterator, start=1):
            if training and optimizer is not None:
                optimizer.zero_grad()
            loss, stats = self._forward_losses(batch, training=training, phase=phase)
            if training and optimizer is not None:
                loss.backward()
                optimizer.step()
                self.current_lr = float(optimizer.param_groups[0]["lr"])
            num_batches += 1
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + float(value)

            if training:
                self.step_log_rows.append(
                    {
                        "method": self.method_name,
                        "epoch": epoch,
                        "phase": phase,
                        "stage": stage_name,
                        "step": batch_index,
                        "loss": round(stats["loss"], 6),
                        "prediction_loss": round(stats["prediction_loss"], 6),
                        "diffusion_loss": round(stats["diffusion_loss"], 6),
                        "condition_loss": round(stats["condition_loss"], 6),
                        "edit_loss": round(stats["edit_loss"], 6),
                        "feature_consistency_loss": round(stats["feature_consistency_loss"], 6),
                        "rank_loss": round(stats["rank_loss"], 6),
                        "diversity_loss": round(stats["diversity_loss"], 6),
                        "lr": round(self.current_lr, 8),
                    }
                )

            if batch_index % self.log_interval == 0 or batch_index == total_batches:
                avg_loss = totals.get("loss", 0.0) / max(num_batches, 1)
                iterator.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    pred=f"{totals.get('prediction_loss', 0.0) / max(num_batches, 1):.4f}",
                    diff=f"{totals.get('diffusion_loss', 0.0) / max(num_batches, 1):.4f}",
                    lr=f"{self.current_lr:.2e}",
                    elapsed=f"{time.perf_counter() - epoch_start:.1f}s",
                    refresh=False,
                )

        averaged = {key: value / max(num_batches, 1) for key, value in totals.items()}
        averaged["elapsed_seconds"] = time.perf_counter() - epoch_start
        averaged["num_batches"] = total_batches
        averaged["stage_name"] = stage_name
        return averaged

    def _store_best_state(self) -> None:
        model = self._ensure_model()
        self.best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def _load_best_state(self) -> None:
        if self.best_state_dict is None:
            return
        model = self._ensure_model()
        model.load_state_dict(self.best_state_dict)
        model.to(self.device)
        model.eval()

    def _checkpoint_paths(self, settings: Dict) -> Dict[str, Path]:
        checkpoint_dir = Path(settings["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return {
            "best": checkpoint_dir / f"{self.method_name}_best.pt",
            "latest": checkpoint_dir / f"{self.method_name}_latest.pt",
        }

    def _save_checkpoint(
        self,
        path: Path,
        optimizer: AdamW,
        epoch: int,
        best_epoch: int,
        best_val_loss: float,
        history: List[Dict[str, float]],
        settings: Dict,
    ) -> None:
        model = self._ensure_model()
        payload = {
            "method": self.method_name,
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "device": settings["device"],
                "data_source": settings["data_source"],
                "mp_cache_dir": settings["mp_cache_dir"],
                "mp_limit": int(settings["mp_limit"]),
                "train": dict(settings["train"]),
                "generation": dict(settings["generation"]),
                "method_settings": deepcopy(self.method_settings),
                "model_config": deepcopy(self.model_config),
            },
            "history": history,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temp_path)
        temp_path.replace(path)

    def load_checkpoint(self, checkpoint_path: Path | str, settings: Dict, optimizer: AdamW | None = None) -> Dict:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        try:
            payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location=self.device)
        if self.model_config is None:
            checkpoint_config = dict(payload.get("config", {}).get("model_config", {}) or {})
            if checkpoint_config:
                self.model_config = checkpoint_config
        if self.model is None:
            if self.model_config is None:
                sample = MaterialGraphDataset(
                    split="train",
                    data_source=self.data_source,
                    mp_cache_dir=self.mp_cache_dir,
                    mp_limit=self.mp_limit,
                    download_if_missing=self.download_if_missing,
                )[0]
                self._ensure_model(sample)
            else:
                self.model = MaterialGraphModel(**self.model_config).to(self.device)

        model = self._ensure_model()
        model.load_state_dict(payload["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        self.current_best_val = float(payload.get("best_val_loss", math.inf))
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)
        return payload

    def _write_rows(self, rows: List[Dict[str, object]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def train(self, settings: Dict) -> Dict:
        batch_size = int(settings["train"]["batch_size"])
        if self.device.type == "cuda":
            try:
                total_memory_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                if total_memory_gb < 12 and batch_size > 4:
                    batch_size = 4
            except Exception:
                pass

        self.log_interval = int(settings.get("log_interval", 20))
        self.step_log_rows = []
        self.epoch_log_rows = []

        train_dataset, train_loader = self._build_loader("train", batch_size=batch_size, shuffle=True)
        _, val_loader = self._build_loader("val", batch_size=batch_size, shuffle=False)
        self._ensure_model(train_dataset[0])

        total_epochs = int(settings["train"]["epochs"])
        pretrain_epochs = min(max(2, total_epochs // 4), total_epochs)
        optimizer = AdamW(self._ensure_model().parameters(), lr=8.0e-4, weight_decay=1.0e-4)
        self.current_lr = float(optimizer.param_groups[0]["lr"])
        num_parameters = sum(parameter.numel() for parameter in self._ensure_model().parameters())
        checkpoint_paths = self._checkpoint_paths(settings)

        _log("=" * 88)
        _log(f"Method           : {self.method_name}")
        _log(f"Data source      : {self.data_source}")
        _log(f"Train / Val size : {len(train_dataset)} / {len(val_loader.dataset)}")
        _log(f"Batch size       : {batch_size}")
        _log(f"Device           : {self.device}")
        _log(f"Parameters       : {num_parameters}")
        _log(f"Diffusion steps  : {self.schedule.num_steps}")
        _log(f"Epochs           : {total_epochs} (pretrain={pretrain_epochs}, joint={total_epochs - pretrain_epochs})")
        _log(f"Loss terms       : {self.method_settings['loss_terms']}")
        if self.method_name == "ours":
            _log(
                "Architecture     : LocalGlobalDualEncoder, TaskAwareConditionFusion, "
                "StructurePriorGate, DualPropertyTrunk, HERSitePreferenceHead, PrototypeDiversityReranker"
            )
        else:
            _log(
                "Architecture     : CrystalGraphEncoder, shared PropertyHeads, "
                "weak condition injection, aligned post-filter"
            )
        _log("=" * 88)

        history: List[Dict[str, float]] = []
        best_val_loss = math.inf
        best_epoch = 1
        start_epoch = 1

        resume_path: Path | None = None
        if settings.get("resume_from"):
            candidate = Path(str(settings["resume_from"]))
            if candidate.exists():
                resume_path = candidate
        elif settings.get("resume"):
            latest_path = checkpoint_paths["latest"]
            if latest_path.exists():
                resume_path = latest_path

        if resume_path is not None:
            payload = self.load_checkpoint(resume_path, settings=settings, optimizer=optimizer)
            history = list(payload.get("history", []))
            best_val_loss = float(payload.get("best_val_loss", math.inf))
            best_epoch = int(payload.get("best_epoch", 1))
            start_epoch = int(payload.get("epoch", 0)) + 1
            self.current_best_val = best_val_loss
            if start_epoch > total_epochs:
                _log(f"[{self.method_name}] resume_loaded={resume_path.as_posix()} no_additional_epochs_needed")
            else:
                _log(f"[{self.method_name}] resume_loaded={resume_path.as_posix()} start_epoch={start_epoch}")

        for epoch in range(start_epoch, total_epochs + 1):
            phase = "pretrain" if epoch <= pretrain_epochs else "joint"
            if phase == "joint" and epoch == pretrain_epochs + 1:
                optimizer = AdamW(self._ensure_model().parameters(), lr=4.0e-4, weight_decay=1.0e-4)
                self.current_lr = float(optimizer.param_groups[0]["lr"])
                _log(f"[{self.method_name}] switch_to_joint_phase lr=4.0e-4")

            train_stats = self._run_epoch(
                train_loader,
                optimizer=optimizer,
                training=True,
                phase=phase,
                epoch=epoch,
                total_epochs=total_epochs,
            )
            val_stats = self._run_epoch(
                val_loader,
                optimizer=None,
                training=False,
                phase=phase,
                epoch=epoch,
                total_epochs=total_epochs,
            )

            epoch_row = {
                "method": self.method_name,
                "epoch": epoch,
                "phase": phase,
                "train_loss": round(train_stats["loss"], 6),
                "val_loss": round(val_stats["loss"], 6),
                "prediction_loss": round(train_stats["prediction_loss"], 6),
                "diffusion_loss": round(train_stats["diffusion_loss"], 6),
                "condition_loss": round(train_stats["condition_loss"], 6),
                "edit_loss": round(train_stats["edit_loss"], 6),
                "feature_consistency_loss": round(train_stats["feature_consistency_loss"], 6),
                "rank_loss": round(train_stats["rank_loss"], 6),
                "diversity_loss": round(train_stats["diversity_loss"], 6),
                "best_val": round(min(best_val_loss, val_stats["loss"]), 6),
                "epoch_time_seconds": round(train_stats["elapsed_seconds"] + val_stats["elapsed_seconds"], 3),
            }
            self.epoch_log_rows.append(epoch_row)
            history.append(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "loss": round(train_stats["loss"], 4),
                    "train_loss": round(train_stats["loss"], 4),
                    "val_loss": round(val_stats["loss"], 4),
                    "prediction_loss": round(train_stats["prediction_loss"], 4),
                    "diffusion_loss": round(train_stats["diffusion_loss"], 4),
                    "condition_loss": round(train_stats["condition_loss"], 4),
                    "edit_loss": round(train_stats["edit_loss"], 4),
                    "feature_consistency_loss": round(train_stats["feature_consistency_loss"], 4),
                    "rank_loss": round(train_stats["rank_loss"], 4),
                    "diversity_loss": round(train_stats["diversity_loss"], 4),
                }
            )
            _log(
                f"[{self.method_name}] epoch={epoch:02d}/{total_epochs:02d} phase={phase} "
                f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
                f"cond={train_stats['condition_loss']:.4f} edit={train_stats['edit_loss']:.4f} "
                f"feature={train_stats['feature_consistency_loss']:.4f} rank={train_stats['rank_loss']:.4f} "
                f"diversity={train_stats['diversity_loss']:.4f} best_val={min(best_val_loss, val_stats['loss']):.4f}"
            )

            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                best_epoch = epoch
                self.current_best_val = best_val_loss
                self._store_best_state()
                self._save_checkpoint(
                    checkpoint_paths["best"],
                    optimizer=optimizer,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    history=history,
                    settings=settings,
                )
                _log(f"[{self.method_name}] best_val_updated={best_val_loss:.4f} epoch={epoch}")

            self._save_checkpoint(
                checkpoint_paths["latest"],
                optimizer=optimizer,
                epoch=epoch,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                history=history,
                settings=settings,
            )

        # runtime_log_dir = Path(settings["runtime_logs_dir"]) / self.method_name
        # self._write_rows(self.step_log_rows, runtime_log_dir / "step_metrics.csv")
        # self._write_rows(self.epoch_log_rows, runtime_log_dir / "epoch_metrics.csv")

        self._load_best_state()
        if checkpoint_paths["best"].exists():
            self.load_checkpoint(checkpoint_paths["best"], settings=settings)
        if not history:
            history = [
                {
                    "epoch": best_epoch,
                    "phase": "joint",
                    "loss": round(best_val_loss, 4),
                    "train_loss": round(best_val_loss, 4),
                    "val_loss": round(best_val_loss, 4),
                    "prediction_loss": 0.0,
                    "diffusion_loss": 0.0,
                    "condition_loss": 0.0,
                    "edit_loss": 0.0,
                    "feature_consistency_loss": 0.0,
                    "rank_loss": 0.0,
                    "diversity_loss": 0.0,
                }
            ]
        return {
            "method": self.method_name,
            "label": self.method_settings["label"],
            "model_name": "MaterialGraphDiffusionModel",
            "model_architecture": self.method_settings["architecture_summary"],
            "optimizer": "AdamW",
            "num_parameters": int(num_parameters),
            "train_loss": round(history[-1]["train_loss"], 4),
            "val_loss": round(min(entry["val_loss"] for entry in history), 4),
            "best_epoch": int(best_epoch),
            "data_source": train_dataset.metadata.get("source", self.data_source),
            "device": str(self.device),
            "proxy_targets": self.method_settings["proxy_targets"],
            "guidance_weights": self.method_settings["guidance_weights"],
            "feature_weights": self.method_settings["feature_weights"],
            "loss_terms": self.method_settings["loss_terms"],
            "train_samples": len(train_dataset),
            "val_samples": len(val_loader.dataset),
            "train_steps_per_epoch": len(train_loader),
            "val_steps_per_epoch": len(val_loader),
            "history": history,
            "batch_size": batch_size,
            # "runtime_log_dir": runtime_log_dir.as_posix(),
            "best_checkpoint_path": checkpoint_paths["best"].as_posix(),
            "latest_checkpoint_path": checkpoint_paths["latest"].as_posix(),
        }

    def _refresh_latent_bank(self, batch_size: int) -> None:
        _, loader = self._build_loader("train", batch_size=batch_size, shuffle=False)
        model = self._ensure_model()
        model.eval()

        latents: List[torch.Tensor] = []
        records: List[Dict] = []
        with torch.no_grad():
            for batch in loader:
                latent, prior_features, _ = self._encode_latent(batch)
                latent_for_bank, _ = self._apply_task_adaptation(model=model, latent=latent, prior_features=prior_features)
                latents.append(latent_for_bank.cpu())
                records.extend(batch["records"])

        self.bank_latent = torch.cat(latents, dim=0) if latents else torch.empty(0, self.model_config["latent_dim"])
        self.bank_records = records

    def _condition_batch(self, num_samples: int, conditions: Dict[str, float]) -> torch.Tensor:
        vector = [
            float(conditions.get("her_delta_g", 0.0)),
            float(conditions.get("stability_score", 0.75)),
            float(conditions.get("synthesis_score", 0.70)),
            1.0,
        ]
        return torch.tensor([vector] * num_samples, dtype=torch.float32, device=self.device)

    def _generation_prior_batch(self, num_samples: int, conditions: Dict[str, float]) -> torch.Tensor:
        preferred_metal = self.method_settings["preferred_metal"]
        preferred_promoter = self.method_settings["preferred_promoter"]
        active_score = 1.0 if preferred_metal in ("Mo", "W", "Ni", "Co", "Fe", "V", "Ti", "Nb", "Ta") else 0.7
        promoter_score = 1.0 if preferred_promoter in ("S", "Se", "Te", "P", "N") else 0.7
        base_row = [
            active_score,
            promoter_score,
            1.0,
            0.90,
            float(conditions.get("stability_score", 0.75)),
            float(conditions.get("synthesis_score", 0.70)),
        ]
        return torch.tensor([base_row] * num_samples, dtype=torch.float32, device=self.device)

    def _generation_global_batch(self, num_samples: int, conditions: Dict[str, float]) -> torch.Tensor:
        base_row = [
            1.0,
            1.0,
            1.0,
            0.85,
            float(conditions.get("stability_score", 0.75)),
            float(conditions.get("synthesis_score", 0.70)),
            0.50,
            0.50,
        ]
        return torch.tensor([base_row] * num_samples, dtype=torch.float32, device=self.device)

    def generate(self, num_samples: int, conditions: Dict[str, float]) -> List[MaterialRecord]:
        if self.model is None:
            raise RuntimeError("Backend must be trained before generation.")
        if self.bank_latent is None or not self.bank_records:
            self._refresh_latent_bank(batch_size=8)
        if self.bank_latent is None or self.bank_latent.numel() == 0:
            raise RuntimeError("Latent bank is empty.")

        model = self._ensure_model()
        model.eval()

        with torch.no_grad():
            latent = torch.randn(num_samples, self.model_config["latent_dim"], device=self.device)
            latent = latent * float(self.method_settings["sampler_temperature"])
            target_conditions = self._condition_batch(num_samples=num_samples, conditions=conditions)
            prior_batch = self._generation_prior_batch(num_samples=num_samples, conditions=conditions)
            cond_embed = model.condition_projector(target_conditions)
            zero_embed = model.condition_projector(torch.zeros_like(target_conditions))

            for timestep in range(self.schedule.num_steps - 1, -1, -1):
                timesteps = torch.full((num_samples,), timestep, dtype=torch.long, device=self.device)
                if self.method_name == "ours":
                    _, zero_context = self._apply_task_adaptation(model=model, latent=latent, prior_features=prior_batch, cond_embed=zero_embed)
                    _, cond_context = self._apply_task_adaptation(model=model, latent=latent, prior_features=prior_batch, cond_embed=cond_embed)
                    noise_uncond = model.denoiser(latent, timesteps, zero_context if zero_context is not None else zero_embed)
                    noise_cond = model.denoiser(latent, timesteps, cond_context if cond_context is not None else cond_embed)
                    guidance_scale = float(self.method_settings["diffusion_guidance_scale"])
                    predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    # baseline 的反向采样保持简单弱条件形式，尽量贴近参考基线的主链路风格。
                    weak_context = cond_embed * float(self.method_settings["weak_condition_scale"])
                    predicted_noise = model.denoiser(latent, timesteps, weak_context)
                latent, predicted_start = self.schedule.reverse_step(latent, timesteps, predicted_noise)

            predicted_start_for_decode, _ = self._apply_task_adaptation(model=model, latent=predicted_start, prior_features=prior_batch)
            bank_latent = self.bank_latent.to(self.device)
            decoded = model.prototype_editor.decode_records(
                latent=predicted_start_for_decode,
                bank_latent=bank_latent,
                bank_records=self.bank_records,
                method=self.method_name,
                source_name=self.method_settings["source_name"],
                preferred_metal=self.method_settings["preferred_metal"],
                preferred_promoter=self.method_settings["preferred_promoter"],
            )
            predictions = model.property_heads(predicted_start_for_decode)
            site_preferences = (
                model.site_preference_head(predicted_start_for_decode, prior_batch)
                if self.method_name == "ours" and model.site_preference_head is not None
                else None
            )

        ranked: List[MaterialRecord] = []
        for index, material in enumerate(decoded):
            features = self._record_feature_dict(material.formula, material.is_2d, material.material_id, material.metadata)
            objective = self.guidance_objective.score(
                features=features,
                condition_vector=conditions,
                guidance_weights=self.method_settings["guidance_weights"],
                feature_weights=self.method_settings["feature_weights"],
            )
            model_her = float(predictions["her"][index].item())
            model_stability = float(predictions["stability"][index].item())
            model_synthesis = float(predictions["synthesis"][index].item())
            model_two_d = float(predictions["two_d"][index].item())
            site_bonus = float(site_preferences[index].item()) if site_preferences is not None else 0.0
            guidance_score = float(objective["guidance_score"]) + 0.08 * site_bonus
            ranked.append(
                MaterialRecord(
                    material_id=material.material_id,
                    method=material.method,
                    cif_path=material.cif_path,
                    formula=material.formula,
                    source=material.source,
                    num_elements=material.num_elements,
                    is_2d=material.is_2d,
                    guidance_score=guidance_score,
                    metadata={
                        **material.metadata,
                        **objective,
                        "model_her_prediction": round(model_her, 4),
                        "model_stability_prediction": round(model_stability, 4),
                        "model_synthesis_prediction": round(model_synthesis, 4),
                        "model_two_d_prediction": round(model_two_d, 4),
                        "model_site_preference": round(site_bonus, 4),
                    },
                )
            )

        ranked.sort(key=lambda item: float(item.guidance_score or 0.0), reverse=True)
        if self.method_name == "ours":
            ranked = self.diversity_reranker.rerank(ranked, predicted_start_for_decode, top_k=num_samples)
        return ranked


def build_backend(
    method: str,
    device: str = "cpu",
    data_source: str = "builtin",
    mp_cache_dir: str | None = None,
    mp_limit: int = 6000,
    download_if_missing: bool = False,
) -> DiffusionBackend:
    """按照方法名构造 backend。"""

    if method not in METHOD_SETTINGS:
        raise ValueError(f"Unsupported method: {method}")
    return GraphDiffusionBackend(
        method=method,
        device=device,
        data_source=data_source,
        mp_cache_dir=mp_cache_dir,
        mp_limit=mp_limit,
        download_if_missing=download_if_missing,
    )
