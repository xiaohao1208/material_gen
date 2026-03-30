from __future__ import annotations

"""结构原型检索、编辑与生成后重排。

这个模块负责把扩散模型产生的 latent 候选映射回可读的材料记录。
它不参与扩散去噪本身，而是专注于三件事：
1. 从训练集 latent bank 中检索相近原型
2. 对原型执行面向 HER 任务的轻量编辑
3. 对生成结果做多样性重排，避免 top-k 候选塌缩成同类公式
"""

from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from utils.geo_utils import MaterialRecord, count_elements_in_formula, mutate_formula, parse_formula


if TYPE_CHECKING:
    from models.diffusion_model import DiffusionBackend


EDIT_ACTIONS = (
    "keep",
    "swap_active_metal",
    "add_promoter",
    "simplify_composition",
    "two_d_bias",
)


class PrototypeEditor(nn.Module):
    """根据 latent 与原型相似性预测 HER 任务化编辑动作。"""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.edit_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(EDIT_ACTIONS)),
        )

    def forward(self, latent: torch.Tensor, prototype_latent: torch.Tensor) -> torch.Tensor:
        """根据当前 latent 与原型 latent 预测编辑动作分布。"""

        return self.edit_head(torch.cat([latent, prototype_latent], dim=-1))

    def retrieve(self, query_latent: torch.Tensor, bank_latent: torch.Tensor, top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 latent bank 中检索最相近的结构原型。"""

        if bank_latent.numel() == 0:
            raise RuntimeError("Prototype latent bank is empty.")
        query_norm = F.normalize(query_latent, dim=-1)
        bank_norm = F.normalize(bank_latent, dim=-1)
        similarities = torch.matmul(query_norm, bank_norm.t())
        return torch.topk(similarities, k=top_k, dim=-1)

    def heuristic_edit_targets(self, records: List[Dict]) -> torch.Tensor:
        """构造编辑头的弱监督目标。

        目标来自任务先验，而不是显式标注：
        - 缺活性金属时优先换金属
        - 缺 promoter 时优先添加 promoter
        - 成分过复杂时优先简化
        - 非二维候选时优先引入二维倾向
        - 已经比较理想时保持原型
        """

        targets: List[int] = []
        active_metals = {"Mo", "W", "Ni", "Co", "Fe", "V", "Ti", "Nb", "Ta"}
        promoters = {"S", "Se", "Te", "P", "N"}
        for record in records:
            formula = str(record["formula"])
            composition = parse_formula(formula)
            if not any(symbol in composition for symbol in active_metals):
                targets.append(1)
            elif not any(symbol in composition for symbol in promoters):
                targets.append(2)
            elif len(composition) > 3:
                targets.append(3)
            elif not bool(record.get("is_2d")):
                targets.append(4)
            else:
                targets.append(0)
        return torch.tensor(targets, dtype=torch.long)

    def _apply_action(self, formula: str, action_name: str, preferred_metal: str, preferred_promoter: str) -> str:
        if action_name == "swap_active_metal":
            return mutate_formula(formula, action="swap_metal", preferred_metal=preferred_metal, preferred_promoter=preferred_promoter)
        if action_name == "add_promoter":
            return mutate_formula(formula, action="add_promoter", preferred_metal=preferred_metal, preferred_promoter=preferred_promoter)
        if action_name == "simplify_composition":
            return mutate_formula(formula, action="light_simplify", preferred_metal=preferred_metal, preferred_promoter=preferred_promoter)
        if action_name == "two_d_bias":
            # 二维偏置不强行修改真实结构，只做轻量公式偏置，优先加入常见层状 promoter。
            return mutate_formula(formula, action="add_promoter", preferred_metal=preferred_metal, preferred_promoter=preferred_promoter)
        return formula

    def decode_records(
        self,
        latent: torch.Tensor,
        bank_latent: torch.Tensor,
        bank_records: List[Dict],
        method: str,
        source_name: str,
        preferred_metal: str,
        preferred_promoter: str,
    ) -> List[MaterialRecord]:
        """把采样 latent 解码成可展示的候选材料。"""

        scores, indices = self.retrieve(latent, bank_latent, top_k=1)
        prototype_latent = bank_latent[indices[:, 0]]
        actions = torch.argmax(self.forward(latent, prototype_latent), dim=-1)

        generated: List[MaterialRecord] = []
        for row, prototype_index in enumerate(indices[:, 0].tolist()):
            prototype = bank_records[prototype_index]
            action_name = EDIT_ACTIONS[int(actions[row].item())]
            formula = self._apply_action(
                prototype["formula"],
                action_name=action_name,
                preferred_metal=preferred_metal,
                preferred_promoter=preferred_promoter,
            )
            generated.append(
                MaterialRecord(
                    material_id=f"{method}_{row + 1:04d}_{prototype['material_id']}",
                    method=method,
                    cif_path=f"results/{method}_{row + 1:04d}.cif",
                    formula=formula,
                    source=source_name,
                    num_elements=count_elements_in_formula(formula),
                    is_2d=bool(prototype.get("is_2d", False) or action_name == "two_d_bias"),
                    guidance_score=float(scores[row, 0].item()),
                    metadata={
                        **dict(prototype.get("metadata", {})),
                        "prototype_material_id": prototype["material_id"],
                        "prototype_formula": prototype["formula"],
                        "prototype_similarity": round(float(scores[row, 0].item()), 4),
                        "edit_action": action_name,
                    },
                )
            )
        return generated


class PrototypeDiversityReranker:
    """对生成候选做多样性重排。

    这个模块只在 `ours` 中启用。它不会替换统一评分器，而是在生成候选排序前
    加入公式去重偏好、编辑动作多样性和 latent 分散度约束，避免 top-k 都是同类材料。
    """

    def __init__(self, duplicate_penalty: float = 0.18, edit_repeat_penalty: float = 0.10) -> None:
        self.duplicate_penalty = duplicate_penalty
        self.edit_repeat_penalty = edit_repeat_penalty

    def rerank(
        self,
        records: Sequence[MaterialRecord],
        latent: torch.Tensor,
        top_k: int | None = None,
    ) -> List[MaterialRecord]:
        if not records:
            return []

        latent_cpu = latent.detach().cpu()
        norms = F.normalize(latent_cpu, dim=-1)
        chosen_indices: List[int] = []
        remaining = list(range(len(records)))
        formula_counts: Dict[str, int] = {}
        edit_counts: Dict[str, int] = {}

        target_size = min(top_k or len(records), len(records))
        while remaining and len(chosen_indices) < target_size:
            best_index = remaining[0]
            best_score = float("-inf")
            for candidate_index in remaining:
                record = records[candidate_index]
                base_score = float(record.guidance_score or 0.0)
                formula_penalty = self.duplicate_penalty * formula_counts.get(record.formula, 0)
                edit_name = str(record.metadata.get("edit_action", "keep"))
                edit_penalty = self.edit_repeat_penalty * edit_counts.get(edit_name, 0)
                diversity_bonus = 0.0
                if chosen_indices:
                    similarities = torch.matmul(norms[candidate_index], norms[chosen_indices].t())
                    diversity_bonus = float((1.0 - similarities.max().item()) * 0.12)
                rerank_score = base_score - formula_penalty - edit_penalty + diversity_bonus
                if rerank_score > best_score:
                    best_score = rerank_score
                    best_index = candidate_index
            chosen_indices.append(best_index)
            chosen = records[best_index]
            formula_counts[chosen.formula] = formula_counts.get(chosen.formula, 0) + 1
            chosen_edit = str(chosen.metadata.get("edit_action", "keep"))
            edit_counts[chosen_edit] = edit_counts.get(chosen_edit, 0) + 1
            remaining.remove(best_index)

        ordered = [records[index] for index in chosen_indices] + [records[index] for index in remaining]
        return ordered


class StructureGenerator:
    """把结构生成逻辑包装成统一入口。"""

    def __init__(self, backend: "DiffusionBackend") -> None:
        self.backend = backend

    def generate(self, num_samples: int, conditions: Dict[str, float]) -> List[MaterialRecord]:
        """委托 backend 生成候选材料。"""

        return self.backend.generate(num_samples=num_samples, conditions=conditions)
