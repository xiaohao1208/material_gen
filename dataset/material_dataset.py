from __future__ import annotations

"""数据集入口与 Materials Project 缓存处理。

这个模块同时支持两条路径：
1. builtin：内置的小规模模板集，用于快速验证代码链路
2. materials_project：从本地缓存的 MP 原始数据构建正式训练子集
"""

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import requests
import torch
from torch.utils.data import Dataset

from utils.geo_utils import (
    MaterialRecord,
    build_formula_graph,
    count_elements_in_formula,
    ensure_directory,
    get_element_properties,
    mutate_formula,
    normalized_hash,
    parse_formula,
)


TARGET_ORDER = ("her_target", "stability_target", "synthesis_target", "is_2d_target")
DEFAULT_MP_CACHE_DIR = "dataset/materials_project"
MP_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "energy_above_hull",
    "formation_energy_per_atom",
    "band_gap",
    "nsites",
    "nelements",
    "theoretical",
]
TRANSITION_METALS = {"Mo", "W", "Ni", "Co", "Fe", "V", "Ti", "Nb", "Ta"}
PROMOTER_ELEMENTS = {"S", "Se", "Te", "P", "N"}


@dataclass(frozen=True)
class BuiltinTemplate:
    """内置模板只用于快速验证，不替代正式 MP 数据训练。"""

    material_id: str
    formula: str
    source: str
    is_2d: bool
    note: str


BUILTIN_TEMPLATES: List[BuiltinTemplate] = [
    BuiltinTemplate("template_0001", "MoS2", "reference", True, "典型二维 HER 候选"),
    BuiltinTemplate("template_0002", "WS2", "reference", True, "二维过渡金属硫化物"),
    BuiltinTemplate("template_0003", "WSe2", "reference", True, "二维过渡金属硒化物"),
    BuiltinTemplate("template_0004", "VS2", "reference", True, "层状金属硫化物"),
    BuiltinTemplate("template_0005", "TiC2", "reference", True, "MXene 风格碳化物"),
    BuiltinTemplate("template_0006", "NbSe2", "reference", True, "层状硒化物"),
    BuiltinTemplate("template_0007", "TaS2", "reference", True, "稳定二维硫化物"),
    BuiltinTemplate("template_0008", "MoTe2", "reference", True, "碲化物二维候选"),
    BuiltinTemplate("template_0009", "GaSe", "reference", True, "二维半导体"),
    BuiltinTemplate("template_0010", "InSe", "reference", True, "二维半导体"),
    BuiltinTemplate("template_0011", "BN", "reference", True, "简单二维绝缘体"),
    BuiltinTemplate("template_0012", "Ti3C2", "reference", True, "MXene 族材料"),
    BuiltinTemplate("template_0013", "NiPS3", "reference", True, "层状三元材料"),
    BuiltinTemplate("template_0014", "CoPS3", "reference", True, "层状三元材料"),
    BuiltinTemplate("template_0015", "FePS3", "reference", True, "层状三元材料"),
    BuiltinTemplate("template_0016", "MgNiSb", "baseline_pool", False, "普通三元金属间化合物"),
    BuiltinTemplate("template_0017", "NaSrSb", "baseline_pool", False, "普通非二维候选"),
    BuiltinTemplate("template_0018", "TmLuOsPd", "baseline_pool", False, "高复杂度多元候选"),
    BuiltinTemplate("template_0019", "Tl2In4SnPb5", "baseline_pool", False, "多元合金候选"),
    BuiltinTemplate("template_0020", "FeSe", "reference", True, "简单活性二维候选"),
]


def _mp_paths(cache_dir: Path) -> Dict[str, Path]:
    raw_dir = cache_dir / "raw"
    structures_dir = raw_dir / "structures"
    processed_dir = cache_dir / "processed"
    return {
        "cache_dir": cache_dir,
        "raw_dir": raw_dir,
        "structures_dir": structures_dir,
        "processed_dir": processed_dir,
        "documents": raw_dir / "documents.jsonl",
        "raw_metadata": raw_dir / "metadata.json",
        "processed_metadata": processed_dir / "metadata.json",
    }


def _require_mp_api_key() -> str:
    # 只有在“需要补抓 Materials Project 数据”时才要求 key。
    # 如果本地已经有可用缓存，默认训练和测试不应该强制依赖 MP_API_KEY。
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY is not set in the current shell.")
    return api_key


def _normalise_doc(item: object) -> Dict:
    if hasattr(item, "model_dump"):
        return dict(item.model_dump())
    if hasattr(item, "dict"):
        return dict(item.dict())
    if isinstance(item, dict):
        return dict(item)
    return dict(vars(item))


def _save_structure(payload: Dict, structures_dir: Path, material_id: str) -> str:
    ensure_directory(structures_dir)
    structure_path = structures_dir / f"{material_id}.json"
    structure_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return structure_path.name


def _mp_query_plans(limit: int) -> List[Tuple[str, Dict, int]]:
    """按任务相关性拆分下载计划，避免默认分页拉回大量无关材料。"""

    metal_budget = max(220, int(limit * 0.46 / len(TRANSITION_METALS)))
    promoter_budget = max(120, int(limit * 0.22 / len(PROMOTER_ELEMENTS)))
    consumed = metal_budget * len(TRANSITION_METALS) + promoter_budget * len(PROMOTER_ELEMENTS)
    remaining = max(limit - consumed, 1200)

    plans: List[Tuple[str, Dict, int]] = []
    for metal in sorted(TRANSITION_METALS):
        plans.append(
            (
                f"metal_{metal}",
                {
                    "elements": [metal],
                    "num_elements": (2, 4),
                    "num_sites": (1, 40),
                    "deprecated": False,
                },
                metal_budget,
            )
        )
    for promoter in sorted(PROMOTER_ELEMENTS):
        plans.append(
            (
                f"promoter_{promoter}",
                {
                    "elements": [promoter],
                    "num_elements": (2, 4),
                    "num_sites": (1, 40),
                    "energy_above_hull": (0.0, 0.30),
                    "deprecated": False,
                },
                promoter_budget,
            )
        )

    plans.append(
        (
            "stable_controls",
            {
                "num_elements": (2, 4),
                "num_sites": (1, 40),
                "energy_above_hull": (0.0, 0.12),
                "is_stable": True,
                "deprecated": False,
            },
            remaining // 2,
        )
    )
    plans.append(
        (
            "broad_fill",
            {
                "num_elements": (2, 4),
                "num_sites": (1, 40),
                "energy_above_hull": (0.0, 0.40),
                "deprecated": False,
            },
            remaining - remaining // 2,
        )
    )
    return plans


def _rest_params_from_query(query: Dict, page_limit: int, skip: int) -> Dict[str, object]:
    alias_map = {"num_elements": "nelements", "num_sites": "nsites"}
    params: Dict[str, object] = {
        "_fields": ",".join(MP_FIELDS),
        "_limit": page_limit,
        "_skip": skip,
    }
    for key, value in query.items():
        query_key = alias_map.get(key, key)
        if isinstance(value, tuple):
            params[f"{query_key}_min"] = value[0]
            params[f"{query_key}_max"] = value[1]
        elif isinstance(value, list):
            params[query_key] = ",".join(str(item) for item in value)
        elif isinstance(value, bool):
            params[query_key] = str(value).lower()
        else:
            params[query_key] = value
    return params


def _download_materials_project_via_mp_api(cache_dir: Path, limit: int, chunk_size: int = 250) -> int:
    try:
        from mp_api.client import MPRester  # type: ignore
    except Exception as exc:  # pragma: no cover - fallback path is expected on broken envs
        raise RuntimeError(f"mp-api client is unavailable: {exc}") from exc

    api_key = _require_mp_api_key()
    paths = _mp_paths(cache_dir)
    ensure_directory(paths["raw_dir"])
    ensure_directory(paths["structures_dir"])

    rows: List[Dict] = []
    seen_ids: set[str] = set()
    with MPRester(api_key=api_key, use_document_model=False, mute_progress_bars=True) as rester:
        for _, query, target_count in _mp_query_plans(limit):
            docs = rester.materials.summary.search(
                all_fields=False,
                fields=MP_FIELDS,
                chunk_size=min(chunk_size, target_count),
                num_chunks=math.ceil(target_count / chunk_size),
                **query,
            )
            for doc in docs:
                if len(rows) >= limit:
                    break
                item = _normalise_doc(doc)
                material_id = str(item["material_id"])
                if material_id in seen_ids:
                    continue
                seen_ids.add(material_id)
                structure = item.pop("structure", None)
                if structure is not None:
                    structure_payload = structure.as_dict() if hasattr(structure, "as_dict") else dict(structure)
                    item["structure_file"] = _save_structure(structure_payload, paths["structures_dir"], material_id)
                rows.append(item)
            if len(rows) >= limit:
                break

    paths["documents"].write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    paths["raw_metadata"].write_text(
        json.dumps({"source": "mp-api", "num_records": len(rows), "limit": limit}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(rows)


def _download_materials_project_via_rest(cache_dir: Path, limit: int, chunk_size: int = 200) -> int:
    api_key = _require_mp_api_key()
    paths = _mp_paths(cache_dir)
    ensure_directory(paths["raw_dir"])
    ensure_directory(paths["structures_dir"])

    session = requests.Session()
    session.headers.update({"x-api-key": api_key, "user-agent": "material-gen/1.0"})
    endpoint = "https://api.materialsproject.org/materials/summary/"

    rows: List[Dict] = []
    seen_ids: set[str] = set()
    for _, query, target_count in _mp_query_plans(limit):
        skip = 0
        collected_for_plan = 0
        while len(rows) < limit and collected_for_plan < target_count:
            page_limit = min(chunk_size, target_count - collected_for_plan, limit - len(rows))
            params = _rest_params_from_query(query=query, page_limit=page_limit, skip=skip)
            response = session.get(endpoint, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            docs = payload.get("data", [])
            if not docs:
                break

            new_count = 0
            for doc in docs:
                item = _normalise_doc(doc)
                material_id = str(item["material_id"])
                if material_id in seen_ids:
                    continue
                seen_ids.add(material_id)
                structure = item.pop("structure", None)
                if structure is not None:
                    item["structure_file"] = _save_structure(structure, paths["structures_dir"], material_id)
                rows.append(item)
                collected_for_plan += 1
                new_count += 1
                if len(rows) >= limit or collected_for_plan >= target_count:
                    break

            if not new_count:
                break
            skip += len(docs)
            time.sleep(0.15)
        if len(rows) >= limit:
            break

    paths["documents"].write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    paths["raw_metadata"].write_text(
        json.dumps({"source": "rest", "num_records": len(rows), "limit": limit}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(rows)


def download_materials_project(cache_dir: Path, limit: int = 6000, force_refresh: bool = False) -> Path:
    """优先使用 mp-api，失败则直接访问 REST 接口。

    密钥只从环境变量读取，不写入任何仓库文件。
    """

    paths = _mp_paths(cache_dir)
    if paths["documents"].exists() and not force_refresh:
        line_count = sum(1 for _ in paths["documents"].read_text(encoding="utf-8").splitlines() if _.strip())
        if line_count >= limit:
            return paths["documents"]

    try:
        _download_materials_project_via_mp_api(cache_dir=cache_dir, limit=limit)
    except Exception:
        _download_materials_project_via_rest(cache_dir=cache_dir, limit=limit)
    return paths["documents"]


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _proxy_her_target(formula: str, metadata: Dict) -> float:
    composition = parse_formula(formula)
    target = 0.30
    if "Mo" in composition:
        target = -0.03
    elif "W" in composition:
        target = -0.05
    elif "Ni" in composition:
        target = 0.08
    elif "Co" in composition:
        target = 0.12
    elif "Fe" in composition:
        target = 0.16
    elif "V" in composition:
        target = 0.20

    for promoter, bonus in {"S": 0.05, "Se": 0.04, "Te": 0.03, "P": 0.04, "N": 0.04}.items():
        if promoter in composition:
            target -= bonus

    anisotropy = float(metadata.get("lattice_anisotropy", 0.0))
    target -= 0.03 * anisotropy
    return max(min(target, 0.8), -0.8)


def _proxy_is_2d(formula: str, metadata: Dict) -> float:
    anisotropy = float(metadata.get("lattice_anisotropy", 0.0))
    composition = parse_formula(formula)
    layered_bonus = 0.18 if any(symbol in composition for symbol in {"S", "Se", "Te", "N", "C", "B", "P"}) else 0.0
    transition_bonus = 0.12 if any(symbol in composition for symbol in TRANSITION_METALS) else 0.0
    score = 0.20 + 0.55 * anisotropy + layered_bonus + transition_bonus
    return max(0.0, min(1.0, score))


def _proxy_stability_target(metadata: Dict) -> float:
    e_hull = float(metadata.get("energy_above_hull", 0.0) or 0.0)
    formation = float(metadata.get("formation_energy_per_atom", -1.0) or -1.0)
    hull_term = max(0.0, 1.0 - e_hull / 0.25)
    formation_term = max(0.0, min(1.0, (-formation + 0.5) / 2.5))
    return max(0.0, min(1.0, 0.65 * hull_term + 0.35 * formation_term))


def _proxy_synthesis_target(formula: str, metadata: Dict) -> float:
    complexity = count_elements_in_formula(formula)
    e_hull = float(metadata.get("energy_above_hull", 0.0) or 0.0)
    band_gap = float(metadata.get("band_gap", 0.0) or 0.0)
    theoretical = 1.0 if metadata.get("theoretical") else 0.0
    easy_band_gap = 1.0 - min(abs(band_gap - 1.0) / 2.5, 1.0)
    low_hull = max(0.0, 1.0 - e_hull / 0.20)
    complexity_penalty = max(0.0, (complexity - 2) * 0.10)
    score = 0.50 * low_hull + 0.25 * easy_band_gap + 0.15 * (1.0 - theoretical) + 0.10 * (1.0 - complexity_penalty)
    return max(0.0, min(1.0, score))


def _build_structure_graph(structure) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    node_features: List[List[float]] = []
    for site in structure:
        props = get_element_properties(site.specie.symbol)
        node_features.append(
            [
                float(props["atomic_number"]) / 100.0,
                float(props["group"]) / 18.0,
                float(props["period"]) / 7.0,
                float(props["electronegativity"]) / 4.0,
                float(props["covalent_radius"]) / 2.5,
                float(props["valence_electrons"]) / 10.0,
                1.0 / max(len(structure), 1),
            ]
        )

    neighbors = structure.get_neighbor_list(4.5)
    center_indices = neighbors[0].tolist() if hasattr(neighbors[0], "tolist") else list(neighbors[0])
    point_indices = neighbors[1].tolist() if hasattr(neighbors[1], "tolist") else list(neighbors[1])
    offsets = neighbors[2]
    distances = neighbors[3].tolist() if hasattr(neighbors[3], "tolist") else list(neighbors[3])

    edge_pairs: List[List[int]] = []
    edge_features: List[List[float]] = []
    for source, target, image, distance in zip(center_indices, point_indices, offsets, distances):
        image_norm = float(sum(abs(int(value)) for value in image))
        edge_pairs.append([int(source), int(target)])
        edge_features.append([float(distance) / 6.0, min(image_norm / 3.0, 1.0), 1.0 / (1.0 + float(distance))])

    if not edge_pairs:
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index, edge_attr = build_formula_graph(structure.composition.reduced_formula)[1:]
    else:
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    lengths = sorted(structure.lattice.abc)
    planar = max(lengths[0], 1.0e-6)
    anisotropy = max(0.0, min(1.0, (lengths[-1] / max(lengths[1], planar) - 1.0) / 4.0))
    return x, edge_index, edge_attr, float(anisotropy)


def _load_structure(raw_dir: Path, structure_file: str | None):
    if not structure_file:
        return None
    structure_path = raw_dir / "structures" / structure_file
    if not structure_path.exists():
        return None
    from pymatgen.core import Structure  # type: ignore

    return Structure.from_dict(json.loads(structure_path.read_text(encoding="utf-8")))


def _document_to_sample(doc: Dict, raw_dir: Path) -> Dict | None:
    formula = str(doc.get("formula_pretty") or "").strip()
    material_id = str(doc.get("material_id") or "").strip()
    if not formula or not material_id:
        return None

    metadata = {
        "energy_above_hull": float(doc.get("energy_above_hull", 0.0) or 0.0),
        "formation_energy_per_atom": float(doc.get("formation_energy_per_atom", -1.0) or -1.0),
        "band_gap": float(doc.get("band_gap", 0.0) or 0.0),
        "theoretical": bool(doc.get("theoretical", False)),
    }
    structure = _load_structure(raw_dir, doc.get("structure_file"))
    if structure is not None:
        x, edge_index, edge_attr, anisotropy = _build_structure_graph(structure)
        metadata["lattice_anisotropy"] = anisotropy
    else:
        x, edge_index, edge_attr = build_formula_graph(formula)
        metadata["lattice_anisotropy"] = 0.25 + normalized_hash(f"anisotropy:{formula}") * 0.25

    is_2d_target = _proxy_is_2d(formula, metadata)
    metadata["is_2d"] = is_2d_target >= 0.58
    sample = {
        "material_id": material_id,
        "formula": formula,
        "source": "materials_project",
        "cif_path": f"dataset/materials_project/raw/structures/{material_id}.json",
        "is_2d": bool(metadata["is_2d"]),
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "targets": {
            "her_target": _proxy_her_target(formula, metadata),
            "stability_target": _proxy_stability_target(metadata),
            "synthesis_target": _proxy_synthesis_target(formula, metadata),
            "is_2d_target": is_2d_target,
        },
        "metadata": metadata,
    }
    return sample


def _priority_score(sample: Dict) -> float:
    composition = parse_formula(sample["formula"])
    has_transition = 1.0 if any(symbol in composition for symbol in TRANSITION_METALS) else 0.0
    has_promoter = 1.0 if any(symbol in composition for symbol in PROMOTER_ELEMENTS) else 0.0
    two_d = float(sample["targets"]["is_2d_target"])
    stability = float(sample["targets"]["stability_target"])
    her = math.exp(-abs(float(sample["targets"]["her_target"])) / 0.18)
    complexity_penalty = max(int(sample["metadata"].get("nelements", len(composition))) - 3, 0) * 0.08
    return 0.26 * has_transition + 0.20 * has_promoter + 0.22 * two_d + 0.20 * stability + 0.12 * her - complexity_penalty


def _select_task_subset(samples: List[Dict], min_processed: int = 3600, max_processed: int = 4800) -> List[Dict]:
    ranked = sorted(samples, key=_priority_score, reverse=True)
    if len(ranked) <= max_processed:
        return ranked

    layered = [sample for sample in ranked if float(sample["targets"]["is_2d_target"]) >= 0.58]
    controls = [sample for sample in ranked if float(sample["targets"]["is_2d_target"]) < 0.58]

    target_total = min(max_processed, len(ranked))
    layered_target = min(len(layered), int(target_total * 0.78))
    selected = layered[:layered_target]

    controls_sorted = sorted(
        controls,
        key=lambda sample: (
            float(sample["targets"]["stability_target"]),
            -float(sample["targets"]["is_2d_target"]),
            abs(float(sample["targets"]["her_target"])),
        ),
        reverse=True,
    )
    selected.extend(controls_sorted[: target_total - len(selected)])

    if len(selected) < min_processed:
        used_ids = {sample["material_id"] for sample in selected}
        for sample in ranked:
            if sample["material_id"] in used_ids:
                continue
            selected.append(sample)
            used_ids.add(sample["material_id"])
            if len(selected) >= min(min_processed, len(ranked)):
                break

    deduped: Dict[str, Dict] = {}
    for sample in selected:
        deduped[sample["material_id"]] = sample
    return list(deduped.values())


def _split_samples(samples: List[Dict]) -> Dict[str, List[Dict]]:
    ranked = sorted(samples, key=lambda item: item["material_id"])
    total = len(ranked)
    train_end = max(1, int(total * 0.80))
    val_end = min(total, train_end + max(1, int(total * 0.10)))
    return {
        "train": ranked[:train_end],
        "val": ranked[train_end:val_end] or ranked[:1],
        "test": ranked[val_end:] or ranked[-max(1, total // 5) :],
    }


def preprocess_materials_project(
    cache_dir: Path,
    min_processed: int = 3600,
    max_processed: int = 4800,
) -> Dict[str, int]:
    """把 MP 原始缓存处理成训练/验证/测试三份张量数据。

    这里会先计算任务 proxy，再按 HER 二维材料任务偏好挑出正式训练子集。
    """

    paths = _mp_paths(cache_dir)
    if not paths["documents"].exists():
        raise FileNotFoundError(f"Missing raw MP cache: {paths['documents']}")

    raw_rows = _read_jsonl(paths["documents"])
    samples: List[Dict] = []
    for row in raw_rows:
        sample = _document_to_sample(row, paths["raw_dir"])
        if sample is None:
            continue
        sample["metadata"]["nelements"] = int(row.get("nelements", 0) or 0)
        sample["metadata"]["nsites"] = int(row.get("nsites", 0) or 0)
        samples.append(sample)

    selected = _select_task_subset(samples, min_processed=min_processed, max_processed=max_processed)
    splits = _split_samples(selected)
    ensure_directory(paths["processed_dir"])
    for split_name, split_samples in splits.items():
        torch.save({"samples": split_samples, "metadata": {"source": "materials_project", "split": split_name, "num_samples": len(split_samples)}}, paths["processed_dir"] / f"{split_name}.pt")

    metadata = {
        "source": "materials_project",
        "raw_count": len(raw_rows),
        "selected_count": len(selected),
        "split_counts": {split: len(items) for split, items in splits.items()},
        "min_processed": min_processed,
        "max_processed": max_processed,
    }
    paths["processed_metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {split: len(items) for split, items in splits.items()}


def ensure_materials_project_dataset(
    cache_dir: Path,
    limit: int = 6000,
    download_if_missing: bool = False,
    force_refresh: bool = False,
) -> Path:
    """确保 MP 处理后数据存在，不存在时按需要下载并预处理。"""

    paths = _mp_paths(cache_dir)
    required = [paths["processed_dir"] / "train.pt", paths["processed_dir"] / "val.pt", paths["processed_dir"] / "test.pt"]
    processed_ready = all(path.exists() for path in required)
    processed_metadata = {}
    raw_metadata = {}
    if paths["processed_metadata"].exists():
        processed_metadata = json.loads(paths["processed_metadata"].read_text(encoding="utf-8"))
    if paths["raw_metadata"].exists():
        raw_metadata = json.loads(paths["raw_metadata"].read_text(encoding="utf-8"))

    min_processed = max(3000, int(limit * 0.60))
    max_processed = max(min_processed, int(limit * 0.80))
    processed_large_enough = int(processed_metadata.get("selected_count", 0) or 0) >= min_processed
    raw_large_enough = int(raw_metadata.get("num_records", 0) or 0) >= int(limit)-5
    if processed_ready and processed_large_enough and raw_large_enough and not force_refresh:
        return paths["processed_dir"]

    if processed_ready and not force_refresh and (not processed_large_enough or not raw_large_enough):
        # 默认优先保证“当前缓存能直接跑”。
        # 只有用户显式要求更大的 mp_limit，才要求下载补齐更大规模数据。
        if not download_if_missing or not os.environ.get("MP_API_KEY"):
            raise RuntimeError(
                "Existing Materials Project cache is smaller than the requested training scale. "
                f"Current request expects at least {limit} raw records and {min_processed} processed samples. "
                "Set MP_API_KEY and pass --download-if-missing, or run with a smaller validated --mp-limit."
            )

    if not paths["documents"].exists() or force_refresh:
        # 当前仓库不把网络下载当成默认路径。
        # 如果用户本地没有缓存，又没显式允许下载，这里就应该明确失败并给出提示。
        if not download_if_missing:
            raise FileNotFoundError(
                "Materials Project cache is missing. Pass --download-if-missing and set MP_API_KEY in the current shell."
            )
        download_materials_project(cache_dir=cache_dir, limit=limit, force_refresh=force_refresh)

    preprocess_materials_project(cache_dir=cache_dir, min_processed=min_processed, max_processed=max_processed)
    return paths["processed_dir"]


def _record_to_sample(record: MaterialRecord) -> Dict:
    x, edge_index, edge_attr = build_formula_graph(record.formula)
    targets = {
        "her_target": _proxy_her_target(record.formula, {"lattice_anisotropy": 1.0 if record.is_2d else 0.0}),
        "stability_target": max(0.0, min(1.0, 0.72 + (0.12 if record.is_2d else -0.05))),
        "synthesis_target": max(0.0, min(1.0, 0.62 + (0.10 if record.num_elements <= 3 else -0.06))),
        "is_2d_target": 1.0 if record.is_2d else 0.0,
    }
    return {
        "material_id": record.material_id,
        "formula": record.formula,
        "source": record.source,
        "cif_path": record.cif_path,
        "is_2d": record.is_2d,
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "targets": targets,
        "metadata": dict(record.metadata),
    }


def _expand_records(records: Iterable[MaterialRecord]) -> List[MaterialRecord]:
    expanded: List[MaterialRecord] = []
    for record in records:
        expanded.append(record)
        for action in ("swap_metal", "add_promoter", "light_simplify"):
            formula = mutate_formula(record.formula, action=action, preferred_metal="Mo", preferred_promoter="S")
            expanded.append(
                MaterialRecord(
                    material_id=f"{record.material_id}_{action}",
                    method="template",
                    cif_path=f"templates/{record.material_id}_{action}.cif",
                    formula=formula,
                    source=record.source,
                    num_elements=count_elements_in_formula(formula),
                    is_2d=record.is_2d,
                    metadata={**record.metadata, "derived_from": record.material_id, "edit_action": action},
                )
            )
    deduped: Dict[str, MaterialRecord] = {}
    for record in expanded:
        deduped[record.material_id] = record
    return list(deduped.values())


class MaterialDataset:
    """builtin 模式下的自包含模板数据库。"""

    def __init__(self, records: List[MaterialRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[MaterialRecord]:
        return iter(self.records)

    @classmethod
    def discover(
        cls,
        max_records: Optional[int] = None,
        source_names: Optional[Iterable[str]] = None,
        include_derived: bool = True,
    ) -> "MaterialDataset":
        selected_sources = set(source_names or [template.source for template in BUILTIN_TEMPLATES])
        records: List[MaterialRecord] = []
        for template in BUILTIN_TEMPLATES:
            if template.source not in selected_sources:
                continue
            records.append(
                MaterialRecord(
                    material_id=template.material_id,
                    method="template",
                    cif_path=f"templates/{template.material_id}.cif",
                    formula=template.formula,
                    source=template.source,
                    num_elements=count_elements_in_formula(template.formula),
                    is_2d=template.is_2d,
                    metadata={"template_note": template.note},
                )
            )
        if include_derived:
            records = _expand_records(records)
        if max_records is not None:
            records = records[:max_records]
        return cls(records=records)

    def split(self) -> Dict[str, List[MaterialRecord]]:
        ordered = sorted(self.records, key=lambda record: record.material_id)
        total = len(ordered)
        train_end = max(1, int(total * 0.80))
        val_end = min(total, train_end + max(1, int(total * 0.10)))
        return {
            "train": ordered[:train_end],
            "val": ordered[train_end:val_end] or ordered[:1],
            "test": ordered[val_end:] or ordered[-max(1, total // 5) :],
        }


class MaterialGraphDataset(Dataset):
    """支持 builtin 和 materials_project 双数据源。"""

    def __init__(
        self,
        split: str,
        data_source: str = "builtin",
        mp_cache_dir: Path | None = None,
        mp_limit: int = 6000,
        download_if_missing: bool = False,
    ) -> None:
        self.split = split
        self.data_source = data_source

        if data_source == "builtin":
            dataset = MaterialDataset.discover()
            splits = dataset.split()
            if split not in splits:
                raise ValueError(f"Unsupported split: {split}")
            self.records = splits[split]
            self.samples = [_record_to_sample(record) for record in self.records]
            self.metadata = {"source": "builtin_material_templates", "split": split, "num_samples": len(self.samples)}
        elif data_source == "materials_project":
            cache_dir = Path(mp_cache_dir or DEFAULT_MP_CACHE_DIR)
            # materials_project 模式会优先使用本地 processed 缓存。
            # 这让默认训练命令在已有缓存时能直接启动，而不是每次都重新准备数据。
            processed_dir = ensure_materials_project_dataset(
                cache_dir=cache_dir,
                limit=mp_limit,
                download_if_missing=download_if_missing,
            )
            split_path = processed_dir / f"{split}.pt"
            if not split_path.exists():
                raise FileNotFoundError(f"Missing processed Materials Project split: {split_path}")
            payload = torch.load(split_path, map_location="cpu")
            self.samples = list(payload.get("samples", []))
            self.records = []
            self.metadata = dict(payload.get("metadata", {}))
        else:
            raise ValueError(f"Unsupported data source: {data_source}")

        if not self.samples:
            raise RuntimeError(f"Split {split} is empty for data source {data_source}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        return self.samples[index]


def collate_graph_samples(samples: List[Dict]) -> Dict[str, torch.Tensor | List[Dict]]:
    """把单个样本拼成训练 batch。

    这个函数负责把变长图样本整理成统一张量，并保留公式、来源等可解释元数据。
    """

    x_parts: List[torch.Tensor] = []
    edge_index_parts: List[torch.Tensor] = []
    edge_attr_parts: List[torch.Tensor] = []
    batch_parts: List[torch.Tensor] = []
    targets: List[List[float]] = []
    records: List[Dict] = []

    node_offset = 0
    for graph_index, sample in enumerate(samples):
        x = sample["x"].float()
        edge_index = sample["edge_index"].long()
        edge_attr = sample["edge_attr"].float()

        x_parts.append(x)
        edge_index_parts.append(edge_index + node_offset)
        edge_attr_parts.append(edge_attr)
        batch_parts.append(torch.full((x.size(0),), graph_index, dtype=torch.long))
        node_offset += x.size(0)

        targets.append([float(sample["targets"][key]) for key in TARGET_ORDER])
        records.append(
            {
                "material_id": sample["material_id"],
                "formula": sample["formula"],
                "source": sample["source"],
                "cif_path": sample["cif_path"],
                "is_2d": bool(sample["is_2d"]),
                "metadata": dict(sample.get("metadata", {})),
            }
        )

    return {
        "x": torch.cat(x_parts, dim=0),
        "edge_index": torch.cat(edge_index_parts, dim=1),
        "edge_attr": torch.cat(edge_attr_parts, dim=0),
        "batch": torch.cat(batch_parts, dim=0),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "records": records,
    }
