from __future__ import annotations

"""共享数据结构与轻量材料图工具。

本仓库里的多个模块都会复用这些函数：
公式解析、原型编辑、轻量图构建、结果表拼装，都在这里统一收口。
"""

import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import torch


FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
ACTIVE_METALS = ("Mo", "W", "Ni", "Co", "Fe", "V", "Ti", "Nb", "Ta", "Pt")
PROMOTERS = ("S", "Se", "Te", "P", "N")
ELEMENT_PROPERTIES: Dict[str, Dict[str, float]] = {
    "B": {"atomic_number": 5, "group": 13, "period": 2, "electronegativity": 2.04, "covalent_radius": 0.84, "valence_electrons": 3},
    "C": {"atomic_number": 6, "group": 14, "period": 2, "electronegativity": 2.55, "covalent_radius": 0.76, "valence_electrons": 4},
    "N": {"atomic_number": 7, "group": 15, "period": 2, "electronegativity": 3.04, "covalent_radius": 0.71, "valence_electrons": 5},
    "Na": {"atomic_number": 11, "group": 1, "period": 3, "electronegativity": 0.93, "covalent_radius": 1.66, "valence_electrons": 1},
    "Mg": {"atomic_number": 12, "group": 2, "period": 3, "electronegativity": 1.31, "covalent_radius": 1.41, "valence_electrons": 2},
    "P": {"atomic_number": 15, "group": 15, "period": 3, "electronegativity": 2.19, "covalent_radius": 1.07, "valence_electrons": 5},
    "S": {"atomic_number": 16, "group": 16, "period": 3, "electronegativity": 2.58, "covalent_radius": 1.05, "valence_electrons": 6},
    "Ti": {"atomic_number": 22, "group": 4, "period": 4, "electronegativity": 1.54, "covalent_radius": 1.60, "valence_electrons": 4},
    "V": {"atomic_number": 23, "group": 5, "period": 4, "electronegativity": 1.63, "covalent_radius": 1.53, "valence_electrons": 5},
    "Fe": {"atomic_number": 26, "group": 8, "period": 4, "electronegativity": 1.83, "covalent_radius": 1.32, "valence_electrons": 8},
    "Co": {"atomic_number": 27, "group": 9, "period": 4, "electronegativity": 1.88, "covalent_radius": 1.26, "valence_electrons": 9},
    "Ni": {"atomic_number": 28, "group": 10, "period": 4, "electronegativity": 1.91, "covalent_radius": 1.24, "valence_electrons": 10},
    "Ga": {"atomic_number": 31, "group": 13, "period": 4, "electronegativity": 1.81, "covalent_radius": 1.22, "valence_electrons": 3},
    "Se": {"atomic_number": 34, "group": 16, "period": 4, "electronegativity": 2.55, "covalent_radius": 1.20, "valence_electrons": 6},
    "Sr": {"atomic_number": 38, "group": 2, "period": 5, "electronegativity": 0.95, "covalent_radius": 1.95, "valence_electrons": 2},
    "Nb": {"atomic_number": 41, "group": 5, "period": 5, "electronegativity": 1.60, "covalent_radius": 1.64, "valence_electrons": 5},
    "Mo": {"atomic_number": 42, "group": 6, "period": 5, "electronegativity": 2.16, "covalent_radius": 1.54, "valence_electrons": 6},
    "Pd": {"atomic_number": 46, "group": 10, "period": 5, "electronegativity": 2.20, "covalent_radius": 1.39, "valence_electrons": 10},
    "In": {"atomic_number": 49, "group": 13, "period": 5, "electronegativity": 1.78, "covalent_radius": 1.42, "valence_electrons": 3},
    "Sb": {"atomic_number": 51, "group": 15, "period": 5, "electronegativity": 2.05, "covalent_radius": 1.39, "valence_electrons": 5},
    "Sn": {"atomic_number": 50, "group": 14, "period": 5, "electronegativity": 1.96, "covalent_radius": 1.39, "valence_electrons": 4},
    "Te": {"atomic_number": 52, "group": 16, "period": 5, "electronegativity": 2.10, "covalent_radius": 1.38, "valence_electrons": 6},
    "Ta": {"atomic_number": 73, "group": 5, "period": 6, "electronegativity": 1.50, "covalent_radius": 1.70, "valence_electrons": 5},
    "W": {"atomic_number": 74, "group": 6, "period": 6, "electronegativity": 2.36, "covalent_radius": 1.62, "valence_electrons": 6},
    "Os": {"atomic_number": 76, "group": 8, "period": 6, "electronegativity": 2.20, "covalent_radius": 1.44, "valence_electrons": 8},
    "Pt": {"atomic_number": 78, "group": 10, "period": 6, "electronegativity": 2.28, "covalent_radius": 1.36, "valence_electrons": 10},
    "Tl": {"atomic_number": 81, "group": 13, "period": 6, "electronegativity": 1.62, "covalent_radius": 1.45, "valence_electrons": 3},
    "Pb": {"atomic_number": 82, "group": 14, "period": 6, "electronegativity": 2.33, "covalent_radius": 1.46, "valence_electrons": 4},
    "Lu": {"atomic_number": 71, "group": 3, "period": 6, "electronegativity": 1.27, "covalent_radius": 1.87, "valence_electrons": 3},
    "Tm": {"atomic_number": 69, "group": 3, "period": 6, "electronegativity": 1.25, "covalent_radius": 1.90, "valence_electrons": 3},
}


@dataclass
class MaterialRecord:
    """统一的候选材料记录结构。"""

    material_id: str
    method: str
    cif_path: str
    formula: str
    source: str
    num_elements: int
    is_2d: bool
    her_delta_g: float | None = None
    stability_score: float | None = None
    synthesis_score: float | None = None
    total_score: float | None = None
    guidance_score: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        row = {
            "material_id": self.material_id,
            "method": self.method,
            "cif_path": self.cif_path,
            "formula": self.formula,
            "source": self.source,
            "num_elements": self.num_elements,
            "is_2d": self.is_2d,
            "her_delta_g": self.her_delta_g,
            "stability_score": self.stability_score,
            "synthesis_score": self.synthesis_score,
            "total_score": self.total_score,
            "guidance_score": self.guidance_score,
        }
        row.update(self.metadata)
        return row

    def with_scores(
        self,
        her_delta_g: float,
        stability_score: float,
        synthesis_score: float,
        total_score: float,
        metric_source: str,
    ) -> "MaterialRecord":
        return MaterialRecord(
            material_id=self.material_id,
            method=self.method,
            cif_path=self.cif_path,
            formula=self.formula,
            source=self.source,
            num_elements=self.num_elements,
            is_2d=self.is_2d,
            her_delta_g=her_delta_g,
            stability_score=stability_score,
            synthesis_score=synthesis_score,
            total_score=total_score,
            guidance_score=self.guidance_score,
            metadata={**self.metadata, "metric_source": metric_source},
        )


def stable_hash_int(value: str) -> int:
    """生成稳定哈希，避免 Python 内置 hash 的随机化差异。"""

    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def normalized_hash(value: str) -> float:
    """把稳定哈希映射到 [0, 1) 区间，用于可复现的小扰动。"""

    return (stable_hash_int(value) % 10_000) / 10_000.0


def parse_formula(formula: str) -> Dict[str, float]:
    """把化学式解析为元素到计量数的映射。"""

    composition: Dict[str, float] = {}
    for symbol, raw_count in FORMULA_TOKEN_RE.findall(formula or ""):
        composition[symbol] = composition.get(symbol, 0.0) + float(raw_count or 1.0)
    return composition


def count_elements_in_formula(formula: str) -> int:
    """统计化学式中不同元素的个数。"""

    return max(len(parse_formula(formula)), 1)


def format_formula(composition: Dict[str, float]) -> str:
    """把元素计量映射重新格式化为化学式字符串。"""

    parts: List[str] = []
    for symbol, count in composition.items():
        if count <= 0:
            continue
        rounded = round(float(count))
        if abs(float(count) - rounded) < 1.0e-6:
            count_repr = str(int(rounded))
        else:
            count_repr = f"{count:.2f}".rstrip("0").rstrip(".")
        parts.append(symbol if count_repr == "1" else f"{symbol}{count_repr}")
    return "".join(parts)


def mutate_formula(
    formula: str,
    action: str,
    preferred_metal: str = "Mo",
    preferred_promoter: str = "S",
) -> str:
    """对原型化学式做轻量编辑。

    这里的编辑动作主要服务于项目中的结构生成展示，不试图保证真实化学可行性。
    """

    composition = parse_formula(formula)
    if not composition:
        return formula

    if action == "swap_metal":
        for element in list(composition.keys()):
            if element in ACTIVE_METALS:
                count = composition.pop(element)
                composition = {preferred_metal: count, **composition}
                return format_formula(composition)
        composition = {preferred_metal: 1.0, **composition}
        return format_formula(composition)

    if action == "add_promoter":
        if not any(element in PROMOTERS for element in composition):
            composition[preferred_promoter] = max(1.0, composition.get(preferred_promoter, 0.0))
        return format_formula(composition)

    if action == "light_simplify":
        if len(composition) > 3:
            for element in list(composition.keys())[3:]:
                composition.pop(element, None)
        return format_formula(composition)

    return formula


def get_element_properties(symbol: str) -> Dict[str, float]:
    """返回轻量元素属性表中的特征。"""

    return dict(
        ELEMENT_PROPERTIES.get(
            symbol,
            {
                "atomic_number": 0.0,
                "group": 0.0,
                "period": 0.0,
                "electronegativity": 0.0,
                "covalent_radius": 1.0,
                "valence_electrons": 0.0,
            },
        )
    )


def _element_feature_vector(symbol: str, stoich_fraction: float) -> List[float]:
    props = get_element_properties(symbol)
    return [
        float(props["atomic_number"]) / 100.0,
        float(props["group"]) / 18.0,
        float(props["period"]) / 7.0,
        float(props["electronegativity"]) / 4.0,
        float(props["covalent_radius"]) / 2.5,
        float(props["valence_electrons"]) / 10.0,
        float(stoich_fraction),
    ]


def _complete_graph(num_nodes: int, edge_value: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_nodes <= 1:
        return torch.zeros((2, 1), dtype=torch.long), torch.tensor([edge_value], dtype=torch.float32)

    edge_pairs: List[List[int]] = []
    edge_features: List[List[float]] = []
    for source in range(num_nodes):
        for target in range(num_nodes):
            if source == target:
                continue
            edge_pairs.append([source, target])
            edge_features.append(edge_value)
    return torch.tensor(edge_pairs, dtype=torch.long).t().contiguous(), torch.tensor(edge_features, dtype=torch.float32)


def build_formula_graph(formula: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把化学式构造成简化图表示。

    当没有真实结构时，模型会退回到这种公式图表示。
    """

    composition = parse_formula(formula)
    total_atoms = max(sum(composition.values()), 1.0)
    x = torch.tensor(
        [_element_feature_vector(symbol, float(count) / total_atoms) for symbol, count in composition.items()],
        dtype=torch.float32,
    )
    edge_index, edge_attr = _complete_graph(x.size(0), [0.5, 0.0, 1.0])
    return x, edge_index, edge_attr


def ensure_directory(path: Path) -> Path:
    """确保目录存在并返回该目录。"""

    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_directory(path: Path) -> Path:
    """清空输出目录，确保每次运行得到干净的结果。"""

    if path.exists():
        for item in path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_path(path: Path, root: Path) -> str:
    """把绝对路径转成相对于仓库根的路径字符串。"""

    return Path(os.path.relpath(path, root)).as_posix()


def write_pseudo_cif(record: MaterialRecord, output_path: Path) -> Path:
    """为候选材料写出轻量 CIF 文件。"""

    composition = parse_formula(record.formula)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"data_{record.material_id}",
        "_symmetry_space_group_name_H-M    'P1'",
        "_symmetry_Int_Tables_number       1",
        "_cell_length_a                    3.2000",
        "_cell_length_b                    3.2000",
        f"_cell_length_c                    {'18.0000' if record.is_2d else '6.4000'}",
        "_cell_angle_alpha                 90.0000",
        "_cell_angle_beta                  90.0000",
        "_cell_angle_gamma                 120.0000",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]

    site_index = 1
    for symbol, count in composition.items():
        multiplicity = max(1, int(round(float(count))))
        for atom_index in range(multiplicity):
            frac_x = (0.17 * site_index) % 1.0
            frac_y = (0.31 * site_index) % 1.0
            frac_z = ((0.50 if record.is_2d else 0.18) + 0.07 * atom_index) % 1.0
            lines.append(f"{symbol}{site_index} {symbol} {frac_x:.4f} {frac_y:.4f} {frac_z:.4f}")
            site_index += 1

    lines.extend(
        [
            "",
            f"# formula = {record.formula}",
            f"# method = {record.method}",
            f"# source = {record.source}",
            f"# her_delta_g = {record.her_delta_g if record.her_delta_g is not None else 'n/a'}",
            f"# stability_score = {record.stability_score if record.stability_score is not None else 'n/a'}",
            f"# synthesis_score = {record.synthesis_score if record.synthesis_score is not None else 'n/a'}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def assemble_results_frame(records: Iterable[MaterialRecord]) -> pd.DataFrame:
    """把候选材料记录列表转换成结果表。"""

    return pd.DataFrame([record.to_dict() for record in records])
