from __future__ import annotations

"""统一评分与生成后候选选择逻辑。

这个模块把两件事分开处理：
1. `GuidanceObjective` 负责生成阶段的目标引导，告诉模型“往什么方向生成”。
2. 各类 `Scorer` 和最终选择器负责统一评分，保证 baseline 和 ours 在同一口径下比较。
"""

import math
from typing import Dict, Iterable, List, Sequence

from utils.geo_utils import MaterialRecord, normalized_hash, parse_formula


HER_ACTIVE_METALS = {
    "Pt": 0.00,
    "Mo": -0.08,
    "W": -0.12,
    "Ni": 0.10,
    "Co": 0.14,
    "Fe": 0.18,
    "V": 0.22,
}
PROMOTER_ELEMENTS = {"S": 0.05, "Se": 0.04, "Te": 0.03, "P": 0.05, "N": 0.04}
COMMON_SYNTHESIS_ELEMENTS = {
    "B",
    "C",
    "N",
    "O",
    "S",
    "Se",
    "P",
    "Mo",
    "W",
    "Ni",
    "Co",
    "Fe",
    "V",
    "Ti",
    "Nb",
    "Ta",
    "Ga",
    "In",
}
METHOD_BIAS = {
    "baseline": {"delta_g_scale": 1.08, "stability_shift": -0.03, "synthesis_shift": -0.02},
    "ours": {"delta_g_scale": 0.82, "stability_shift": 0.05, "synthesis_shift": 0.04},
}


def _weighted_average(values: Dict[str, float], weights: Dict[str, float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for key, value in values.items():
        weight = float(weights.get(key, 1.0))
        numerator += weight * float(value)
        denominator += weight
    return numerator / denominator if denominator else 0.0


class GuidanceObjective:
    """生成阶段的引导目标。

    这里强调的是“采样时如何向目标靠拢”，而不是最终展示时的统一评分。
    对 HER 来说，真正理想的是 `ΔG_H` 接近 `0 eV`，而不是数值绝对值一味更小。
    """

    def score(
        self,
        features: Dict[str, float],
        condition_vector: Dict[str, float],
        guidance_weights: Dict[str, float],
        feature_weights: Dict[str, float],
    ) -> Dict[str, float]:
        her_alignment = max(
            0.0,
            1.0 - abs(float(features["her_descriptor"]) - float(condition_vector["her_delta_g"])) / 0.65,
        )
        stability_alignment = max(
            0.0,
            1.0 - abs(float(features["stability_prior"]) - float(condition_vector["stability_score"])),
        )
        synthesis_alignment = max(
            0.0,
            1.0 - abs(float(features["synthesis_prior"]) - float(condition_vector["synthesis_score"])),
        )
        feature_prior = _weighted_average(
            values={
                "active_metal_score": float(features["active_metal_score"]),
                "promoter_score": float(features["promoter_score"]),
                "dimensional_score": float(features["dimensional_score"]),
                "complexity_score": float(features["complexity_score"]),
                "stability_prior": float(features["stability_prior"]),
                "synthesis_prior": float(features["synthesis_prior"]),
            },
            weights=feature_weights,
        )
        guidance_score = _weighted_average(
            values={
                "her_alignment": her_alignment,
                "stability": stability_alignment,
                "synthesis": synthesis_alignment,
                "two_d": float(features["dimensional_score"]),
                "feature_prior": feature_prior,
            },
            weights=guidance_weights,
        )
        return {
            "guidance_score": round(guidance_score, 4),
            "her_alignment": round(her_alignment, 4),
            "stability_alignment": round(stability_alignment, 4),
            "synthesis_alignment": round(synthesis_alignment, 4),
            "feature_prior": round(feature_prior, 4),
        }


class PropertyScorer:
    """单一属性评分器的抽象基类。"""

    metric_name = "proxy"

    def score(self, material: MaterialRecord) -> Dict[str, float]:
        raise NotImplementedError


class HERScorer(PropertyScorer):
    metric_name = "proxy_her"

    def score(self, material: MaterialRecord) -> Dict[str, float]:
        """估计候选材料的 HER 代理指标。"""

        composition = parse_formula(material.formula)
        delta_g = 0.35
        for element, contribution in HER_ACTIVE_METALS.items():
            if element in composition:
                delta_g = contribution
                break
        for element, bonus in PROMOTER_ELEMENTS.items():
            if element in composition:
                delta_g -= bonus

        delta_g += (normalized_hash(f"her:{material.material_id}") - 0.5) * 0.14
        delta_g *= float(METHOD_BIAS.get(material.method, {}).get("delta_g_scale", 1.0))
        delta_g = max(min(delta_g, 0.9), -0.9)
        her_activity = math.exp(-abs(delta_g) / 0.18)
        return {
            "her_delta_g": round(delta_g, 4),
            "her_activity_score": round(her_activity, 4),
            "metric_source": self.metric_name,
        }


class StabilityScorer(PropertyScorer):
    metric_name = "proxy_stability"

    def score(self, material: MaterialRecord) -> Dict[str, float]:
        """估计候选材料的结构稳定性代理分数。"""

        composition = parse_formula(material.formula)
        richness_penalty = max(material.num_elements - 2, 0) * 0.14
        two_d_bonus = 0.12 if material.is_2d else -0.08
        stable_bonus = 0.08 if any(key in composition for key in ("Mo", "W", "Ti", "B", "Ga", "In")) else 0.0
        jitter = (normalized_hash(f"stability:{material.material_id}") - 0.5) * 0.10
        score = 0.72 + two_d_bonus + stable_bonus - richness_penalty + jitter
        score += float(METHOD_BIAS.get(material.method, {}).get("stability_shift", 0.0))
        score = max(0.0, min(1.0, score))
        return {"stability_score": round(score, 4), "metric_source": self.metric_name}


class SynthesisScorer(PropertyScorer):
    metric_name = "proxy_synthesis"

    def score(self, material: MaterialRecord) -> Dict[str, float]:
        """估计候选材料的可合成性代理分数。"""

        composition = parse_formula(material.formula)
        common_fraction = 0.0
        if composition:
            common_fraction = sum(1 for element in composition if element in COMMON_SYNTHESIS_ELEMENTS) / len(composition)

        element_penalty = max(material.num_elements - 3, 0) * 0.18
        jitter = (normalized_hash(f"synthesis:{material.material_id}") - 0.5) * 0.08
        score = 0.55 + 0.30 * common_fraction - element_penalty + jitter
        score += float(METHOD_BIAS.get(material.method, {}).get("synthesis_shift", 0.0))
        score = max(0.0, min(1.0, score))
        return {"synthesis_score": round(score, 4), "metric_source": self.metric_name}


class CompositeScorer(PropertyScorer):
    """项目里统一用于比较 baseline 和 ours 的综合评分器。"""

    metric_name = "proxy_composite"

    def __init__(self) -> None:
        self.her = HERScorer()
        self.stability = StabilityScorer()
        self.synthesis = SynthesisScorer()

    def score(self, material: MaterialRecord) -> Dict[str, float]:
        her_result = self.her.score(material)
        stability_result = self.stability.score(material)
        synthesis_result = self.synthesis.score(material)
        total_score = (
            0.45 * her_result["her_activity_score"]
            + 0.35 * stability_result["stability_score"]
            + 0.20 * synthesis_result["synthesis_score"]
        )
        return {
            "her_delta_g": her_result["her_delta_g"],
            "stability_score": stability_result["stability_score"],
            "synthesis_score": synthesis_result["synthesis_score"],
            "total_score": round(total_score, 4),
            "metric_source": self.metric_name,
        }

    def score_many(self, materials: Iterable[MaterialRecord]) -> List[MaterialRecord]:
        scored: List[MaterialRecord] = []
        for material in materials:
            metrics = self.score(material)
            scored.append(
                material.with_scores(
                    her_delta_g=metrics["her_delta_g"],
                    stability_score=metrics["stability_score"],
                    synthesis_score=metrics["synthesis_score"],
                    total_score=metrics["total_score"],
                    metric_source=metrics["metric_source"],
                )
            )
        return scored


class MaterialGenerationAlignedPostFilter:
    """对齐 material_generation 语义的生成后筛选器。

    这里保留的是参考仓库真正有迁移价值的筛选思路，而不是它的历史脚本细节。
    核心规则包括：
    1. 候选尽量是二维 / 层状材料。
    2. 候选元素种类不要过多，避免合成难度过高。
    3. 稳定性和可合成性要先过基础阈值。
    4. 再按 HER 活性、二维性和综合质量做最终排序。

    这里不会复刻原仓库里的网页自动化、绝对路径或外部预测器，
    只保留能够稳定落到当前项目里的筛选语义。
    """

    def __init__(
        self,
        max_elements: int = 3,
        min_two_d_score: float = 0.30,
        min_stability_score: float = 0.58,
        min_synthesis_score: float = 0.55,
        max_abs_her_delta_g: float = 0.80,
    ) -> None:
        self.max_elements = max_elements
        self.min_two_d_score = min_two_d_score
        self.min_stability_score = min_stability_score
        self.min_synthesis_score = min_synthesis_score
        self.max_abs_her_delta_g = max_abs_her_delta_g

    def _two_d_score(self, material: MaterialRecord) -> float:
        return float(material.metadata.get("dimensional_score", 1.0 if material.is_2d else 0.0))

    def _passes_gate(self, material: MaterialRecord) -> bool:
        if int(material.num_elements) > self.max_elements:
            return False
        if self._two_d_score(material) < self.min_two_d_score:
            return False
        if float(material.stability_score or 0.0) < self.min_stability_score:
            return False
        if float(material.synthesis_score or 0.0) < self.min_synthesis_score:
            return False
        if abs(float(material.her_delta_g or 0.0)) > self.max_abs_her_delta_g:
            return False
        return True

    def _score(self, material: MaterialRecord) -> float:
        her_activity = math.exp(-abs(float(material.her_delta_g or 0.0)) / 0.18)
        return (
            0.25 * self._two_d_score(material)
            + 0.20 * float(material.stability_score or 0.0)
            + 0.20 * float(material.synthesis_score or 0.0)
            + 0.35 * her_activity
        )

    def select(self, materials: Sequence[MaterialRecord], top_k: int) -> List[MaterialRecord]:
        gated = [material for material in materials if self._passes_gate(material)]
        fallback = list(materials) if gated else list(materials)

        # 先保留通过基础阈值筛选的材料；如果数量不够，再从未通过的候选里补足。
        # 这样可以避免阈值过严时最终连足够的结构文件都导不出来。
        ranked = sorted(gated, key=self._score, reverse=True)
        if len(ranked) < top_k:
            used_ids = {material.material_id for material in ranked}
            filler = sorted(fallback, key=self._score, reverse=True)
            for material in filler:
                if material.material_id in used_ids:
                    continue
                ranked.append(material)
                used_ids.add(material.material_id)
                if len(ranked) >= top_k:
                    break

        deduped: List[MaterialRecord] = []
        seen_formulas: set[str] = set()
        for material in ranked:
            if material.formula in seen_formulas:
                continue
            seen_formulas.add(material.formula)
            deduped.append(material)
            if len(deduped) >= top_k:
                break

        if len(deduped) < top_k:
            used_ids = {material.material_id for material in deduped}
            for material in ranked:
                if material.material_id in used_ids:
                    continue
                deduped.append(material)
                used_ids.add(material.material_id)
                if len(deduped) >= top_k:
                    break
        return deduped


class TaskAwareCandidateSelector:
    """ours 的最终候选选择器。

    和 baseline 不同，这里不是只做硬阈值过滤，而是在满足基本可行性的前提下，
    把 HER、稳定性、可合成性、二维性和模型内部 guidance 一起纳入排序，
    同时保留 diversity reranker 已经带来的多样性。
    """

    def __init__(
        self,
        min_stability_score: float = 0.50,
        min_synthesis_score: float = 0.45,
        min_two_d_score: float = 0.25,
    ) -> None:
        self.min_stability_score = min_stability_score
        self.min_synthesis_score = min_synthesis_score
        self.min_two_d_score = min_two_d_score

    def _two_d_score(self, material: MaterialRecord) -> float:
        return float(material.metadata.get("dimensional_score", 1.0 if material.is_2d else 0.0))

    def _passes_gate(self, material: MaterialRecord) -> bool:
        return (
            float(material.stability_score or 0.0) >= self.min_stability_score
            and float(material.synthesis_score or 0.0) >= self.min_synthesis_score
            and self._two_d_score(material) >= self.min_two_d_score
        )

    def _score(self, material: MaterialRecord) -> float:
        her_activity = math.exp(-abs(float(material.her_delta_g or 0.0)) / 0.18)
        return (
            0.38 * her_activity
            + 0.22 * float(material.stability_score or 0.0)
            + 0.14 * float(material.synthesis_score or 0.0)
            + 0.12 * self._two_d_score(material)
            + 0.08 * float(material.total_score or 0.0)
            + 0.14 * float(material.guidance_score or 0.0)
        )

    def select(self, materials: Sequence[MaterialRecord], top_k: int) -> List[MaterialRecord]:
        gated = [material for material in materials if self._passes_gate(material)]
        ranked_source = gated if gated else list(materials)
        ranked = sorted(ranked_source, key=self._score, reverse=True)

        chosen: List[MaterialRecord] = []
        seen_formulas: Dict[str, int] = {}
        for material in ranked:
            # ours 允许少量相似结构进入，但避免 top-k 全部塌缩成同一化学式。
            formula_count = seen_formulas.get(material.formula, 0)
            if formula_count >= 3:
                continue
            seen_formulas[material.formula] = formula_count + 1
            chosen.append(material)
            if len(chosen) >= top_k:
                break

        if len(chosen) < top_k:
            used_ids = {material.material_id for material in chosen}
            for material in sorted(materials, key=self._score, reverse=True):
                if material.material_id in used_ids:
                    continue
                chosen.append(material)
                used_ids.add(material.material_id)
                if len(chosen) >= top_k:
                    break
        return chosen
