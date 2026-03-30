from __future__ import annotations

from matplotlib.ticker import MultipleLocator

"""项目绘图工具。"""

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METHOD_COLORS = {"baseline": "#E07A5F", "ours": "#1D4E89"}
METHOD_FILLS = {"baseline": "#FFF1EA", "ours": "#EAF3FF"}
METHOD_LABELS = {"baseline": "baseline", "ours": "Ours"}


def _prepare_output(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def _metric_to_float(value) -> float:
    if isinstance(value, str):
        value = value.strip().lstrip("↑↓")
    if isinstance(value, str) and value.endswith("%"):
        return float(value.rstrip("%")) / 100.0
    return float(value)


def _smooth(values: List[float], window: int = 3) -> List[float]:
    if len(values) < 3:
        return values
    smoothed: List[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def _style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, edge_color: str, face_color: str = "#F9FAFB") -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.03",
        linewidth=2,
        edgecolor=edge_color,
        facecolor=face_color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def _draw_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color="#334155")
    ax.add_patch(arrow)


def plot_loss_curve(histories: Dict[str, List[dict]], output_path: Path) -> None:
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for method, history in histories.items():
        if not history:
            continue
        epochs = [entry["epoch"] for entry in history]
        losses = [entry["train_loss"] for entry in history]
        # smooth_losses = _smooth(losses, window=4)
        # best_epoch = min(history, key=lambda entry: entry["val_loss"])["epoch"]
        # best_value = min(entry["val_loss"] for entry in history)
        ax.plot(epochs, losses, linewidth=1.2, color=METHOD_COLORS[method], label=f"{METHOD_LABELS[method]}")
        # ax.plot(epochs, smooth_losses, linewidth=2.8, color=METHOD_COLORS[method], label=f"{METHOD_LABELS[method]} (smoothed)")
        # ax.scatter([best_epoch], [best_value], color=METHOD_COLORS[method], s=70, zorder=5)
        # ax.annotate(
        #     f"best@{best_epoch}",
        #     xy=(best_epoch, best_value),
        #     xytext=(best_epoch, best_value + 0.12),
        #     arrowprops={"arrowstyle": "->", "color": METHOD_COLORS[method]},
        #     fontsize=9,
        #     color=METHOD_COLORS[method],
        # )
    _style_axis(ax, "Training Loss Curve", "Epoch", "Loss")
    ax.legend()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_model_architecture(output_path: Path) -> None:
    _prepare_output(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.8), constrained_layout=True)

    left = axes[0]
    left.axis("off")
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.set_title("Model Architecture", fontsize=14, fontweight="bold", pad=10)

    _draw_box(left, 0.05, 0.83, 0.22, 0.10, "Materials Dataset", "#6C8EBF")
    _draw_box(left, 0.36, 0.83, 0.22, 0.10, "Graph Representation", "#6C8EBF")
    _draw_box(left, 0.67, 0.83, 0.22, 0.10, "Unified Evaluation", "#6C8EBF")
    _draw_arrow(left, (0.27, 0.88), (0.36, 0.88))
    _draw_arrow(left, (0.58, 0.88), (0.67, 0.88))

    _draw_box(left, 0.08, 0.56, 0.22, 0.13, "Diffusion Model\nGraph encoder\nCondition projector\nLatent denoiser", METHOD_COLORS["ours"], "#EEF4FB")
    _draw_box(left, 0.39, 0.56, 0.22, 0.13, "Structure Generator\nPrototype retrieval\nFormula editing\nCandidate decoding", METHOD_COLORS["ours"], "#EEF4FB")
    _draw_box(left, 0.70, 0.56, 0.20, 0.13, "Optimization Module\nHER activity\nStability\nSynthesis", "#4F772D", "#F3F9EF")
    _draw_arrow(left, (0.19, 0.83), (0.19, 0.69))
    _draw_arrow(left, (0.50, 0.83), (0.50, 0.69))
    _draw_arrow(left, (0.80, 0.69), (0.80, 0.83))
    _draw_arrow(left, (0.30, 0.62), (0.39, 0.62))
    _draw_arrow(left, (0.61, 0.62), (0.70, 0.62))

    _draw_box(left, 0.09, 0.28, 0.20, 0.10, "HER activity\noptimization", "#4F772D", "#F3F9EF")
    _draw_box(left, 0.40, 0.28, 0.20, 0.10, "Stability\noptimization", "#4F772D", "#F3F9EF")
    _draw_box(left, 0.71, 0.28, 0.18, 0.10, "Synthesis\nconstraint", "#4F772D", "#F3F9EF")
    _draw_arrow(left, (0.19, 0.56), (0.19, 0.38))
    _draw_arrow(left, (0.50, 0.56), (0.50, 0.38))
    _draw_arrow(left, (0.80, 0.56), (0.80, 0.38))

    right = axes[1]
    right.axis("off")
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.set_title("Baseline vs Ours", fontsize=14, fontweight="bold", pad=10)

    _draw_box(right, 0.06, 0.78, 0.34, 0.12, "Baseline\nWeak condition injection\nShared property head\nPrototype editing\nAligned post-filter", METHOD_COLORS["baseline"], METHOD_FILLS["baseline"])
    _draw_box(right, 0.58, 0.78, 0.34, 0.12, "Ours\nDual encoder\nCondition fusion\nPrior gate\nDual property trunk", METHOD_COLORS["ours"], METHOD_FILLS["ours"])

    _draw_box(right, 0.06, 0.50, 0.34, 0.14, "Condition path\nDirect projection\nWeak conditioning\nThreshold-based post selection", METHOD_COLORS["baseline"], METHOD_FILLS["baseline"])
    _draw_box(right, 0.58, 0.50, 0.34, 0.14, "Condition path\nTask-aware fusion of HER /\nstability / synthesis / 2D priors", METHOD_COLORS["ours"], METHOD_FILLS["ours"])

    _draw_box(right, 0.06, 0.22, 0.34, 0.14, "Training target\nL_pred + 0.5 * L_diff", METHOD_COLORS["baseline"], METHOD_FILLS["baseline"])
    _draw_box(right, 0.58, 0.22, 0.34, 0.14, "Training target\nL_pred + L_diff + L_cond +\nL_edit + L_feature + L_rank + L_diversity", METHOD_COLORS["ours"], METHOD_FILLS["ours"])

    _draw_arrow(right, (0.23, 0.78), (0.23, 0.64))
    _draw_arrow(right, (0.23, 0.50), (0.23, 0.36))
    _draw_arrow(right, (0.75, 0.78), (0.75, 0.64))
    _draw_arrow(right, (0.75, 0.50), (0.75, 0.36))

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_her_performance(frames: Dict[str, pd.DataFrame], output_path: Path) -> None:
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for method, frame in frames.items():
        if frame.empty:
            continue
        ranked = frame.reindex(frame["her_delta_g"].abs().sort_values().index).head(12).reset_index(drop=True)
        x = ranked.index + 1
        y = ranked["her_delta_g"].tolist()
        ax.plot(x, y, marker="o", markersize=6, linewidth=2.6, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        best_row = ranked.iloc[0]
        ax.annotate(
            best_row["formula"],
            xy=(1, best_row["her_delta_g"]),
            xytext=(1.35, best_row["her_delta_g"] + 0.08),
            fontsize=9,
            color=METHOD_COLORS[method],
            arrowprops={"arrowstyle": "->", "color": METHOD_COLORS[method]},
        )
    ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.2, label="Target ΔG_H = 0 eV")
    _style_axis(ax, "HER Performance of Generated Candidates", "Candidate Rank", "ΔG_H (eV)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_stability_curve(frames: Dict[str, pd.DataFrame], output_path: Path) -> None:
    _prepare_output(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    metric_specs = [
        ("stability_score", "Stability Score", axes[0]),
        ("synthesis_score", "Synthesis Score", axes[1]),
    ]
    for metric_key, metric_title, ax in metric_specs:
        for method, frame in frames.items():
            if frame.empty:
                continue
            ranked = frame.sort_values(metric_key, ascending=False).head(12).reset_index(drop=True)
            x = ranked.index + 1
            y = ranked[metric_key].tolist()
            ax.plot(x, y, marker="o", markersize=6, linewidth=2.4, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            ax.axhline(sum(y) / len(y), color=METHOD_COLORS[method], linestyle=":", linewidth=1.4, alpha=0.75)
        _style_axis(ax, metric_title, "Top Candidate Rank", metric_title)
        ax.set_ylim(0.0, 1.05)
    axes[1].legend(frameon=False)
    fig.suptitle("Stability and Synthesis Assessment", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_generated_structures(frame: pd.DataFrame, output_path: Path) -> None:
    _prepare_output(output_path)
    top_per_method = 4
    columns = ["baseline", "ours"]
    fig, axes = plt.subplots(top_per_method + 1, len(columns), figsize=(12.8, 10.0))

    for col, method in enumerate(columns):
        header_ax = axes[0, col]
        header_ax.axis("off")
        header_ax.text(0.5, 0.55, METHOD_LABELS[method], ha="center", va="center", fontsize=16, fontweight="bold", color=METHOD_COLORS[method])

        subset = frame[frame["method"] == method].sort_values("total_score", ascending=False).head(top_per_method)
        cards = subset.to_dict("records")
        for row in range(top_per_method):
            ax = axes[row + 1, col]
            ax.axis("off")
            if row >= len(cards):
                continue
            card = cards[row]
            action = str(card.get("edit_action", card.get("prototype_edit_action", card.get("metadata", ""))))
            text = (
                f"{card['formula']}\n"
                f"|ΔG_H| = {abs(card['her_delta_g']):.3f} eV\n"
                f"Stability = {card['stability_score']:.3f}\n"
                f"Synthesis = {card['synthesis_score']:.3f}\n"
                f"2D = {'Yes' if bool(card['is_2d']) else 'No'}\n"
                f"Edit = {card.get('edit_action', 'keep')}"
            )
            ax.text(
                0.5,
                0.5,
                text,
                ha="center",
                va="center",
                fontsize=10.5,
                bbox={
                    "boxstyle": "round,pad=0.75",
                    "facecolor": METHOD_FILLS[method],
                    "edgecolor": METHOD_COLORS[method],
                    "linewidth": 2.0,
                },
            )
    fig.suptitle("Generated Material Structure Cards", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_baseline_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    _prepare_output(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9))

    labels = comparison_df["Method"].tolist()
    her_raw = comparison_df["Avg HER ΔG (eV)"].tolist()
    stability_raw = comparison_df["Stability Score"].tolist()
    synthesis_raw = comparison_df["Synthesis Success Rate"].tolist()
    her_values = [_metric_to_float(value) for value in her_raw]
    stability_values = [_metric_to_float(value) for value in stability_raw]
    synthesis_values = [_metric_to_float(value) for value in synthesis_raw]
    colors = [METHOD_COLORS["baseline"], METHOD_COLORS["ours"]]

    metric_configs = [
        ("Avg HER ΔG (eV)", her_values, her_raw, "Lower is better"),
        ("Stability Score", stability_values, stability_raw, "Higher is better"),
        ("Synthesis Success Rate", synthesis_values, synthesis_raw, "Higher is better"),
    ]

    for ax, (title, values, raw_values, subtitle) in zip(axes, metric_configs):
        bars = ax.bar(labels, values, color=colors, width=0.58)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.22, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if title == "Synthesis Success Rate":
            ax.set_ylim(0.0, 1.05)
        for bar, raw_value in zip(bars, raw_values):
            label = raw_value if isinstance(raw_value, str) else f"{float(raw_value):.4f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, str(label), ha="center", va="bottom", fontsize=9)

    fig.suptitle("Baseline vs Ours", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
