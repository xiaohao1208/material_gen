from __future__ import annotations

"""项目评估与可视化入口。"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from models import (
    METHOD_SETTINGS,
    CompositeScorer,
    MaterialGenerationAlignedPostFilter,
    StructureGenerator,
    TaskAwareCandidateSelector,
    build_backend,
)
from train import (
    build_runtime_settings,
    prepare_output_layout,
    print_experiment_banner,
    repo_root,
    selected_methods,
    train_methods,
)
from utils.geo_utils import assemble_results_frame, relative_path, write_pseudo_cif
from utils.vis import (
    plot_baseline_comparison,
    plot_generated_structures,
    plot_her_performance,
    plot_loss_curve,
    plot_model_architecture,
    plot_stability_curve,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _format_metric(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def _with_arrow(current_value: float, baseline_value: float, lower_is_better: bool, formatter) -> str:
    """按 README 约定给 ours 行的指标加方向箭头。"""

    formatted = formatter(current_value)
    if abs(current_value - baseline_value) < 1.0e-10:
        return formatted

    improved = current_value < baseline_value if lower_is_better else current_value > baseline_value
    if lower_is_better:
        arrow = "↓" if improved else "↑"
    else:
        arrow = "↑" if improved else "↓"
    return f"{arrow}{formatted}"


def _compare_methods(method_records: Dict[str, List], top_k: int) -> pd.DataFrame:
    metric_rows: Dict[str, Dict[str, float]] = {}
    for method in ("baseline", "ours"):
        records = sorted(method_records.get(method, []), key=lambda item: float(item.total_score or 0.0), reverse=True)[:top_k]
        her_value = float(sum(abs(float(item.her_delta_g or 0.0)) for item in records) / len(records)) if records else 0.0
        stability_value = float(sum(float(item.stability_score or 0.0) for item in records) / len(records)) if records else 0.0
        synthesis_value = float(sum(float(item.synthesis_score or 0.0) >= 0.60 for item in records) / len(records)) if records else 0.0
        metric_rows[method] = {
            "her": her_value,
            "stability": stability_value,
            "synthesis": synthesis_value,
        }

    baseline_metrics = metric_rows["baseline"]
    ours_metrics = metric_rows["ours"]
    return pd.DataFrame(
        [
            {
                "Method": "baseline",
                "Avg HER ΔG (eV)": _format_metric(baseline_metrics["her"]),
                "Stability Score": _format_metric(baseline_metrics["stability"]),
                "Synthesis Success Rate": _format_percent(baseline_metrics["synthesis"]),
            },
            {
                "Method": "Ours",
                "Avg HER ΔG (eV)": _with_arrow(ours_metrics["her"], baseline_metrics["her"], True, _format_metric),
                "Stability Score": _with_arrow(ours_metrics["stability"], baseline_metrics["stability"], False, _format_metric),
                "Synthesis Success Rate": _with_arrow(ours_metrics["synthesis"], baseline_metrics["synthesis"], False, _format_percent),
            },
        ]
    )


def _apply_final_selection(method: str, scored_records: List, top_k: int) -> List:
    """在统一评分之后再执行最终候选选择。

    baseline 这里使用的是对齐 `material_generation` 主链路的后处理筛选：
    先按二维性、稳定性、可合成性和元素复杂度做门槛过滤，再按 HER 主导的综合分数排序。
    ours 则保留更强的任务化选择逻辑，把 guidance、综合评分和多样性一起考虑进去。
    """

    if method == "baseline":
        selector = MaterialGenerationAlignedPostFilter()
    else:
        selector = TaskAwareCandidateSelector()
    return selector.select(scored_records, top_k=top_k)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.to_dict("records"):
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _write_metrics(comparison_df: pd.DataFrame, metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "final_comparison.csv"
    md_path = metrics_dir / "final_comparison.md"
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(_markdown_table(comparison_df) + "\n", encoding="utf-8")
    _log(f"[test] wrote_metrics={csv_path.as_posix()}")
    _log(f"[test] wrote_metrics={md_path.as_posix()}")


def _evaluate_method(method: str, backend, num_samples: int):
    generator = StructureGenerator(backend=backend)
    scorer = CompositeScorer()

    _log(f"[test][{method}] generate_candidates start num_samples={num_samples}")
    generated = generator.generate(num_samples=num_samples, conditions=METHOD_SETTINGS[method]["proxy_targets"])
    _log(f"[test][{method}] generate_candidates done count={len(generated)}")

    _log(f"[test][{method}] score_candidates start")
    scored = scorer.score_many(generated)
    ranked = _apply_final_selection(method, scored_records=scored, top_k=max(10, num_samples))
    _log(
        f"[test][{method}] score_candidates done "
        f"scored={len(scored)} selected={len(ranked)} "
        f"top_total_score={max((float(item.total_score or 0.0) for item in ranked), default=0.0):.4f}"
    )
    return ranked


def _write_structure_files(
    records_by_method: Dict[str, List],
    output_dir: Path,
    repo_dir: Path,
    min_files_per_method: int,
) -> Dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    method_frames: Dict[str, pd.DataFrame] = {}

    for method, records in records_by_method.items():
        ranked_records = sorted(records, key=lambda item: float(item.total_score or 0.0), reverse=True)
        target_count = min(len(ranked_records), max(min_files_per_method, 10))

        # 这里不是只改结果表里的路径字符串，而是真的把候选材料写成结构文件。
        # 这样最终结果目录里才会有可查看、可追踪的 `.cif` 输出。
        for index, record in enumerate(ranked_records[:target_count], start=1):
            cif_path = output_dir / f"{method}_{index:04d}.cif"
            write_pseudo_cif(record, cif_path)
            record.cif_path = relative_path(cif_path, repo_dir)

        method_frames[method] = assemble_results_frame(ranked_records).sort_values("total_score", ascending=False).reset_index(drop=True)
        _log(f"[test][{method}] wrote_structure_files={target_count}")

    return method_frames


def _load_or_train_backends(methods: List[str], settings: Dict, output_dir: Path):
    backends: Dict[str, object] = {}
    histories: Dict[str, List[dict]] = {}
    trained_methods: Dict[str, Dict] = {}
    checkpoint_dir = Path(settings["checkpoint_dir"])

    missing_methods: List[str] = []
    for method in methods:
        checkpoint_path = checkpoint_dir / f"{method}_best.pt"
        if checkpoint_path.exists():
            # 测试阶段优先加载已有最优权重，这样评估不会无谓地重复训练。
            backend = build_backend(
                method=method,
                device=settings["device"],
                data_source=settings["data_source"],
                mp_cache_dir=settings["mp_cache_dir"],
                mp_limit=int(settings["mp_limit"]),
                download_if_missing=settings["download_if_missing"],
            )
            payload = backend.load_checkpoint(checkpoint_path, settings=settings)
            backends[method] = backend
            histories[method] = list(payload.get("history", []))
            trained_methods[method] = {
                "label": METHOD_SETTINGS[method]["label"],
                "history": histories[method],
                "best_checkpoint_path": checkpoint_path.as_posix(),
                "latest_checkpoint_path": str(checkpoint_dir / f"{method}_latest.pt"),
            }
            _log(f"[test][{method}] loaded_checkpoint={checkpoint_path.as_posix()}")
        else:
            missing_methods.append(method)

    if missing_methods:
        _log(f"[test] phase=train_backends methods={', '.join(missing_methods)}")
        trained = train_methods(missing_methods, settings, output_dir)
        backends.update(trained["backends"])
        histories.update(trained["histories"])
        trained_methods.update(trained["methods"])
        _log("[test] phase=train_backends completed")
    else:
        _log("[test] phase=load_checkpoints completed")

    # if histories:
    #     plot_loss_curve(histories, output_dir / "loss_curve.png")
    plot_model_architecture(output_dir / "model_architecture.png")
    return {"backends": backends, "histories": histories, "methods": trained_methods}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the material generation pipeline.")
    parser.add_argument("--method", choices=["baseline", "ours", "both"], default="both")
    parser.add_argument("--data-source", choices=["builtin", "materials_project"], default="materials_project")
    parser.add_argument("--mp-cache-dir", type=str, default="dataset/materials_project")
    parser.add_argument("--mp-limit", type=int, default=10000)
    parser.add_argument("--download-if-missing",  type=bool, default=False)
    parser.add_argument("--num-samples", type=int, default=32, help="Generated candidates per method")
    parser.add_argument("--top-k", type=int, default=10, help="Top rows used for summary display")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--log-interval", type=int, default=20, help="Step logging interval")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if needed")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume a single method from a specific checkpoint")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=str, default="results", help="Relative output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    total_start = time.perf_counter()
    args = parse_args(argv)
    output_dir = repo_root() / args.output_dir
    layout = prepare_output_layout(output_dir, reset_generated_structures=True)
    methods = selected_methods(args.method)
    settings = build_runtime_settings(args, output_dir=layout["output_dir"])
    settings["generation"]["num_samples"] = int(args.num_samples)
    settings["generation"]["top_k"] = int(args.top_k)

    print_experiment_banner(methods, settings)
    trained = _load_or_train_backends(methods, settings, layout["output_dir"])

    records_by_method: Dict[str, List] = {}
    for method in methods:
        # 每种方法都经过“生成 -> 统一评分 -> 方法专属最终筛选”三步。
        # 这样 baseline 和 ours 共享同一评分器，但保留不同的后处理语义。
        records_by_method[method] = _evaluate_method(
            method=method,
            backend=trained["backends"][method],
            num_samples=int(settings["generation"]["num_samples"]),
        )

    method_frames = _write_structure_files(
        records_by_method=records_by_method,
        output_dir=layout["generated_structures"],
        repo_dir=repo_root(),
        min_files_per_method=int(settings["generation"]["top_k"]),
    )

    combined_frame = pd.concat([method_frames[method] for method in methods], ignore_index=True)
    combined_frame = combined_frame.sort_values("total_score", ascending=False).reset_index(drop=True)
    comparison_df = _compare_methods(records_by_method, top_k=int(settings["generation"]["top_k"]))

    _log("[test] phase=write_figures")
    plot_her_performance(method_frames, layout["output_dir"] / "her_performance.png")
    plot_stability_curve(method_frames, layout["output_dir"] / "stability_curve.png")
    plot_generated_structures(combined_frame, layout["output_dir"] / "generated_structures.png")
    plot_baseline_comparison(comparison_df, layout["output_dir"] / "baseline_comparison.png")
    plot_model_architecture(layout["output_dir"] / "model_architecture.png")
    # _log(f"[test] wrote_figure={(layout['output_dir'] / 'loss_curve.png').as_posix()}")
    _log(f"[test] wrote_figure={(layout['output_dir'] / 'her_performance.png').as_posix()}")
    _log(f"[test] wrote_figure={(layout['output_dir'] / 'stability_curve.png').as_posix()}")
    _log(f"[test] wrote_figure={(layout['output_dir'] / 'generated_structures.png').as_posix()}")
    _log(f"[test] wrote_figure={(layout['output_dir'] / 'baseline_comparison.png').as_posix()}")
    # _log(f"[test] wrote_figure={(layout['output_dir'] / 'model_architecture.png').as_posix()}")

    _log("[test] phase=write_metrics")
    _write_metrics(comparison_df, layout["metrics"])

    print("Comparison Table")
    print(_markdown_table(comparison_df))
    print("\nTop Candidates")
    print(
        combined_frame.head(int(settings["generation"]["top_k"]))[
            ["method", "formula", "her_delta_g", "stability_score", "synthesis_score", "total_score", "cif_path"]
        ].to_string(index=False)
    )
    _log(f"[test] generated_structure_dir={layout['generated_structures'].as_posix()}")
    _log(f"[test] total_elapsed={time.perf_counter() - total_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
