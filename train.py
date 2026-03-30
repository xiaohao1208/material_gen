from __future__ import annotations

"""项目训练入口。"""

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import torch

from models import build_backend
from utils.geo_utils import ensure_directory, reset_directory
from utils.vis import plot_loss_curve, plot_model_architecture


DEFAULT_SETTINGS: Dict[str, Dict] = {
    "train": {"epochs": 40, "batch_size": 16},
    "generation": {"num_samples": 32, "top_k": 10},
}


def _log(message: str) -> None:
    print(message, flush=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def selected_methods(method: str) -> List[str]:
    return ["baseline", "ours"] if method == "both" else [method]


def prepare_output_layout(output_dir: Path, reset_generated_structures: bool = False) -> Dict[str, Path]:
    # 训练和测试都共享同一套输出目录结构。
    # 这里把运行日志、指标、权重和结构文件目录一次性准备好，后续模块只管往里写。
    ensure_directory(output_dir)
    # runtime_logs = reset_directory(output_dir / "runtime_logs")
    metrics_dir = reset_directory(output_dir / "metrics")
    checkpoints_dir = ensure_directory(output_dir / "checkpoints")
    generated_structures_dir = output_dir / "generated_structures"
    if reset_generated_structures:
        generated_structures_dir = reset_directory(generated_structures_dir)
    else:
        generated_structures_dir = ensure_directory(generated_structures_dir)
    return {
        "output_dir": output_dir,
        # "runtime_logs": runtime_logs,
        "metrics": metrics_dir,
        "checkpoints": checkpoints_dir,
        "generated_structures": generated_structures_dir,
    }


def build_runtime_settings(args: argparse.Namespace, output_dir: Path | None = None) -> Dict:
    settings = deepcopy(DEFAULT_SETTINGS)
    if getattr(args, "epochs", None) is not None:
        settings["train"]["epochs"] = int(args.epochs)
    if getattr(args, "batch_size", None) is not None:
        settings["train"]["batch_size"] = int(args.batch_size)
    settings["device"] = args.device
    settings["output_dir"] = str(output_dir or args.output_dir)
    # settings["runtime_logs_dir"] = str((output_dir or repo_root() / args.output_dir) / "runtime_logs")
    settings["metrics_dir"] = str((output_dir or repo_root() / args.output_dir) / "metrics")
    settings["checkpoint_dir"] = str((output_dir or repo_root() / args.output_dir) / "checkpoints")
    settings["generated_structures_dir"] = str((output_dir or repo_root() / args.output_dir) / "generated_structures")
    settings["data_source"] = args.data_source
    settings["mp_cache_dir"] = args.mp_cache_dir
    settings["mp_limit"] = int(args.mp_limit)
    settings["download_if_missing"] = bool(args.download_if_missing)
    settings["log_interval"] = int(args.log_interval)
    settings["resume"] = bool(getattr(args, "resume", False))
    settings["resume_from"] = getattr(args, "resume_from", None)
    return settings


def print_experiment_banner(methods: List[str], settings: Dict) -> None:
    _log("=" * 92)
    _log("HER Material Generation Training")
    _log(f"Methods          : {', '.join(methods)}")
    _log(f"Data source      : {settings['data_source']}")
    _log(f"Device           : {settings['device']}")
    _log(f"MP cache dir     : {settings['mp_cache_dir']}")
    _log(f"MP raw limit     : {settings['mp_limit']}")
    _log(f"Epochs           : {settings['train']['epochs']}")
    _log(f"Batch size       : {settings['train']['batch_size']}")
    _log(f"Samples / top-k  : {settings['generation']['num_samples']} / {settings['generation']['top_k']}")
    _log(f"Log interval     : {settings['log_interval']}")
    _log(f"Resume           : {settings['resume']}")
    if settings.get("resume_from"):
        _log(f"Resume from      : {settings['resume_from']}")
    _log(f"Checkpoint dir   : {settings['checkpoint_dir']}")
    _log(f"Output dir       : {settings['output_dir']}")
    _log("=" * 92)


def train_methods(methods: List[str], settings: Dict, output_dir: Path) -> Dict:
    histories: Dict[str, List[dict]] = {}
    summaries: Dict[str, Dict] = {}
    backends: Dict[str, object] = {}

    for method in methods:
        # baseline 和 ours 共用同一套 backend 构建入口。
        # 真正的结构差异和损失差异都封装在 backend 内部，而不是靠训练脚本写分支。
        backend = build_backend(
            method=method,
            device=settings["device"],
            data_source=settings["data_source"],
            mp_cache_dir=settings["mp_cache_dir"],
            mp_limit=int(settings["mp_limit"]),
            download_if_missing=settings["download_if_missing"],
        )
        summary = backend.train(settings=settings)
        histories[method] = summary["history"]
        summaries[method] = summary
        backends[method] = backend

    if histories:
        plot_loss_curve(histories, output_dir / "loss_curve.png")
    plot_model_architecture(output_dir / "model_architecture.png")
    return {"methods": summaries, "backends": backends, "histories": histories}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the material graph diffusion model.")
    parser.add_argument("--method", choices=["baseline", "ours", "both"], default="both")
    parser.add_argument("--data-source", choices=["builtin", "materials_project"], default="materials_project")
    parser.add_argument("--mp-cache-dir", type=str, default="dataset/materials_project")
    parser.add_argument("--mp-limit", type=int, default=10000, help="Maximum raw Materials Project records to cache")
    parser.add_argument("--download-if-missing", type=bool, default=False, help="Download MP data if local cache is missing")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--log-interval", type=int, default=20, help="Step logging interval")
    parser.add_argument("--resume", action="store_true", help="Resume each method from its latest checkpoint if present")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume a single method from a specific checkpoint path")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=str, default="results", help="Relative output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = repo_root() / args.output_dir
    layout = prepare_output_layout(output_dir)
    methods = selected_methods(args.method)
    settings = build_runtime_settings(args, output_dir=layout["output_dir"])

    print_experiment_banner(methods, settings)
    trained = train_methods(methods, settings, layout["output_dir"])

    _log("[train] completed")
    for method in methods:
        summary = trained["methods"][method]
        _log(
            f"[train] {summary['label']} train_loss={summary['train_loss']:.4f} "
            f"val_loss={summary['val_loss']:.4f} params={summary['num_parameters']} "
            f"train_samples={summary['train_samples']} val_samples={summary['val_samples']} "
            f"steps={summary['train_steps_per_epoch']}/{summary['val_steps_per_epoch']} "
            f"checkpoint_best={summary['best_checkpoint_path']} "
            f"checkpoint_latest={summary['latest_checkpoint_path']}"
        )
    _log(f"[train] wrote_figure={(layout['output_dir'] / 'loss_curve.png').as_posix()}")
    _log(f"[train] wrote_figure={(layout['output_dir'] / 'model_architecture.png').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
