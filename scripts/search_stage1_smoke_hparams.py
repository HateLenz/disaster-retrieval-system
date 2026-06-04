from __future__ import annotations

import argparse
import csv
import itertools
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_PROFILE = REPO_ROOT / "configs" / "experiment" / "smoke_stage1_visual_v2.yaml"
DEFAULT_SUBSET_PROFILE = REPO_ROOT / "configs" / "experiment" / "smoke_stage1_visual.yaml"
DEFAULT_FULL_BASE_PROFILE = REPO_ROOT / "configs" / "experiment" / "full_stage1_visual.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "hparam_search" / "stage1_smoke"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "hparam_search" / "stage1_smoke"
DEFAULT_INDEX_ROOT = REPO_ROOT / "indexes" / "hparam_search" / "stage1_smoke"

SCRIPT_PATHS = {
    "subset": REPO_ROOT / "scripts" / "build_smoke_subset.py",
    "train": REPO_ROOT / "scripts" / "train_clip_visual_baseline.py",
    "extract": REPO_ROOT / "scripts" / "extract_pre_clip_features.py",
    "eval": REPO_ROOT / "scripts" / "eval_clip_retrieval.py",
    "tile_eval": REPO_ROOT / "scripts" / "eval_clip_tile_retrieval.py",
    "cases": REPO_ROOT / "scripts" / "export_retrieval_cases.py",
    "sanity": REPO_ROOT / "scripts" / "report_retrieval_sanity.py",
    "limitations": REPO_ROOT / "scripts" / "summarize_visual_limitations.py",
}

BEST_FINAL_STEPS = ["extract", "eval", "tile_eval", "cases", "sanity", "limitations"]
CORE_HPARAM_KEYS = [
    "lr",
    "batch_size",
    "temperature",
    "weight_decay",
    "epochs",
    "loss_weighting",
    "disable_balanced_sampling",
    "embedding_dim",
    "unfreeze_backbone",
    "seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coarse-to-fine search for stage-1 CLIP visual retrieval hyperparameters on the smoke dataset, "
            "then export final retrieval results for the best trial."
        ),
    )
    parser.add_argument("--base-profile", type=Path, default=DEFAULT_BASE_PROFILE)
    parser.add_argument("--subset-profile", type=Path, default=DEFAULT_SUBSET_PROFILE)
    parser.add_argument("--full-base-profile", type=Path, default=DEFAULT_FULL_BASE_PROFILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--python", type=str, default=sys.executable)

    parser.add_argument(
        "--search-mode",
        choices=["single", "coarse_fine"],
        default="coarse_fine",
        help="single runs one flat grid; coarse_fine runs the coarse grid, then refines around top coarse trials.",
    )
    parser.add_argument("--lr-grid", type=float, nargs="+", default=[2e-4, 1e-4, 5e-5])
    parser.add_argument("--batch-size-grid", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--temperature-grid", type=float, nargs="+", default=[0.05, 0.07, 0.1])
    parser.add_argument("--weight-decay-grid", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--epochs-grid", type=int, nargs="+", default=[30])
    parser.add_argument("--embedding-dim-grid", type=int, nargs="+", default=[256])
    parser.add_argument("--seed-grid", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--loss-weighting-grid",
        nargs="+",
        choices=["none", "inverse_freq"],
        default=["inverse_freq"],
    )
    parser.add_argument(
        "--balanced-sampling-grid",
        nargs="+",
        choices=["enabled", "disabled"],
        default=["enabled"],
        help="Whether to use disaster-type balanced sampling in the training loader.",
    )
    parser.add_argument(
        "--backbone-grid",
        nargs="+",
        choices=["frozen", "unfrozen"],
        default=["frozen"],
        help="Whether to keep CLIP backbone frozen or fine-tune it. Frozen is recommended for smoke search.",
    )

    parser.add_argument(
        "--fine-top-n",
        type=int,
        default=3,
        help="Number of top coarse trials used as anchors for fine search.",
    )
    parser.add_argument(
        "--fine-lr-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Fine-stage learning-rate multipliers applied to each selected coarse lr.",
    )
    parser.add_argument(
        "--fine-temperature-deltas",
        type=float,
        nargs="+",
        default=[-0.02, 0.0, 0.02],
        help="Fine-stage temperature offsets applied to each selected coarse temperature.",
    )
    parser.add_argument("--fine-temperature-min", type=float, default=0.01)
    parser.add_argument("--fine-temperature-max", type=float, default=0.20)
    parser.add_argument(
        "--fine-weight-decay-grid",
        type=float,
        nargs="+",
        default=[0.0, 1e-5, 1e-4, 1e-3],
        help="Fine-stage weight decay values. Coarse stage intentionally keeps weight_decay narrow.",
    )
    parser.add_argument(
        "--fine-batch-policy",
        choices=["same", "neighbors"],
        default="neighbors",
        help="same keeps the coarse winner's batch size; neighbors also tests adjacent values from --batch-size-grid.",
    )
    parser.add_argument(
        "--fine-epochs-grid",
        type=int,
        nargs="+",
        default=None,
        help="Fine-stage epochs. Defaults to --epochs-grid, so the current seed/epoch policy is preserved.",
    )
    parser.add_argument("--max-fine-trials", type=int, default=None, help="Optional cap for fine-stage debugging.")

    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap for single/coarse-stage debugging.")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="best_checkpoint_metric",
        help="Metric path read from train_summary.json, e.g. best_checkpoint_metric or best_eval_macro.recall@1.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override training/eval device from the base profile.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override train/extract/eval worker count.")
    parser.add_argument("--top-k", type=int, default=None, help="Override patch-level top_k from the base profile.")
    parser.add_argument("--rebuild-subset", action="store_true", help="Recreate the smoke subset before searching.")
    parser.add_argument("--force", action="store_true", help="Re-run trials even when their summaries already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print child commands without executing them.")
    parser.add_argument("--skip-final-eval", action="store_true", help="Only run search; do not export best final results.")
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def repo_path(path: Path | str) -> str:
    resolved = resolve_repo_path(Path(path))
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def metric_from_path(payload: dict[str, Any], metric_path: str) -> float:
    current: Any = payload
    for part in metric_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        raise KeyError(f"Metric path '{metric_path}' not found at '{part}'")
    return float(current)


def command_from_mapping(script_path: Path, params: dict[str, Any], python_executable: str) -> list[str]:
    command = [python_executable, str(script_path)]
    for key, value in params.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                command.extend([flag, str(item)])
            continue
        command.extend([flag, str(value)])
    return command


def write_command_log(log_path: Path, command: list[str], completed: subprocess.CompletedProcess[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as file:
        file.write(f"$ {' '.join(shlex.quote(part) for part in command)}\n\n")
        file.write(completed.stdout)
        if completed.stderr:
            file.write("\n[stderr]\n")
            file.write(completed.stderr)


def run_command(command: list[str], log_path: Path, dry_run: bool) -> None:
    print(f"$ {' '.join(shlex.quote(part) for part in command)}")
    if dry_run:
        return
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    write_command_log(log_path=log_path, command=command, completed=completed)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}. See log: {log_path}")


def subset_command_from_profile(profile: dict[str, Any], python_executable: str) -> list[str] | None:
    subset_cfg = dict(profile.get("subset", {}))
    if not subset_cfg or not subset_cfg.get("enabled", False):
        return None

    split_limits = subset_cfg.pop("split_limits", {})
    group_by = subset_cfg.pop("group_by", [])
    subset_cfg.pop("enabled", None)
    command = [python_executable, str(SCRIPT_PATHS["subset"])]
    for key, value in subset_cfg.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif value is not None:
            command.extend([flag, str(value)])
    if group_by:
        command.extend(["--group-by", *[str(item) for item in group_by]])
    for split_name, count in split_limits.items():
        command.extend(["--split-limit", f"{split_name}={count}"])
    return command


def ensure_smoke_subset(args: argparse.Namespace, base_profile: dict[str, Any]) -> None:
    csv_path = resolve_repo_path(Path(base_profile["train"]["csv"]))
    if csv_path.exists() and not args.rebuild_subset:
        print(f"Smoke CSV exists: {repo_path(csv_path)}")
        return

    subset_profile = load_yaml(args.subset_profile)
    command = subset_command_from_profile(subset_profile, python_executable=args.python)
    if command is None:
        raise RuntimeError(f"Cannot build smoke subset from profile: {args.subset_profile}")
    run_command(
        command=command,
        log_path=resolve_repo_path(args.output_root) / "subset.log",
        dry_run=args.dry_run,
    )


def clean_float(value: float) -> float:
    return float(f"{value:.8g}")


def param_key(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    normalized: dict[str, Any] = {}
    for key in CORE_HPARAM_KEYS:
        value = params.get(key)
        if isinstance(value, float):
            normalized[key] = clean_float(value)
        else:
            normalized[key] = value
    return tuple((key, normalized.get(key)) for key in CORE_HPARAM_KEYS)


def dedupe_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for params in trials:
        key = param_key(params)
        if key in seen:
            continue
        seen.add(key)
        unique.append(params)
    return unique


def build_grid_from_axes(
    lr_grid: list[float],
    batch_size_grid: list[int],
    temperature_grid: list[float],
    weight_decay_grid: list[float],
    epochs_grid: list[int],
    embedding_dim_grid: list[int],
    seed_grid: list[int],
    loss_weighting_grid: list[str],
    balanced_sampling_grid: list[str],
    backbone_grid: list[str],
) -> list[dict[str, Any]]:
    raw_grid = itertools.product(
        lr_grid,
        batch_size_grid,
        temperature_grid,
        weight_decay_grid,
        epochs_grid,
        embedding_dim_grid,
        seed_grid,
        loss_weighting_grid,
        balanced_sampling_grid,
        backbone_grid,
    )
    trials: list[dict[str, Any]] = []
    for (
        lr,
        batch_size,
        temperature,
        weight_decay,
        epochs,
        embedding_dim,
        seed,
        loss_weighting,
        balanced_sampling,
        backbone,
    ) in raw_grid:
        trials.append(
            {
                "lr": clean_float(lr),
                "batch_size": batch_size,
                "temperature": clean_float(temperature),
                "weight_decay": clean_float(weight_decay),
                "epochs": epochs,
                "embedding_dim": embedding_dim,
                "seed": seed,
                "loss_weighting": loss_weighting,
                "disable_balanced_sampling": balanced_sampling == "disabled",
                "unfreeze_backbone": backbone == "unfrozen",
            }
        )
    return dedupe_trials(trials)


def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    trials = build_grid_from_axes(
        lr_grid=args.lr_grid,
        batch_size_grid=args.batch_size_grid,
        temperature_grid=args.temperature_grid,
        weight_decay_grid=args.weight_decay_grid,
        epochs_grid=args.epochs_grid,
        embedding_dim_grid=args.embedding_dim_grid,
        seed_grid=args.seed_grid,
        loss_weighting_grid=args.loss_weighting_grid,
        balanced_sampling_grid=args.balanced_sampling_grid,
        backbone_grid=args.backbone_grid,
    )
    if args.max_trials is not None:
        return trials[: max(args.max_trials, 0)]
    return trials


def fine_batch_sizes(best_batch_size: int, args: argparse.Namespace) -> list[int]:
    if args.fine_batch_policy == "same":
        return [best_batch_size]

    ordered = sorted({int(value) for value in args.batch_size_grid})
    if best_batch_size not in ordered:
        return [best_batch_size]
    index = ordered.index(best_batch_size)
    candidates = ordered[max(0, index - 1) : min(len(ordered), index + 2)]
    return candidates or [best_batch_size]


def fine_temperatures(best_temperature: float, args: argparse.Namespace) -> list[float]:
    values = []
    for delta in args.fine_temperature_deltas:
        value = best_temperature + delta
        if args.fine_temperature_min <= value <= args.fine_temperature_max:
            values.append(clean_float(value))
    return sorted(set(values))


def fine_lrs(best_lr: float, args: argparse.Namespace) -> list[float]:
    return sorted({clean_float(best_lr * multiplier) for multiplier in args.fine_lr_multipliers})


def build_fine_grid(
    coarse_results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not coarse_results:
        return []

    ranked = sorted(coarse_results, key=lambda item: item["metric"], reverse=True)
    anchors = ranked[: max(args.fine_top_n, 0)]
    if not anchors:
        return []

    existing_keys = {param_key(result["params"]) for result in coarse_results}
    fine_epochs_grid = args.fine_epochs_grid or args.epochs_grid
    trials: list[dict[str, Any]] = []
    for anchor in anchors:
        params = anchor["params"]
        anchor_trials = build_grid_from_axes(
            lr_grid=fine_lrs(float(params["lr"]), args),
            batch_size_grid=fine_batch_sizes(int(params["batch_size"]), args),
            temperature_grid=fine_temperatures(float(params["temperature"]), args),
            weight_decay_grid=args.fine_weight_decay_grid,
            epochs_grid=fine_epochs_grid,
            embedding_dim_grid=[int(params["embedding_dim"])],
            seed_grid=[int(params["seed"])],
            loss_weighting_grid=[str(params["loss_weighting"])],
            balanced_sampling_grid=["disabled" if params["disable_balanced_sampling"] else "enabled"],
            backbone_grid=["unfrozen" if params["unfreeze_backbone"] else "frozen"],
        )
        trials.extend(anchor_trials)

    unique_trials = []
    seen = set(existing_keys)
    for params in dedupe_trials(trials):
        key = param_key(params)
        if key in seen:
            continue
        seen.add(key)
        unique_trials.append(params)

    if args.max_fine_trials is not None:
        return unique_trials[: max(args.max_fine_trials, 0)]
    return unique_trials


def trial_id(stage: str, index: int, params: dict[str, Any]) -> str:
    sampler = "unbalanced" if params["disable_balanced_sampling"] else "balanced"
    backbone = "unfrozen" if params["unfreeze_backbone"] else "frozen"
    return (
        f"{stage}_{index:04d}"
        f"_lr{params['lr']:.0e}"
        f"_bs{params['batch_size']}"
        f"_t{params['temperature']:.2f}"
        f"_wd{params['weight_decay']:.0e}"
        f"_ep{params['epochs']}"
        f"_{params['loss_weighting']}"
        f"_{sampler}"
        f"_{backbone}"
        f"_s{params['seed']}"
    ).replace("+", "")


def build_train_params(
    base_profile: dict[str, Any],
    params: dict[str, Any],
    trial_name: str,
    stage: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_cfg = dict(base_profile["train"])
    train_cfg.update(params)
    train_cfg["checkpoint"] = repo_path(resolve_repo_path(args.checkpoint_root) / f"{trial_name}.pt")
    train_cfg["train_log_jsonl"] = repo_path(resolve_repo_path(args.output_root) / "trials" / stage / trial_name / "train_history.jsonl")
    train_cfg["summary_json"] = repo_path(resolve_repo_path(args.output_root) / "trials" / stage / trial_name / "train_summary.json")
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.top_k is not None:
        train_cfg["top_k"] = args.top_k
    return train_cfg


def run_trial(
    stage: str,
    index: int,
    params: dict[str, Any],
    base_profile: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = trial_id(stage=stage, index=index, params=params)
    output_root = resolve_repo_path(args.output_root)
    summary_path = output_root / "trials" / stage / name / "train_summary.json"
    log_path = output_root / "trials" / stage / name / "train.log"

    train_params = build_train_params(
        base_profile=base_profile,
        params=params,
        trial_name=name,
        stage=stage,
        args=args,
    )
    command = command_from_mapping(
        script_path=SCRIPT_PATHS["train"],
        params=train_params,
        python_executable=args.python,
    )

    if summary_path.exists() and not args.force:
        print(f"[{name}] existing summary found, skipping training")
    else:
        print(f"[{name}] training")
        run_command(command=command, log_path=log_path, dry_run=args.dry_run)

    if args.dry_run:
        metric = 0.0
        summary: dict[str, Any] = {}
    else:
        summary = read_json(summary_path)
        metric = metric_from_path(summary, args.selection_metric)

    return {
        "stage": stage,
        "trial_name": name,
        "metric": metric,
        "params": params,
        "checkpoint": train_params["checkpoint"],
        "summary_json": repo_path(summary_path),
        "train_log_jsonl": train_params["train_log_jsonl"],
        "command": command,
        "summary": summary,
    }


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary", {})
    best_eval = summary.get("best_eval", {}) if isinstance(summary, dict) else {}
    best_eval_macro = summary.get("best_eval_macro", {}) if isinstance(summary, dict) else {}

    row: dict[str, Any] = {
        "stage": result.get("stage", "single"),
        "trial_name": result["trial_name"],
        "metric": result["metric"],
        "checkpoint": result["checkpoint"],
        "summary_json": result["summary_json"],
    }
    row.update(result["params"])
    for key, value in best_eval.items():
        row[f"best_eval_{key}"] = value
    for key, value in best_eval_macro.items():
        row[f"best_eval_macro_{key}"] = value
    row["best_epoch"] = summary.get("best_epoch") if isinstance(summary, dict) else None
    return row


def write_search_results(output_root: Path, results: list[dict[str, Any]]) -> None:
    rows = [flatten_result(result) for result in results]
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "search_results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_step_common(
    step_cfg: dict[str, Any],
    checkpoint: str,
    features: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    updated = dict(step_cfg)
    updated["checkpoint"] = checkpoint
    if features is not None and "features" in updated:
        updated["features"] = features
    if args.device is not None and "device" in updated:
        updated["device"] = args.device
    if args.num_workers is not None and "num_workers" in updated:
        updated["num_workers"] = args.num_workers
    return updated


def build_best_smoke_profile(base_profile: dict[str, Any], best: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_root = resolve_repo_path(args.output_root)
    index_root = resolve_repo_path(args.index_root)
    best_output = output_root / "best"
    best_index = index_root / "best"
    checkpoint = best.get("best_checkpoint_copy", best["checkpoint"])
    features = repo_path(best_index / "pre_clip_features.npy")

    profile = dict(base_profile)
    profile["name"] = "smoke_stage1_visual_hparam_best"
    profile["description"] = "Best stage-1 smoke profile selected by search_stage1_smoke_hparams.py."
    profile["log_dir"] = repo_path(best_output / "logs")
    profile["subset"] = {"enabled": False}

    train_cfg = dict(base_profile["train"])
    train_cfg.update(best["params"])
    train_cfg["checkpoint"] = checkpoint
    train_cfg["train_log_jsonl"] = best["train_log_jsonl"]
    train_cfg["summary_json"] = best["summary_json"]
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.top_k is not None:
        train_cfg["top_k"] = args.top_k
    profile["train"] = train_cfg

    extract_cfg = update_step_common(dict(base_profile["extract"]), checkpoint, None, args)
    extract_cfg["output"] = features
    profile["extract"] = extract_cfg

    eval_cfg = update_step_common(dict(base_profile["eval"]), checkpoint, features, args)
    eval_cfg["index"] = repo_path(best_index / "pre_clip.faiss")
    eval_cfg["results_csv"] = repo_path(best_output / "clip_retrieval_results.csv")
    eval_cfg["metrics_json"] = repo_path(best_output / "clip_retrieval_metrics.json")
    profile["eval"] = eval_cfg

    if "tile_eval" in base_profile:
        tile_cfg = update_step_common(dict(base_profile["tile_eval"]), checkpoint, features, args)
        tile_cfg["index"] = repo_path(best_index / "pre_clip_tile.faiss")
        tile_cfg["results_csv"] = repo_path(best_output / "clip_tile_retrieval_results.csv")
        tile_cfg["metrics_json"] = repo_path(best_output / "clip_tile_retrieval_metrics.json")
        profile["tile_eval"] = tile_cfg

    if "cases" in base_profile:
        cases_cfg = dict(base_profile["cases"])
        cases_cfg["results_csv"] = eval_cfg["results_csv"]
        cases_cfg["output_dir"] = repo_path(best_output / "retrieval_cases")
        profile["cases"] = cases_cfg

    if "sanity" in base_profile:
        sanity_cfg = dict(base_profile["sanity"])
        sanity_cfg["features"] = features
        sanity_cfg["feature_metadata_csv"] = repo_path(best_index / "pre_clip_features.csv")
        sanity_cfg["retrieval_results_csv"] = eval_cfg["results_csv"]
        sanity_cfg["report_json"] = repo_path(best_output / "sanity_report.json")
        profile["sanity"] = sanity_cfg

    if "limitations" in base_profile:
        limitations_cfg = dict(base_profile["limitations"])
        limitations_cfg["metrics_json"] = eval_cfg["metrics_json"]
        limitations_cfg["results_csv"] = eval_cfg["results_csv"]
        limitations_cfg["summary_json"] = repo_path(best_output / "visual_limitations_summary.json")
        limitations_cfg["summary_md"] = repo_path(best_output / "visual_limitations_summary.md")
        limitations_cfg["failed_cases_csv"] = repo_path(best_output / "visual_limitations_failed_cases.csv")
        profile["limitations"] = limitations_cfg

    return profile


def build_full_transfer_profile(full_base_profile: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    profile = dict(full_base_profile)
    profile["name"] = "full_stage1_visual_from_smoke_best"
    profile["description"] = "Full-data stage-1 profile using hyperparameters selected on smoke."
    profile["log_dir"] = "outputs/logs/full_stage1_visual_from_smoke_best"
    profile["subset"] = {"enabled": False}

    train_cfg = dict(full_base_profile["train"])
    for key in CORE_HPARAM_KEYS:
        if key in best["params"]:
            train_cfg[key] = best["params"][key]
    train_cfg["checkpoint"] = "checkpoints/clip_visual_baseline_full_stage1_from_smoke_best.pt"
    train_cfg["train_log_jsonl"] = "outputs/logs/full_stage1_visual_from_smoke_best/train_history.jsonl"
    train_cfg["summary_json"] = "outputs/logs/full_stage1_visual_from_smoke_best/train_summary.json"
    profile["train"] = train_cfg

    if "extract" in profile:
        profile["extract"] = dict(profile["extract"])
        profile["extract"]["checkpoint"] = train_cfg["checkpoint"]
        profile["extract"]["output"] = "indexes/pre_clip_features_full_stage1_from_smoke_best.npy"
    if "eval" in profile:
        profile["eval"] = dict(profile["eval"])
        profile["eval"]["checkpoint"] = train_cfg["checkpoint"]
        profile["eval"]["features"] = "indexes/pre_clip_features_full_stage1_from_smoke_best.npy"
        profile["eval"]["index"] = "indexes/pre_clip_full_stage1_from_smoke_best.faiss"
        profile["eval"]["results_csv"] = "outputs/predictions/clip_retrieval_results_full_stage1_from_smoke_best.csv"
        profile["eval"]["metrics_json"] = "outputs/predictions/clip_retrieval_metrics_full_stage1_from_smoke_best.json"
    if "tile_eval" in profile:
        profile["tile_eval"] = dict(profile["tile_eval"])
        profile["tile_eval"]["checkpoint"] = train_cfg["checkpoint"]
        profile["tile_eval"]["features"] = "indexes/pre_clip_features_full_stage1_from_smoke_best.npy"
        profile["tile_eval"]["index"] = "indexes/pre_clip_full_stage1_from_smoke_best_tile.faiss"
        profile["tile_eval"]["results_csv"] = "outputs/predictions/clip_tile_retrieval_results_full_stage1_from_smoke_best.csv"
        profile["tile_eval"]["metrics_json"] = "outputs/predictions/clip_tile_retrieval_metrics_full_stage1_from_smoke_best.json"
    if "cases" in profile:
        profile["cases"] = dict(profile["cases"])
        profile["cases"]["results_csv"] = "outputs/predictions/clip_retrieval_results_full_stage1_from_smoke_best.csv"
        profile["cases"]["output_dir"] = "outputs/figures/retrieval_cases_full_stage1_from_smoke_best"
    if "sanity" in profile:
        profile["sanity"] = dict(profile["sanity"])
        profile["sanity"]["features"] = "indexes/pre_clip_features_full_stage1_from_smoke_best.npy"
        profile["sanity"]["feature_metadata_csv"] = "indexes/pre_clip_features_full_stage1_from_smoke_best.csv"
        profile["sanity"]["retrieval_results_csv"] = "outputs/predictions/clip_retrieval_results_full_stage1_from_smoke_best.csv"
        profile["sanity"]["report_json"] = "outputs/logs/full_stage1_visual_from_smoke_best/sanity_report.json"
    if "limitations" in profile:
        profile["limitations"] = dict(profile["limitations"])
        profile["limitations"]["metrics_json"] = "outputs/predictions/clip_retrieval_metrics_full_stage1_from_smoke_best.json"
        profile["limitations"]["results_csv"] = "outputs/predictions/clip_retrieval_results_full_stage1_from_smoke_best.csv"
        profile["limitations"]["summary_json"] = "outputs/logs/full_stage1_visual_from_smoke_best/visual_limitations_summary.json"
        profile["limitations"]["summary_md"] = "outputs/logs/full_stage1_visual_from_smoke_best/visual_limitations_summary.md"
        profile["limitations"]["failed_cases_csv"] = "outputs/logs/full_stage1_visual_from_smoke_best/visual_limitations_failed_cases.csv"
    return profile


def run_best_final_steps(profile: dict[str, Any], args: argparse.Namespace) -> None:
    output_root = resolve_repo_path(args.output_root)
    logs_dir = output_root / "best" / "logs"
    for step in BEST_FINAL_STEPS:
        if step not in profile:
            continue
        command = command_from_mapping(
            script_path=SCRIPT_PATHS[step],
            params=dict(profile[step]),
            python_executable=args.python,
        )
        print(f"[best:{step}]")
        run_command(command=command, log_path=logs_dir / f"{step}.log", dry_run=args.dry_run)


def copy_best_checkpoint(best: dict[str, Any], args: argparse.Namespace) -> str | None:
    if args.dry_run:
        return None
    source = resolve_repo_path(Path(best["checkpoint"]))
    target = resolve_repo_path(args.checkpoint_root) / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return repo_path(target)


def main() -> None:
    args = parse_args()
    output_root = resolve_repo_path(args.output_root)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        resolve_repo_path(args.checkpoint_root).mkdir(parents=True, exist_ok=True)
        resolve_repo_path(args.index_root).mkdir(parents=True, exist_ok=True)

    base_profile = load_yaml(args.base_profile)
    ensure_smoke_subset(args=args, base_profile=base_profile)

    coarse_trials = build_grid(args)
    if not coarse_trials:
        raise RuntimeError("The hyperparameter grid is empty.")
    coarse_stage_name = "single" if args.search_mode == "single" else "coarse"
    print(
        json.dumps(
            {
                "search_mode": args.search_mode,
                "coarse_trial_count": len(coarse_trials),
                "selection_metric": args.selection_metric,
                "seed_grid": args.seed_grid,
            },
            ensure_ascii=False,
        )
    )

    results: list[dict[str, Any]] = []
    coarse_results: list[dict[str, Any]] = []
    for index, params in enumerate(coarse_trials, start=1):
        result = run_trial(
            stage=coarse_stage_name,
            index=index,
            params=params,
            base_profile=base_profile,
            args=args,
        )
        coarse_results.append(result)
        results.append(result)
        if not args.dry_run:
            write_search_results(output_root=output_root, results=results)

    fine_trials: list[dict[str, Any]] = []
    fine_results: list[dict[str, Any]] = []
    if args.search_mode == "coarse_fine":
        fine_trials = build_fine_grid(coarse_results=coarse_results, args=args)
        print(
            json.dumps(
                {
                    "fine_top_n": args.fine_top_n,
                    "fine_trial_count": len(fine_trials),
                    "fine_batch_policy": args.fine_batch_policy,
                    "fine_epochs_grid": args.fine_epochs_grid or args.epochs_grid,
                },
                ensure_ascii=False,
            )
        )
        for index, params in enumerate(fine_trials, start=1):
            result = run_trial(
                stage="fine",
                index=index,
                params=params,
                base_profile=base_profile,
                args=args,
            )
            fine_results.append(result)
            results.append(result)
            if not args.dry_run:
                write_search_results(output_root=output_root, results=results)

    best = max(results, key=lambda item: item["metric"])
    best_checkpoint_copy = copy_best_checkpoint(best=best, args=args)
    if best_checkpoint_copy is not None:
        best["best_checkpoint_copy"] = best_checkpoint_copy

    best_smoke_profile = build_best_smoke_profile(base_profile=base_profile, best=best, args=args)
    best_profile_path = output_root / "best_smoke_stage1_profile.yaml"

    full_transfer_profile_path = output_root / "full_stage1_from_smoke_best.yaml"
    if not args.dry_run:
        write_yaml(best_profile_path, best_smoke_profile)

    if resolve_repo_path(args.full_base_profile).exists() and not args.dry_run:
        full_profile = build_full_transfer_profile(load_yaml(args.full_base_profile), best)
        write_yaml(full_transfer_profile_path, full_profile)

    summary = {
        "base_profile": repo_path(args.base_profile),
        "search_mode": args.search_mode,
        "selection_metric": args.selection_metric,
        "coarse_trial_count": len(coarse_results),
        "fine_trial_count": len(fine_results),
        "trial_count": len(results),
        "top_coarse_trials": [
            flatten_result(result)
            for result in sorted(coarse_results, key=lambda item: item["metric"], reverse=True)[: max(args.fine_top_n, 0)]
        ],
        "best_trial": {
            key: value
            for key, value in best.items()
            if key not in {"command", "summary"}
        },
        "best_train_summary": best.get("summary", {}),
        "search_results_csv": repo_path(output_root / "search_results.csv"),
        "best_smoke_profile": repo_path(best_profile_path),
        "full_transfer_profile": repo_path(full_transfer_profile_path),
    }

    if not args.skip_final_eval:
        run_best_final_steps(best_smoke_profile, args=args)
        if not args.dry_run:
            patch_metrics_path = resolve_repo_path(Path(best_smoke_profile["eval"]["metrics_json"]))
            summary["best_final_patch_metrics_json"] = repo_path(patch_metrics_path)
            if patch_metrics_path.exists():
                summary["best_final_patch_metrics"] = read_json(patch_metrics_path)

            tile_cfg = best_smoke_profile.get("tile_eval", {})
            tile_metrics_raw = tile_cfg.get("metrics_json") if isinstance(tile_cfg, dict) else None
            if tile_metrics_raw:
                tile_metrics_path = resolve_repo_path(Path(str(tile_metrics_raw)))
                summary["best_final_tile_metrics_json"] = repo_path(tile_metrics_path)
                if tile_metrics_path.exists():
                    summary["best_final_tile_metrics"] = read_json(tile_metrics_path)

    if not args.dry_run:
        write_json(output_root / "search_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
