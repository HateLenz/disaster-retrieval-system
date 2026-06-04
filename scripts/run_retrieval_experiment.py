from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

STEP_TO_SCRIPT = {
    "subset": REPO_ROOT / "scripts" / "build_smoke_subset.py",
    "train": REPO_ROOT / "scripts" / "train_clip_visual_baseline.py",
    "extract": REPO_ROOT / "scripts" / "extract_pre_clip_features.py",
    "eval": REPO_ROOT / "scripts" / "eval_clip_retrieval.py",
    "tile_eval": REPO_ROOT / "scripts" / "eval_clip_tile_retrieval.py",
    "cases": REPO_ROOT / "scripts" / "export_retrieval_cases.py",
    "sanity": REPO_ROOT / "scripts" / "report_retrieval_sanity.py",
    "limitations": REPO_ROOT / "scripts" / "summarize_visual_limitations.py",
}

DEFAULT_STEPS = ["subset", "train", "extract", "eval", "tile_eval", "cases", "sanity", "limitations"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a stage-1 retrieval experiment from a single YAML profile.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Experiment YAML profile path.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", *DEFAULT_STEPS],
        help="Steps to run. Default is all.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to run child scripts.")
    return parser.parse_args()


def load_profile(profile_path: Path) -> dict[str, Any]:
    resolved = profile_path if profile_path.is_absolute() else (REPO_ROOT / profile_path).resolve()
    with resolved.open("r", encoding="utf-8") as file:
        profile = yaml.safe_load(file) or {}
    profile["_profile_path"] = str(resolved)
    return profile


def resolve_steps(raw_steps: list[str]) -> list[str]:
    if "all" in raw_steps:
        return DEFAULT_STEPS
    return raw_steps


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


def script_for_step(profile: dict[str, Any], step_name: str) -> Path:
    script_overrides = dict(profile.get("scripts", {}))
    raw_script = script_overrides.get(step_name)
    if raw_script is None:
        return STEP_TO_SCRIPT[step_name]
    script_path = Path(str(raw_script))
    return script_path if script_path.is_absolute() else (REPO_ROOT / script_path).resolve()


def subset_command(profile: dict[str, Any], python_executable: str) -> list[str] | None:
    subset_cfg = dict(profile.get("subset", {}))
    if not subset_cfg or not subset_cfg.get("enabled", False):
        return None

    command = [python_executable, str(script_for_step(profile, "subset"))]
    split_limits = subset_cfg.pop("split_limits", {})
    group_by = subset_cfg.pop("group_by", [])
    subset_cfg.pop("enabled", None)

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


def step_command(step_name: str, profile: dict[str, Any], python_executable: str) -> list[str] | None:
    if step_name == "subset":
        return subset_command(profile, python_executable=python_executable)

    params = dict(profile.get(step_name, {}))
    if not params:
        return None
    return command_from_mapping(
        script_path=script_for_step(profile, step_name),
        params=params,
        python_executable=python_executable,
    )


def log_directory(profile: dict[str, Any]) -> Path:
    name = str(profile.get("name", "retrieval_experiment"))
    raw_path = profile.get("log_dir")
    if raw_path:
        path = Path(str(raw_path))
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return (REPO_ROOT / "outputs" / "logs" / name).resolve()


def write_resolved_profile(profile: dict[str, Any], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = target_dir / "resolved_profile.json"
    with resolved_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, ensure_ascii=False, indent=2)


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as file:
        file.write(f"$ {' '.join(shlex.quote(part) for part in command)}\n\n")
        file.write(completed.stdout)
        if completed.stderr:
            file.write("\n[stderr]\n")
            file.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Step failed with exit code {completed.returncode}. See log: {log_path}"
        )


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    steps = resolve_steps(args.steps)
    logs_dir = log_directory(profile)
    if not args.dry_run:
        write_resolved_profile(profile=profile, target_dir=logs_dir)

    plan = []
    for step_name in steps:
        command = step_command(step_name=step_name, profile=profile, python_executable=args.python)
        if command is None:
            continue
        plan.append((step_name, command, logs_dir / f"{step_name}.log"))

    print(json.dumps({"profile": profile.get("name"), "steps": [step for step, _, _ in plan]}, ensure_ascii=False))
    for step_name, command, log_path in plan:
        print(f"[{step_name}] {' '.join(shlex.quote(part) for part in command)}")
        if args.dry_run:
            continue
        run_command(command=command, cwd=REPO_ROOT, log_path=log_path)
        print(f"[{step_name}] log saved to: {log_path}")


if __name__ == "__main__":
    main()
