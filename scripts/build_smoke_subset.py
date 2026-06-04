from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches_smoke.csv"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "outputs" / "logs" / "smoke_subset_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a representative smoke subset for stage-1 image retrieval experiments.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_SOURCE_CSV, help="Source building-patch CSV.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output smoke CSV.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["disaster_type", "damage_label"],
        help="Columns used to preserve semantic diversity during sampling.",
    )
    parser.add_argument(
        "--split-limit",
        action="append",
        default=[],
        help="Repeated split=count rule, for example --split-limit tier3=4096 --split-limit hold=1024.",
    )
    parser.add_argument(
        "--max-per-tile",
        type=int,
        default=24,
        help="Maximum sampled rows per tile within each split.",
    )
    parser.add_argument(
        "--max-per-disaster",
        type=int,
        default=0,
        help="Optional maximum sampled rows per disaster within each split. Zero disables the cap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-unclassified", action="store_true")
    return parser.parse_args()


def parse_split_limits(items: list[str]) -> dict[str, int]:
    if not items:
        return {"tier3": 4096, "hold": 1024}

    split_limits: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid split limit: {item}")
        split_name, raw_count = item.split("=", maxsplit=1)
        split_limits[split_name.strip()] = int(raw_count.strip())
    return split_limits


def validate_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in source CSV: {missing}")


def stable_text_seed(value: object) -> int:
    text = str(value)
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % 10_000


def apply_tile_cap(frame: pd.DataFrame, max_per_tile: int, seed: int) -> pd.DataFrame:
    if max_per_tile <= 0 or "tile_id" not in frame.columns:
        return frame

    sampled_frames: list[pd.DataFrame] = []
    for tile_id, tile_frame in frame.groupby("tile_id", sort=False):
        take_count = min(len(tile_frame), max_per_tile)
        sampled_frames.append(tile_frame.sample(n=take_count, random_state=seed + stable_text_seed(tile_id)))
    return pd.concat(sampled_frames, ignore_index=True)


def round_robin_sample(
    frame: pd.DataFrame,
    limit: int,
    group_by: list[str],
    max_per_tile: int,
    max_per_disaster: int,
    seed: int,
) -> pd.DataFrame:
    if limit <= 0 or frame.empty:
        return frame.iloc[0:0].copy()

    working = frame.copy()
    if max_per_tile > 0:
        working = apply_tile_cap(working, max_per_tile=max_per_tile, seed=seed)

    working = working.reset_index(drop=True)
    working["_row_id"] = np.arange(len(working))
    working["_group_key"] = working[group_by].astype(str).agg(" | ".join, axis=1)
    working = working.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    group_frames: dict[str, pd.DataFrame] = {
        str(group_key): group_frame.reset_index(drop=True)
        for group_key, group_frame in working.groupby("_group_key", sort=True)
    }
    group_positions = {group_key: 0 for group_key in group_frames}
    disaster_counts: Counter[str] = Counter()
    selected_indices: list[int] = []

    while len(selected_indices) < limit:
        made_progress = False
        for group_key in group_frames:
            group_frame = group_frames[group_key]
            position = group_positions[group_key]
            while position < len(group_frame):
                row = group_frame.iloc[position]
                position += 1
                disaster_name = str(row["disaster"]) if "disaster" in row else ""
                if max_per_disaster > 0 and disaster_name and disaster_counts[disaster_name] >= max_per_disaster:
                    continue
                selected_indices.append(int(row["_row_id"]))
                if disaster_name:
                    disaster_counts[disaster_name] += 1
                made_progress = True
                break
            group_positions[group_key] = position
            if len(selected_indices) >= limit:
                break
        if not made_progress:
            break

    sampled = working.set_index("_row_id").loc[selected_indices].reset_index(drop=True)
    sampled = sampled.drop(columns=["_group_key"]).copy()
    return sampled.reset_index(drop=True)


def build_summary(source_frame: pd.DataFrame, sampled_frame: pd.DataFrame, split_limits: dict[str, int]) -> dict[str, object]:
    def counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
        if column not in frame.columns:
            return {}
        return {str(key): int(value) for key, value in frame[column].value_counts().sort_index().items()}

    summary: dict[str, object] = {
        "source_rows": int(len(source_frame)),
        "sampled_rows": int(len(sampled_frame)),
        "requested_split_limits": split_limits,
        "sampled_unique_tiles": int(sampled_frame["tile_id"].nunique()) if "tile_id" in sampled_frame.columns else 0,
        "sampled_unique_disasters": int(sampled_frame["disaster"].nunique()) if "disaster" in sampled_frame.columns else 0,
        "sampled_by_split": counts(sampled_frame, "split"),
        "sampled_by_disaster_type": counts(sampled_frame, "disaster_type"),
        "sampled_by_damage_label": counts(sampled_frame, "damage_label"),
    }

    per_split: dict[str, object] = {}
    for split_name, split_frame in sampled_frame.groupby("split", sort=True):
        per_split[str(split_name)] = {
            "rows": int(len(split_frame)),
            "unique_tiles": int(split_frame["tile_id"].nunique()) if "tile_id" in split_frame.columns else 0,
            "by_disaster_type": counts(split_frame, "disaster_type"),
            "by_damage_label": counts(split_frame, "damage_label"),
        }
    summary["per_split"] = per_split
    return summary


def main() -> None:
    args = parse_args()
    split_limits = parse_split_limits(args.split_limit)

    frame = pd.read_csv(args.csv)
    validate_columns(frame, ["split", *args.group_by])
    if not args.include_unclassified and "damage_label" in frame.columns:
        frame = frame.loc[frame["damage_label"] != "un-classified"].copy()

    sampled_parts: list[pd.DataFrame] = []
    for split_name, limit in split_limits.items():
        split_frame = frame.loc[frame["split"] == split_name].copy()
        if split_frame.empty:
            continue
        sampled_parts.append(
            round_robin_sample(
                frame=split_frame,
                limit=limit,
                group_by=args.group_by,
                max_per_tile=args.max_per_tile,
                max_per_disaster=args.max_per_disaster,
                seed=args.seed + len(sampled_parts),
            )
        )

    sampled_frame = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else frame.iloc[0:0].copy()
    if {"split", "disaster_type", "tile_id", "positive_id"}.issubset(sampled_frame.columns):
        sampled_frame = sampled_frame.sort_values(
            by=["split", "disaster_type", "tile_id", "positive_id"],
            kind="stable",
        ).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled_frame.to_csv(args.output_csv, index=False)

    summary = build_summary(source_frame=frame, sampled_frame=sampled_frame, split_limits=split_limits)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Smoke subset saved to: {args.output_csv}")
    print(f"Subset summary saved to: {args.summary_json}")


if __name__ == "__main__":
    main()
