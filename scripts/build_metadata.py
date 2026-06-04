from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw" / "xbd" / "geotiffs"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "xbd_tile_dataset.csv"
KEEP_TYPES = {"fire", "flooding", "wind"}
DAMAGE_LABELS = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
    "un-classified",
]
CSV_COLUMNS = [
    "tile_id",
    "pre_image_path",
    "post_image_path",
    "pre_label_path",
    "post_label_path",
    "disaster",
    "disaster_type",
    "split",
    "num_features",
    "num_no_damage",
    "num_minor_damage",
    "num_major_damage",
    "num_destroyed",
    "num_unclassified",
    "num_classified_features",
    "damage_ratio",
    "severe_damage_ratio",
    "text_prompt",
]


@dataclass(frozen=True)
class TileRecord:
    tile_id: str
    pre_image_path: str
    post_image_path: str
    pre_label_path: str
    post_label_path: str
    disaster: str
    disaster_type: str
    split: str
    num_features: int
    num_no_damage: int
    num_minor_damage: int
    num_major_damage: int
    num_destroyed: int
    num_unclassified: int
    num_classified_features: int
    damage_ratio: float
    severe_damage_ratio: float
    text_prompt: str

    def to_csv_row(self) -> dict[str, str | int]:
        return {
            "tile_id": self.tile_id,
            "pre_image_path": self.pre_image_path,
            "post_image_path": self.post_image_path,
            "pre_label_path": self.pre_label_path,
            "post_label_path": self.post_label_path,
            "disaster": self.disaster,
            "disaster_type": self.disaster_type,
            "split": self.split,
            "num_features": self.num_features,
            "num_no_damage": self.num_no_damage,
            "num_minor_damage": self.num_minor_damage,
            "num_major_damage": self.num_major_damage,
            "num_destroyed": self.num_destroyed,
            "num_unclassified": self.num_unclassified,
            "num_classified_features": self.num_classified_features,
            "damage_ratio": f"{self.damage_ratio:.6f}",
            "severe_damage_ratio": f"{self.severe_damage_ratio:.6f}",
            "text_prompt": self.text_prompt,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tile-level xBD dataset CSV filtered to fire, flooding, and wind. "
            "Each row corresponds to one pre/post tile pair."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing xBD split folders under geotiffs/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute file paths instead of repo-relative paths.",
    )
    return parser.parse_args()


def normalize_path(path: Path, absolute_paths: bool) -> str:
    normalized = path.resolve() if absolute_paths else path.resolve().relative_to(REPO_ROOT.resolve())
    return normalized.as_posix()


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_building_damage_counts(features: list[dict]) -> Counter:
    damage_counts: Counter = Counter()

    for feature in features:
        properties = feature.get("properties", {})
        if properties.get("feature_type") != "building":
            continue

        subtype = str(properties.get("subtype", "")).strip() or "un-classified"
        if subtype not in DAMAGE_LABELS:
            subtype = "un-classified"
        damage_counts[subtype] += 1

    return damage_counts


def build_tile_record(
    split_dir: Path,
    post_label_path: Path,
    post_label: dict,
    absolute_paths: bool,
) -> TileRecord:
    split = split_dir.name
    tile_id = post_label_path.name.replace("_post_disaster.json", "")
    label_dir = split_dir / "labels"
    image_dir = split_dir / "images"

    pre_label_path = label_dir / f"{tile_id}_pre_disaster.json"
    pre_image_path = image_dir / f"{tile_id}_pre_disaster.tif"
    post_image_path = image_dir / f"{tile_id}_post_disaster.tif"

    required_paths = [pre_label_path, post_label_path, pre_image_path, post_image_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"Missing files for tile {tile_id}: {', '.join(missing_paths)}"
        )

    metadata = post_label.get("metadata", {})
    disaster = str(metadata.get("disaster", "")).strip()
    disaster_type = str(metadata.get("disaster_type", "")).strip()

    features = post_label.get("features", {}).get("xy", [])
    damage_counts = collect_building_damage_counts(features)
    num_features = sum(damage_counts.values())
    num_no_damage = damage_counts.get("no-damage", 0)
    num_minor_damage = damage_counts.get("minor-damage", 0)
    num_major_damage = damage_counts.get("major-damage", 0)
    num_destroyed = damage_counts.get("destroyed", 0)
    num_unclassified = damage_counts.get("un-classified", 0)
    num_classified_features = (
        num_no_damage + num_minor_damage + num_major_damage + num_destroyed
    )

    if num_classified_features == 0:
        damage_ratio = 0.0
        severe_damage_ratio = 0.0
    else:
        damage_ratio = (
            num_minor_damage + num_major_damage + num_destroyed
        ) / num_classified_features
        severe_damage_ratio = (
            num_major_damage + num_destroyed
        ) / num_classified_features

    text_prompt = f"a post-disaster satellite tile after {disaster_type}"

    return TileRecord(
        tile_id=tile_id,
        pre_image_path=normalize_path(pre_image_path, absolute_paths),
        post_image_path=normalize_path(post_image_path, absolute_paths),
        pre_label_path=normalize_path(pre_label_path, absolute_paths),
        post_label_path=normalize_path(post_label_path, absolute_paths),
        disaster=disaster,
        disaster_type=disaster_type,
        split=split,
        num_features=num_features,
        num_no_damage=num_no_damage,
        num_minor_damage=num_minor_damage,
        num_major_damage=num_major_damage,
        num_destroyed=num_destroyed,
        num_unclassified=num_unclassified,
        num_classified_features=num_classified_features,
        damage_ratio=damage_ratio,
        severe_damage_ratio=severe_damage_ratio,
        text_prompt=text_prompt,
    )


def iter_tile_records(data_root: Path, absolute_paths: bool) -> list[TileRecord]:
    records: list[TileRecord] = []

    for split_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        label_dir = split_dir / "labels"
        for post_label_path in sorted(label_dir.glob("*_post_disaster.json")):
            post_label = load_json(post_label_path)
            disaster_type = str(post_label.get("metadata", {}).get("disaster_type", "")).strip()
            if disaster_type not in KEEP_TYPES:
                continue

            records.append(
                build_tile_record(
                    split_dir=split_dir,
                    post_label_path=post_label_path,
                    post_label=post_label,
                    absolute_paths=absolute_paths,
                )
            )

    if not records:
        raise FileNotFoundError(
            f"No matching post-disaster labels found under {data_root} for {sorted(KEEP_TYPES)}"
        )

    return sorted(records, key=lambda record: (record.split, record.disaster_type, record.tile_id))


def write_csv(records: list[TileRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())


def summarize(records: list[TileRecord]) -> str:
    count_by_type: Counter = Counter(record.disaster_type for record in records)
    count_by_split: Counter = Counter(record.split for record in records)
    feature_total = sum(record.num_features for record in records)
    classified_total = sum(record.num_classified_features for record in records)
    unclassified_total = sum(record.num_unclassified for record in records)
    summary_lines = [
        f"Total tiles kept: {len(records)}",
        f"Total features counted: {feature_total}",
        f"Total classified features: {classified_total}",
        f"Total un-classified features: {unclassified_total}",
        "Tiles by disaster_type: "
        + ", ".join(f"{dtype}={count_by_type[dtype]}" for dtype in sorted(count_by_type)),
        "Tiles by split: "
        + ", ".join(f"{split}={count_by_split[split]}" for split in sorted(count_by_split)),
    ]
    return "\n".join(summary_lines)


def main() -> None:
    args = parse_args()
    records = iter_tile_records(args.data_root, absolute_paths=args.absolute_paths)
    write_csv(records, args.output)
    print(f"Tile dataset written to: {args.output}")
    print(summarize(records))


if __name__ == "__main__":
    main()
