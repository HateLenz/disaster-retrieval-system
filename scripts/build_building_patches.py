from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.xbd_dataset import (
    DAMAGE_LABEL_TO_ID,
    extract_building_annotations,
    load_label_json,
)
DEFAULT_TILE_CSV = REPO_ROOT / "data" / "processed" / "xbd_tile_dataset.csv"
DEFAULT_PATCH_DIR = REPO_ROOT / "data" / "processed" / "xbd_building_patches"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches.csv"

CSV_COLUMNS = [
    "building_id",
    "positive_id",
    "building_uid",
    "tile_id",
    "pre_image_path",
    "post_image_path",
    "pre_label_path",
    "post_label_path",
    "pre_patch_path",
    "post_patch_path",
    "disaster",
    "disaster_type",
    "damage_label",
    "damage_id",
    "split",
    "polygon_wkt",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "pre_bbox_x1",
    "pre_bbox_y1",
    "pre_bbox_x2",
    "pre_bbox_y2",
    "post_bbox_x1",
    "post_bbox_y1",
    "post_bbox_x2",
    "post_bbox_y2",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "crop_width",
    "crop_height",
    "margin_px",
    "patch_size",
]


@dataclass(frozen=True)
class PatchRecord:
    building_id: str
    positive_id: str
    building_uid: str
    tile_id: str
    pre_image_path: str
    post_image_path: str
    pre_label_path: str
    post_label_path: str
    pre_patch_path: str
    post_patch_path: str
    disaster: str
    disaster_type: str
    damage_label: str
    damage_id: int
    split: str
    polygon_wkt: str
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    pre_bbox_x1: float
    pre_bbox_y1: float
    pre_bbox_x2: float
    pre_bbox_y2: float
    post_bbox_x1: float
    post_bbox_y1: float
    post_bbox_x2: float
    post_bbox_y2: float
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    crop_width: int
    crop_height: int
    margin_px: int
    patch_size: int

    def to_row(self) -> dict[str, str | int | float]:
        return {
            "building_id": self.building_id,
            "positive_id": self.positive_id,
            "building_uid": self.building_uid,
            "tile_id": self.tile_id,
            "pre_image_path": self.pre_image_path,
            "post_image_path": self.post_image_path,
            "pre_label_path": self.pre_label_path,
            "post_label_path": self.post_label_path,
            "pre_patch_path": self.pre_patch_path,
            "post_patch_path": self.post_patch_path,
            "disaster": self.disaster,
            "disaster_type": self.disaster_type,
            "damage_label": self.damage_label,
            "damage_id": self.damage_id,
            "split": self.split,
            "polygon_wkt": self.polygon_wkt,
            "bbox_x1": self.bbox_x1,
            "bbox_y1": self.bbox_y1,
            "bbox_x2": self.bbox_x2,
            "bbox_y2": self.bbox_y2,
            "pre_bbox_x1": f"{self.pre_bbox_x1:.3f}",
            "pre_bbox_y1": f"{self.pre_bbox_y1:.3f}",
            "pre_bbox_x2": f"{self.pre_bbox_x2:.3f}",
            "pre_bbox_y2": f"{self.pre_bbox_y2:.3f}",
            "post_bbox_x1": f"{self.post_bbox_x1:.3f}",
            "post_bbox_y1": f"{self.post_bbox_y1:.3f}",
            "post_bbox_x2": f"{self.post_bbox_x2:.3f}",
            "post_bbox_y2": f"{self.post_bbox_y2:.3f}",
            "crop_x1": self.crop_x1,
            "crop_y1": self.crop_y1,
            "crop_x2": self.crop_x2,
            "crop_y2": self.crop_y2,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "margin_px": self.margin_px,
            "patch_size": self.patch_size,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build building-level pre/post paired patches from xBD tile metadata.",
    )
    parser.add_argument(
        "--tile-csv",
        type=Path,
        default=DEFAULT_TILE_CSV,
        help="Input tile-level metadata CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output building-level patch CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PATCH_DIR,
        help="Directory used to save cropped patch PNG files.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Final square patch size.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=16,
        help="Context margin added around each building bbox.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=16,
        help="Minimum crop width and height after expansion.",
    )
    parser.add_argument(
        "--include-unclassified",
        action="store_true",
        help="Keep post-disaster buildings labeled as un-classified.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute file paths into the output CSV.",
    )
    parser.add_argument(
        "--stretch-mode",
        type=str,
        default="percentile",
        choices=("auto", "none", "minmax", "percentile"),
        help=(
            "How to convert TIFF values to uint8 RGB before cropping. "
            "'percentile' is the default training-oriented mode; "
            "'auto' keeps 0..255 images as-is and otherwise uses percentile stretch."
        ),
    )
    parser.add_argument(
        "--stretch-scope",
        type=str,
        default="pair",
        choices=("image", "pair"),
        help=(
            "Whether stretch statistics are computed per image or shared by the pre/post pair."
        ),
    )
    parser.add_argument(
        "--stretch-low-percentile",
        type=float,
        default=2.0,
        help="Lower percentile used by percentile stretch.",
    )
    parser.add_argument(
        "--stretch-high-percentile",
        type=float,
        default=98.0,
        help="Upper percentile used by percentile stretch.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional split filter, e.g. --splits tier3 hold test.",
    )
    parser.add_argument(
        "--disaster-types",
        nargs="*",
        default=None,
        help="Optional disaster-type filter, e.g. --disaster-types fire flooding wind.",
    )
    parser.add_argument(
        "--limit-tiles",
        type=int,
        default=None,
        help="Optional cap on the number of tiles to process after filtering.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
        help="Print a progress update every N processed tiles. Set to 0 to disable.",
    )
    return parser.parse_args()


def normalize_path(path: Path, absolute_paths: bool) -> str:
    normalized = path.resolve() if absolute_paths else path.resolve().relative_to(REPO_ROOT.resolve())
    return normalized.as_posix()


def resolve_input_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def sanitize_uid(uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", uid)


def build_annotation_lookup(label_data: dict) -> dict[str, dict]:
    features = label_data.get("features", {}).get("xy", [])
    annotations = extract_building_annotations(
        label_data,
        image_size=None,
        include_unclassified=True,
    )
    feature_by_uid = {
        str(feature.get("properties", {}).get("uid", "")): feature
        for feature in features
        if feature.get("properties", {}).get("feature_type") == "building"
    }

    lookup: dict[str, dict] = {}
    for annotation in annotations:
        feature = feature_by_uid.get(annotation.uid)
        if feature is None:
            continue
        lookup[annotation.uid] = {
            "annotation": annotation,
            "wkt": str(feature.get("wkt", "")),
        }
    return lookup


def expand_box(
    box: tuple[float, float, float, float],
    margin: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1 = max(0, int(math.floor(box[0] - margin)))
    y1 = max(0, int(math.floor(box[1] - margin)))
    x2 = min(width, int(math.ceil(box[2] + margin)))
    y2 = min(height, int(math.ceil(box[3] + margin)))
    return x1, y1, x2, y2


def union_box(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def save_patch_image(
    source_image: Image.Image,
    crop_box: tuple[int, int, int, int],
    output_path: Path,
    patch_size: int,
) -> None:
    patch = source_image.crop(crop_box)
    patch = ImageOps.pad(
        patch,
        size=(patch_size, patch_size),
        method=Image.Resampling.BICUBIC,
        color=(0, 0, 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch.save(output_path)


def ensure_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    return image


def candidate_tifffile_site_packages() -> list[Path]:
    candidates: list[Path] = []

    python_executables = []
    for command_name in ("python", "python3"):
        executable = shutil.which(command_name)
        if executable is not None:
            python_executables.append(Path(executable).resolve())

    current_python = Path(sys.executable).resolve()
    for executable in python_executables:
        if executable == current_python:
            continue

        try:
            result = subprocess.run(
                [
                    str(executable),
                    "-c",
                    "import json, site; print(json.dumps(site.getsitepackages()))",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            discovered = json.loads(result.stdout.strip())
        except Exception:
            continue

        for site_package in discovered:
            candidate = Path(site_package)
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates


@lru_cache(maxsize=1)
def load_tifffile_module() -> ModuleType | None:
    try:
        return importlib.import_module("tifffile")
    except ModuleNotFoundError:
        pass

    for site_package in candidate_tifffile_site_packages():
        if not (site_package / "tifffile").exists():
            continue

        site_package_str = str(site_package)
        if site_package_str not in sys.path:
            sys.path.insert(0, site_package_str)
        importlib.invalidate_caches()
        try:
            return importlib.import_module("tifffile")
        except ModuleNotFoundError:
            continue

    return None


def load_raw_tiff(image_path: Path) -> tuple[np.ndarray, str]:
    tifffile_module = load_tifffile_module()
    if tifffile_module is not None:
        image = tifffile_module.imread(image_path)
        image = ensure_hwc(image)
        if image.ndim == 2:
            return image, "gray"
        if image.ndim == 3 and image.shape[-1] == 4:
            return image, "rgba"
        return image, "rgb"

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    image = ensure_hwc(image)
    if image.ndim == 2:
        return image, "gray"
    if image.ndim == 3 and image.shape[-1] == 4:
        return image, "bgra"
    return image, "bgr"


def prepare_channel_view(image: np.ndarray) -> np.ndarray:
    image = ensure_hwc(image).astype(np.float32)
    if image.ndim == 2:
        image = image[..., None]
    return image


def prepare_channel_view_raw(image: np.ndarray) -> np.ndarray:
    image = ensure_hwc(np.asarray(image))
    if image.ndim == 2:
        image = image[..., None]
    return image


def select_stretch_mode(image: np.ndarray, stretch_mode: str) -> str:
    if stretch_mode != "auto":
        return stretch_mode

    image_view = prepare_channel_view(image)
    if float(image_view.min()) >= 0.0 and float(image_view.max()) <= 255.0:
        return "none"
    return "percentile"


def compute_stretch_stats(
    images: list[np.ndarray],
    stretch_mode: str,
    stretch_percentiles: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    if stretch_mode not in {"minmax", "percentile"}:
        return None

    prepared = [prepare_channel_view_raw(image) for image in images]
    channel_count = prepared[0].shape[-1]
    if any(image.shape[-1] != channel_count for image in prepared[1:]):
        return None

    low_p, high_p = stretch_percentiles
    low_values: list[float] = []
    high_values: list[float] = []
    for channel_idx in range(channel_count):
        channel_values = np.concatenate(
            [image[..., channel_idx].reshape(-1) for image in prepared],
            axis=0,
        )
        if stretch_mode == "minmax":
            lo = float(channel_values.min())
            hi = float(channel_values.max())
        else:
            if (
                np.issubdtype(channel_values.dtype, np.integer)
                and int(channel_values.min()) >= 0
                and int(channel_values.max()) <= 255
            ):
                histogram = np.bincount(channel_values.astype(np.uint8), minlength=256)
                cumulative = np.cumsum(histogram)
                total = int(cumulative[-1])
                low_rank = total * (low_p / 100.0)
                high_rank = total * (high_p / 100.0)
                lo = float(np.searchsorted(cumulative, low_rank, side="left"))
                hi = float(np.searchsorted(cumulative, high_rank, side="left"))
            else:
                lo, hi = np.percentile(
                    channel_values.astype(np.float32),
                    [low_p, high_p],
                ).astype(np.float32)
            if hi <= lo:
                lo = float(channel_values.min())
                hi = float(channel_values.max())

        low_values.append(lo)
        high_values.append(hi)

    return np.asarray(low_values, dtype=np.float32), np.asarray(high_values, dtype=np.float32)


def to_uint8_image(
    image: np.ndarray,
    stretch_mode: str,
    stretch_percentiles: tuple[float, float],
    stretch_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    effective_mode = select_stretch_mode(image, stretch_mode)
    image_view = prepare_channel_view(image)

    if effective_mode == "none":
        clipped = np.clip(image_view, 0.0, 255.0).astype(np.uint8)
        return clipped[..., 0] if clipped.shape[-1] == 1 else clipped

    if stretch_stats is None:
        stretch_stats = compute_stretch_stats(
            images=[image_view],
            stretch_mode=effective_mode,
            stretch_percentiles=stretch_percentiles,
        )

    if stretch_stats is None:
        clipped = np.clip(image_view, 0.0, 255.0).astype(np.uint8)
        return clipped[..., 0] if clipped.shape[-1] == 1 else clipped

    low_values, high_values = stretch_stats
    display = np.empty_like(image_view, dtype=np.uint8)
    for channel_idx in range(image_view.shape[-1]):
        channel = image_view[..., channel_idx]
        lo = float(low_values[min(channel_idx, len(low_values) - 1)])
        hi = float(high_values[min(channel_idx, len(high_values) - 1)])
        if hi <= lo:
            scaled = np.zeros_like(channel, dtype=np.uint8)
        else:
            clipped = np.clip(channel, lo, hi)
            scaled = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
        display[..., channel_idx] = scaled

    return display[..., 0] if display.shape[-1] == 1 else display


def open_rgb_image(
    image: np.ndarray,
    color_layout: str,
    stretch_mode: str,
    stretch_percentiles: tuple[float, float],
    stretch_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> Image.Image:
    image = to_uint8_image(
        image=image,
        stretch_mode=stretch_mode,
        stretch_percentiles=stretch_percentiles,
        stretch_stats=stretch_stats,
    )
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
        return Image.fromarray(image, mode="RGB")

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape after TIFF conversion: {image.shape}")

    channels = image.shape[2]
    if channels == 1:
        rgb_image = np.repeat(image, 3, axis=2)
    elif channels == 3 and color_layout == "bgr":
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif channels == 4 and color_layout == "bgra":
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif channels == 4 and color_layout == "rgba":
        rgb_image = image[:, :, :3]
    else:
        rgb_image = image[:, :, :3]

    return Image.fromarray(rgb_image, mode="RGB")


def filter_tile_table(
    tile_table: pd.DataFrame,
    splits: list[str] | None,
    disaster_types: list[str] | None,
    limit_tiles: int | None,
) -> pd.DataFrame:
    filtered = tile_table.copy()
    if splits:
        filtered = filtered.loc[filtered["split"].astype(str).isin([str(split) for split in splits])]
    if disaster_types:
        filtered = filtered.loc[filtered["disaster_type"].astype(str).isin([str(item) for item in disaster_types])]

    filtered = filtered.sort_values(
        by=["split", "disaster_type", "tile_id"],
        kind="stable",
    ).reset_index(drop=True)
    if limit_tiles is not None:
        filtered = filtered.iloc[:limit_tiles].reset_index(drop=True)
    return filtered


def iter_patch_records(
    tile_table: pd.DataFrame,
    output_dir: Path,
    patch_size: int,
    margin: int,
    min_size: int,
    absolute_paths: bool,
    include_unclassified: bool,
    stretch_mode: str,
    stretch_scope: str,
    stretch_percentiles: tuple[float, float],
    stats: dict[str, int],
    log_every: int,
) -> Iterable[PatchRecord]:
    tile_rows = tile_table.to_dict("records")
    total_tiles = len(tile_rows)
    for tile_index, row in enumerate(tile_rows, start=1):
        stats["tiles"] += 1
        tile_id = str(row["tile_id"])
        split = str(row["split"])
        disaster = str(row["disaster"])
        disaster_type = str(row["disaster_type"])

        pre_image_path = resolve_input_path(row["pre_image_path"])
        post_image_path = resolve_input_path(row["post_image_path"])
        pre_label_path = resolve_input_path(row["pre_label_path"])
        post_label_path = resolve_input_path(row["post_label_path"])

        pre_label = load_label_json(pre_label_path)
        post_label = load_label_json(post_label_path)
        pre_lookup = build_annotation_lookup(pre_label)
        post_lookup = build_annotation_lookup(post_label)

        pre_raw_image, pre_color_layout = load_raw_tiff(pre_image_path)
        post_raw_image, post_color_layout = load_raw_tiff(post_image_path)

        shared_stretch_stats = None
        if stretch_scope == "pair":
            effective_modes = {
                select_stretch_mode(pre_raw_image, stretch_mode),
                select_stretch_mode(post_raw_image, stretch_mode),
            }
            if len(effective_modes) == 1:
                effective_mode = next(iter(effective_modes))
                shared_stretch_stats = compute_stretch_stats(
                    images=[pre_raw_image, post_raw_image],
                    stretch_mode=effective_mode,
                    stretch_percentiles=stretch_percentiles,
                )

        with open_rgb_image(
            pre_raw_image,
            color_layout=pre_color_layout,
            stretch_mode=stretch_mode,
            stretch_percentiles=stretch_percentiles,
            stretch_stats=shared_stretch_stats,
        ) as pre_image, open_rgb_image(
            post_raw_image,
            color_layout=post_color_layout,
            stretch_mode=stretch_mode,
            stretch_percentiles=stretch_percentiles,
            stretch_stats=shared_stretch_stats,
        ) as post_image:

            for uid in sorted(post_lookup):
                if uid not in pre_lookup:
                    stats["skipped_missing_uid"] += 1
                    continue

                post_item = post_lookup[uid]
                pre_item = pre_lookup[uid]
                post_annotation = post_item["annotation"]
                pre_annotation = pre_item["annotation"]

                damage_label = post_annotation.damage_label
                if damage_label == "un-classified" and not include_unclassified:
                    continue

                merged_box = union_box(pre_annotation.bbox, post_annotation.bbox)
                merged_box_int = (
                    int(math.floor(merged_box[0])),
                    int(math.floor(merged_box[1])),
                    int(math.ceil(merged_box[2])),
                    int(math.ceil(merged_box[3])),
                )
                crop_box = expand_box(merged_box, margin=margin, image_size=pre_image.size)
                crop_width = crop_box[2] - crop_box[0]
                crop_height = crop_box[3] - crop_box[1]
                if crop_width < min_size or crop_height < min_size:
                    stats["skipped_small_box"] += 1
                    continue

                safe_uid = sanitize_uid(uid)
                building_id = f"{tile_id}__{safe_uid}"
                positive_id = building_id
                base_dir = output_dir / split / disaster_type / tile_id
                pre_patch_path = base_dir / f"{safe_uid}_pre.png"
                post_patch_path = base_dir / f"{safe_uid}_post.png"

                save_patch_image(pre_image, crop_box, pre_patch_path, patch_size=patch_size)
                save_patch_image(post_image, crop_box, post_patch_path, patch_size=patch_size)

                yield PatchRecord(
                    building_id=building_id,
                    positive_id=positive_id,
                    building_uid=uid,
                    tile_id=tile_id,
                    pre_image_path=normalize_path(pre_image_path, absolute_paths),
                    post_image_path=normalize_path(post_image_path, absolute_paths),
                    pre_label_path=normalize_path(pre_label_path, absolute_paths),
                    post_label_path=normalize_path(post_label_path, absolute_paths),
                    pre_patch_path=normalize_path(pre_patch_path, absolute_paths),
                    post_patch_path=normalize_path(post_patch_path, absolute_paths),
                    disaster=disaster,
                    disaster_type=disaster_type,
                    damage_label=damage_label,
                    damage_id=DAMAGE_LABEL_TO_ID[damage_label],
                    split=split,
                    polygon_wkt=post_item["wkt"],
                    bbox_x1=merged_box_int[0],
                    bbox_y1=merged_box_int[1],
                    bbox_x2=merged_box_int[2],
                    bbox_y2=merged_box_int[3],
                    pre_bbox_x1=pre_annotation.bbox[0],
                    pre_bbox_y1=pre_annotation.bbox[1],
                    pre_bbox_x2=pre_annotation.bbox[2],
                    pre_bbox_y2=pre_annotation.bbox[3],
                    post_bbox_x1=post_annotation.bbox[0],
                    post_bbox_y1=post_annotation.bbox[1],
                    post_bbox_x2=post_annotation.bbox[2],
                    post_bbox_y2=post_annotation.bbox[3],
                    crop_x1=crop_box[0],
                    crop_y1=crop_box[1],
                    crop_x2=crop_box[2],
                    crop_y2=crop_box[3],
                    crop_width=crop_width,
                    crop_height=crop_height,
                    margin_px=margin,
                    patch_size=patch_size,
                )
                stats["pairs"] += 1

        if log_every > 0 and (tile_index == 1 or tile_index % log_every == 0 or tile_index == total_tiles):
            print(
                json.dumps(
                    {
                        "event": "build_building_patches_progress",
                        "tiles_processed": tile_index,
                        "tiles_total": total_tiles,
                        "paired_buildings": stats["pairs"],
                        "skipped_missing_uid": stats["skipped_missing_uid"],
                        "skipped_small_box": stats["skipped_small_box"],
                        "current_split": split,
                        "current_disaster_type": disaster_type,
                        "current_tile_id": tile_id,
                    },
                    ensure_ascii=False,
                )
            )


def write_csv(records: Iterable[PatchRecord], output_csv: Path) -> dict[str, object]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_counts: Counter[str] = Counter()
    split_disaster_counts: Counter[tuple[str, str]] = Counter()
    damage_counts: Counter[str] = Counter()
    row_count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
            row_count += 1
            split_counts[record.split] += 1
            split_disaster_counts[(record.split, record.disaster_type)] += 1
            damage_counts[record.damage_label] += 1
    return {
        "rows": row_count,
        "by_split": dict(split_counts),
        "by_split_disaster_type": {
            f"{split}/{disaster_type}": count
            for (split, disaster_type), count in sorted(split_disaster_counts.items())
        },
        "by_damage_label": dict(sorted(damage_counts.items())),
    }


def main() -> None:
    args = parse_args()
    tile_table = pd.read_csv(args.tile_csv)
    tile_table = filter_tile_table(
        tile_table=tile_table,
        splits=args.splits,
        disaster_types=args.disaster_types,
        limit_tiles=args.limit_tiles,
    )
    stats = {
        "tiles": 0,
        "pairs": 0,
        "skipped_missing_uid": 0,
        "skipped_small_box": 0,
    }
    write_summary = write_csv(
        records=iter_patch_records(
            tile_table=tile_table,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            margin=args.margin,
            min_size=args.min_size,
            absolute_paths=args.absolute_paths,
            include_unclassified=args.include_unclassified,
            stretch_mode=args.stretch_mode,
            stretch_scope=args.stretch_scope,
            stretch_percentiles=(
                float(args.stretch_low_percentile),
                float(args.stretch_high_percentile),
            ),
            stats=stats,
            log_every=max(int(args.log_every), 0),
        ),
        output_csv=args.output_csv,
    )
    print(f"Building patch dataset written to: {args.output_csv}")
    print(f"Tiles processed: {stats['tiles']}")
    print(f"Paired building patches: {stats['pairs']}")
    print(f"Skipped missing pre/post uid matches: {stats['skipped_missing_uid']}")
    print(f"Skipped small crops: {stats['skipped_small_box']}")
    print(json.dumps(write_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
