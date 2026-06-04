from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    class TorchDataset:
        """Fallback base class so this module stays importable without torch."""


REPO_ROOT = Path(__file__).resolve().parents[2]
DAMAGE_LABEL_TO_ID = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 5,
}


@dataclass(frozen=True)
class BuildingAnnotation:
    uid: str
    damage_label: str
    damage_id: int
    bbox: tuple[float, float, float, float]
    rings: tuple[np.ndarray, ...]


def resolve_repo_path(path_value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_label_json(label_path: str | Path) -> dict[str, Any]:
    with Path(label_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_image_size_from_label(label_data: dict[str, Any]) -> tuple[int, int] | None:
    metadata = label_data.get("metadata", {})
    width = metadata.get("width") or metadata.get("original_width")
    height = metadata.get("height") or metadata.get("original_height")
    if width is None or height is None:
        return None
    return int(width), int(height)


def normalize_damage_label(raw_label: str | None) -> str:
    damage_label = str(raw_label or "").strip() or "un-classified"
    if damage_label not in DAMAGE_LABEL_TO_ID:
        return "un-classified"
    return damage_label


def parse_polygon_wkt(wkt: str) -> tuple[np.ndarray, ...]:
    text = wkt.strip()
    if not text.upper().startswith("POLYGON"):
        raise ValueError(f"Unsupported WKT geometry: {wkt[:32]}")

    match = re.fullmatch(r"POLYGON\s*\(\((.*)\)\)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Malformed polygon WKT: {wkt[:32]}")

    ring_blocks = re.split(r"\)\s*,\s*\(", match.group(1))
    rings: list[np.ndarray] = []
    for ring_block in ring_blocks:
        points: list[tuple[float, float]] = []
        for point_text in ring_block.split(","):
            coords = point_text.strip().split()
            if len(coords) < 2:
                continue
            points.append((float(coords[0]), float(coords[1])))

        if len(points) >= 3:
            rings.append(np.asarray(points, dtype=np.float32))

    if not rings:
        raise ValueError(f"Empty polygon WKT: {wkt[:32]}")

    return tuple(rings)


def clip_ring(ring: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    clipped = ring.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(width - 1, 0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(height - 1, 0))
    return clipped


def extract_building_annotations(
    label_data: dict[str, Any],
    image_size: tuple[int, int] | None = None,
    include_unclassified: bool = True,
) -> list[BuildingAnnotation]:
    features = label_data.get("features", {}).get("xy", [])
    annotations: list[BuildingAnnotation] = []

    for feature in features:
        properties = feature.get("properties", {})
        if properties.get("feature_type") != "building":
            continue

        damage_label = normalize_damage_label(properties.get("subtype"))
        if not include_unclassified and damage_label == "un-classified":
            continue

        rings = parse_polygon_wkt(str(feature.get("wkt", "")))
        if image_size is not None:
            rings = tuple(clip_ring(ring, image_size) for ring in rings)

        all_points = np.concatenate(rings, axis=0)
        bbox = (
            float(all_points[:, 0].min()),
            float(all_points[:, 1].min()),
            float(all_points[:, 0].max()),
            float(all_points[:, 1].max()),
        )
        annotations.append(
            BuildingAnnotation(
                uid=str(properties.get("uid", "")),
                damage_label=damage_label,
                damage_id=DAMAGE_LABEL_TO_ID[damage_label],
                bbox=bbox,
                rings=rings,
            )
        )

    return annotations


def _draw_annotation(draw: ImageDraw.ImageDraw, annotation: BuildingAnnotation, fill_value: int) -> None:
    outer_ring = [tuple(point) for point in annotation.rings[0]]
    draw.polygon(outer_ring, fill=fill_value)
    for hole_ring in annotation.rings[1:]:
        draw.polygon([tuple(point) for point in hole_ring], fill=0)


def build_building_mask(
    annotations: list[BuildingAnnotation],
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for annotation in annotations:
        _draw_annotation(draw, annotation, fill_value=1)
    return np.asarray(mask_image, dtype=np.uint8)


def build_damage_mask(
    annotations: list[BuildingAnnotation],
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for annotation in annotations:
        _draw_annotation(draw, annotation, fill_value=annotation.damage_id)
    return np.asarray(mask_image, dtype=np.uint8)


def count_damage_labels(annotations: list[BuildingAnnotation]) -> dict[str, int]:
    counts = {label: 0 for label in DAMAGE_LABEL_TO_ID}
    for annotation in annotations:
        counts[annotation.damage_label] += 1
    return counts


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if pd.isna(value):
        return default
    return int(value)


class XBDTileDataset(TorchDataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str | None = None,
        repo_root: str | Path = REPO_ROOT,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        image_mode: str | None = "RGB",
        label_source: str = "post",
        include_unclassified: bool = True,
        load_images: bool = True,
        load_masks: bool = True,
        return_polygons: bool = False,
    ) -> None:
        if label_source not in {"pre", "post"}:
            raise ValueError("label_source must be 'pre' or 'post'")

        records = pd.read_csv(csv_path)
        if split is not None:
            records = records.loc[records["split"] == split]

        self.records = records.to_dict("records")
        self.repo_root = Path(repo_root).resolve()
        self.transform = transform
        self.image_mode = image_mode
        self.label_source = label_source
        self.include_unclassified = include_unclassified
        self.load_images = load_images
        self.load_masks = load_masks
        self.return_polygons = return_polygons

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        pre_image_path = resolve_repo_path(row["pre_image_path"], self.repo_root)
        post_image_path = resolve_repo_path(row["post_image_path"], self.repo_root)
        pre_label_path = resolve_repo_path(row["pre_label_path"], self.repo_root)
        post_label_path = resolve_repo_path(row["post_label_path"], self.repo_root)

        pre_image = self._load_image(pre_image_path) if self.load_images else None
        post_image = self._load_image(post_image_path) if self.load_images else None

        label_path = post_label_path if self.label_source == "post" else pre_label_path
        label_data = load_label_json(label_path)
        image_size = self._resolve_image_size(label_data, pre_image, post_image)
        annotations = extract_building_annotations(
            label_data,
            image_size=image_size,
            include_unclassified=self.include_unclassified,
        )

        counts = count_damage_labels(annotations)
        target: dict[str, Any] = {
            "boxes": np.asarray([annotation.bbox for annotation in annotations], dtype=np.float32)
            if annotations
            else np.zeros((0, 4), dtype=np.float32),
            "damage_labels": np.asarray(
                [annotation.damage_id for annotation in annotations],
                dtype=np.int64,
            ),
            "damage_label_names": [annotation.damage_label for annotation in annotations],
            "building_uids": [annotation.uid for annotation in annotations],
            "num_features": _coerce_int(row.get("num_features"), len(annotations)),
            "num_classified_features": _coerce_int(
                row.get("num_classified_features"),
                counts["no-damage"]
                + counts["minor-damage"]
                + counts["major-damage"]
                + counts["destroyed"],
            ),
            "num_unclassified": _coerce_int(row.get("num_unclassified"), counts["un-classified"]),
        }

        if self.load_masks:
            target["building_mask"] = build_building_mask(annotations, image_size)
            target["damage_mask"] = build_damage_mask(annotations, image_size)

        if self.return_polygons:
            target["polygons"] = [
                [ring.copy() for ring in annotation.rings]
                for annotation in annotations
            ]

        sample = {
            "tile_id": str(row["tile_id"]),
            "split": str(row["split"]),
            "disaster": str(row["disaster"]),
            "disaster_type": str(row["disaster_type"]),
            "text_prompt": str(row.get("text_prompt", "")),
            "pre_image_path": pre_image_path,
            "post_image_path": post_image_path,
            "pre_label_path": pre_label_path,
            "post_label_path": post_label_path,
            "pre_image": pre_image,
            "post_image": post_image,
            "target": target,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        if image.ndim == 2:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.image_mode is not None:
            pil_image = pil_image.convert(self.image_mode)
        return np.asarray(pil_image)

    def _resolve_image_size(
        self,
        label_data: dict[str, Any],
        pre_image: np.ndarray | None,
        post_image: np.ndarray | None,
    ) -> tuple[int, int]:
        if post_image is not None:
            return int(post_image.shape[1]), int(post_image.shape[0])
        if pre_image is not None:
            return int(pre_image.shape[1]), int(pre_image.shape[0])

        image_size = infer_image_size_from_label(label_data)
        if image_size is None:
            raise ValueError("Unable to infer image size from label metadata")
        return image_size


def collate_xbd_tiles(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


__all__ = [
    "BuildingAnnotation",
    "DAMAGE_LABEL_TO_ID",
    "XBDTileDataset",
    "build_building_mask",
    "build_damage_mask",
    "collate_xbd_tiles",
    "extract_building_annotations",
    "infer_image_size_from_label",
    "load_label_json",
    "parse_polygon_wkt",
    "resolve_repo_path",
]
