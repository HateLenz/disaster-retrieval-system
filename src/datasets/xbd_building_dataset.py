from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_building_patch_table(
    csv_path: str | Path,
    split: str | None = None,
    include_unclassified: bool = False,
    disaster_types: list[str] | None = None,
) -> pd.DataFrame:
    table = pd.read_csv(csv_path)
    if split is not None:
        table = table.loc[table["split"] == split]
    if not include_unclassified and "damage_label" in table.columns:
        table = table.loc[table["damage_label"] != "un-classified"]
    if disaster_types:
        table = table.loc[table["disaster_type"].isin(disaster_types)]
    table = table.sort_values(
        by=["split", "disaster_type", "tile_id", "positive_id"],
        kind="stable",
    )
    return table.reset_index(drop=True)


class XBDBuildingPatchDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str | None = None,
        repo_root: str | Path = REPO_ROOT,
        transform: Callable[[Image.Image], Any] | None = None,
        include_unclassified: bool = False,
        disaster_types: list[str] | None = None,
        view: str = "pair",
    ) -> None:
        if view not in {"pair", "pre", "post"}:
            raise ValueError("view must be one of {'pair', 'pre', 'post'}")

        table = load_building_patch_table(
            csv_path=csv_path,
            split=split,
            include_unclassified=include_unclassified,
            disaster_types=disaster_types,
        )
        self.records = table.to_dict("records")
        self.repo_root = Path(repo_root).resolve()
        self.transform = transform
        self.view = view

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        pre_patch_path = resolve_repo_path(row["pre_patch_path"], self.repo_root)
        post_patch_path = resolve_repo_path(row["post_patch_path"], self.repo_root)

        sample = {
            "index": index,
            "building_id": str(row["building_id"]),
            "positive_id": str(row["positive_id"]),
            "building_uid": str(row["building_uid"]),
            "tile_id": str(row["tile_id"]),
            "disaster": str(row["disaster"]),
            "disaster_type": str(row["disaster_type"]),
            "damage_label": str(row["damage_label"]),
            "damage_id": int(row["damage_id"]),
            "split": str(row["split"]),
            "pre_patch_path": pre_patch_path,
            "post_patch_path": post_patch_path,
        }

        if self.view == "pair":
            sample["pre_image"] = self._load_image(pre_patch_path)
            sample["post_image"] = self._load_image(post_patch_path)
            return sample

        image_path = pre_patch_path if self.view == "pre" else post_patch_path
        sample["image_path"] = image_path
        sample["image"] = self._load_image(image_path)
        return sample

    def _load_image(self, image_path: Path) -> Any:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                return self.transform(image)
            return image.copy()


def collate_building_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pre_image": torch.stack([item["pre_image"] for item in batch]),
        "post_image": torch.stack([item["post_image"] for item in batch]),
        "positive_id": [item["positive_id"] for item in batch],
        "building_id": [item["building_id"] for item in batch],
        "building_uid": [item["building_uid"] for item in batch],
        "tile_id": [item["tile_id"] for item in batch],
        "disaster": [item["disaster"] for item in batch],
        "disaster_type": [item["disaster_type"] for item in batch],
        "damage_label": [item["damage_label"] for item in batch],
        "damage_id": torch.tensor([item["damage_id"] for item in batch], dtype=torch.long),
        "split": [item["split"] for item in batch],
        "pre_patch_path": [str(item["pre_patch_path"]) for item in batch],
        "post_patch_path": [str(item["post_patch_path"]) for item in batch],
    }


def collate_single_view(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "positive_id": [item["positive_id"] for item in batch],
        "building_id": [item["building_id"] for item in batch],
        "building_uid": [item["building_uid"] for item in batch],
        "tile_id": [item["tile_id"] for item in batch],
        "disaster": [item["disaster"] for item in batch],
        "disaster_type": [item["disaster_type"] for item in batch],
        "damage_label": [item["damage_label"] for item in batch],
        "damage_id": torch.tensor([item["damage_id"] for item in batch], dtype=torch.long),
        "split": [item["split"] for item in batch],
        "image_path": [str(item["image_path"]) for item in batch],
    }


__all__ = [
    "XBDBuildingPatchDataset",
    "collate_building_pairs",
    "collate_single_view",
    "load_building_patch_table",
    "resolve_repo_path",
]
