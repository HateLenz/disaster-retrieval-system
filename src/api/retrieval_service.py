from __future__ import annotations

import io
import os
import sqlite3
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.datasets.transforms import build_clip_transform
from src.models.clip_visual_encoder import DEFAULT_CLIP_MODEL_NAME, load_model_checkpoint
from src.retrieval.faiss_index import build_index, search_index
from src.utils.device import get_device


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "clip_visual_baseline_smoke_stage1_v2.pt"
DEFAULT_FULL_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches.csv"
DEFAULT_SMOKE_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches_smoke_stage1.csv"
DEFAULT_FULL_FEATURES = REPO_ROOT / "indexes" / "api_hold_stage1_v2" / "pre_features.npy"
DEFAULT_FULL_METADATA = REPO_ROOT / "indexes" / "api_hold_stage1_v2" / "pre_metadata.csv"
DEFAULT_FULL_DB = REPO_ROOT / "db" / "retrieval_hold_stage1_v2.db"
DEFAULT_SMOKE_FEATURES = REPO_ROOT / "indexes" / "pre_clip_features_smoke_stage1_v2.npy"
DEFAULT_SMOKE_METADATA = REPO_ROOT / "indexes" / "pre_clip_features_smoke_stage1_v2.csv"
DEFAULT_SMOKE_DB = REPO_ROOT / "db" / "retrieval_hold_smoke_stage1_v2.db"
DEFAULT_SAMPLE_RESULTS_BY_CHECKPOINT: dict[str, Path] = {
    "clip_visual_baseline_smoke_stage1.pt": REPO_ROOT
    / "outputs"
    / "predictions"
    / "clip_retrieval_results_smoke_stage1.csv",
    "clip_visual_baseline_smoke_stage1_v2.pt": REPO_ROOT
    / "outputs"
    / "predictions"
    / "clip_retrieval_results_smoke_stage1_v2.csv",
}
ALLOWED_APP_CHECKPOINTS: dict[str, str] = {
    "clip_visual_baseline_smoke_stage1.pt": "检索模型1.0",
    "clip_visual_baseline_smoke_stage1_v2.pt": "检索模型1.1",
}
SAMPLE_DAMAGE_LABELS = {"minor-damage", "no-damage"}
SAMPLE_SPLIT = "hold"
EXCLUDED_SAMPLE_DISASTERS = {"socialdisaster"}

PATH_MARKERS = (
    "data/processed/",
    "data/raw/",
    "outputs/",
    "indexes/",
)

SOURCE_METADATA_COLUMNS = [
    "positive_id",
    "building_id",
    "building_uid",
    "tile_id",
    "pre_image_path",
    "post_image_path",
    "pre_patch_path",
    "post_patch_path",
    "disaster",
    "disaster_type",
    "damage_label",
    "damage_id",
    "split",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "patch_size",
]


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class ArtifactPaths:
    features_path: Path
    metadata_path: Path
    source_csv: Path
    sqlite_path: Path
    scope: str


@dataclass(frozen=True)
class RetrievalSettings:
    checkpoint_path: Path
    model_name: str
    local_files_only: bool
    device_preference: str
    image_size: int
    batch_size: int
    amp: bool
    query_window_size: int
    query_stride: int
    max_query_patches: int
    patch_top_k: int
    aggregation_top_m: int
    artifact_paths: ArtifactPaths

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        explicit_features = os.getenv("DFR_FEATURES_PATH")
        explicit_metadata = os.getenv("DFR_METADATA_PATH")
        explicit_csv = os.getenv("DFR_SOURCE_CSV")

        if explicit_features or explicit_metadata or explicit_csv:
            artifact_paths = ArtifactPaths(
                features_path=_env_path("DFR_FEATURES_PATH", DEFAULT_FULL_FEATURES),
                metadata_path=_env_path("DFR_METADATA_PATH", DEFAULT_FULL_METADATA),
                source_csv=_env_path("DFR_SOURCE_CSV", DEFAULT_FULL_CSV),
                sqlite_path=_env_path("DFR_SQLITE_PATH", DEFAULT_FULL_DB),
                scope="custom",
            )
        elif DEFAULT_FULL_FEATURES.exists() and DEFAULT_FULL_METADATA.exists():
            artifact_paths = ArtifactPaths(
                features_path=DEFAULT_FULL_FEATURES,
                metadata_path=DEFAULT_FULL_METADATA,
                source_csv=DEFAULT_FULL_CSV,
                sqlite_path=DEFAULT_FULL_DB,
                scope="hold_full",
            )
        elif DEFAULT_SMOKE_FEATURES.exists() and DEFAULT_SMOKE_METADATA.exists():
            artifact_paths = ArtifactPaths(
                features_path=DEFAULT_SMOKE_FEATURES,
                metadata_path=DEFAULT_SMOKE_METADATA,
                source_csv=DEFAULT_SMOKE_CSV,
                sqlite_path=DEFAULT_SMOKE_DB,
                scope="hold_smoke_stage1_v2",
            )
        else:
            artifact_paths = ArtifactPaths(
                features_path=DEFAULT_FULL_FEATURES,
                metadata_path=DEFAULT_FULL_METADATA,
                source_csv=DEFAULT_FULL_CSV,
                sqlite_path=DEFAULT_FULL_DB,
                scope="hold_full_missing",
            )

        return cls(
            checkpoint_path=_env_path("DFR_CHECKPOINT_PATH", DEFAULT_CHECKPOINT),
            model_name=os.getenv("DFR_MODEL_NAME", DEFAULT_CLIP_MODEL_NAME),
            local_files_only=_env_bool("DFR_LOCAL_FILES_ONLY", True),
            device_preference=os.getenv("DFR_DEVICE", "cuda"),
            image_size=_env_int("DFR_IMAGE_SIZE", 224),
            batch_size=_env_int("DFR_BATCH_SIZE", 64),
            amp=_env_bool("DFR_AMP", True),
            query_window_size=_env_int("DFR_QUERY_WINDOW_SIZE", 224),
            query_stride=_env_int("DFR_QUERY_STRIDE", 112),
            max_query_patches=_env_int("DFR_MAX_QUERY_PATCHES", 64),
            patch_top_k=_env_int("DFR_PATCH_TOP_K", 100),
            aggregation_top_m=_env_int("DFR_AGGREGATION_TOP_M", 20),
            artifact_paths=artifact_paths,
        )


def resolve_data_path(path_value: str | Path | None, repo_root: Path = REPO_ROOT) -> Path | None:
    if path_value is None:
        return None
    if isinstance(path_value, float) and pd.isna(path_value):
        return None

    raw_value = str(path_value).strip()
    if not raw_value or raw_value.lower() == "nan":
        return None

    normalized = raw_value.replace("\\", "/")
    for marker in PATH_MARKERS:
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            return (repo_root / normalized[marker_index:]).resolve()

    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def public_asset_url(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    return f"/api/assets?path={quote(str(path_value), safe='')}"


def _is_non_empty(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.len() > 0) & (series.astype(str).str.lower() != "nan")


def _normalized_label(value: Any) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def _is_excluded_sample_disaster(value: Any) -> bool:
    return _normalized_label(value) in EXCLUDED_SAMPLE_DISASTERS


def _is_true_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _is_hold_post_patch_path(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.replace("\\", "/", regex=False).str.lower()
    return normalized.str.contains("/hold/", regex=False) & normalized.str.contains("_post", regex=False)


def load_source_metadata(source_csv: Path) -> pd.DataFrame:
    if not source_csv.exists():
        return pd.DataFrame()
    header = pd.read_csv(source_csv, nrows=0)
    usecols = [column for column in SOURCE_METADATA_COLUMNS if column in header.columns]
    if not usecols:
        return pd.DataFrame()
    return pd.read_csv(source_csv, usecols=usecols)


def load_verified_sample_metadata(checkpoint_path: Path, source: pd.DataFrame) -> pd.DataFrame:
    results_path = DEFAULT_SAMPLE_RESULTS_BY_CHECKPOINT.get(checkpoint_path.name)
    if results_path is None or not results_path.exists():
        return pd.DataFrame()

    columns = [
        "query_positive_id",
        "query_building_id",
        "query_tile_id",
        "query_disaster",
        "query_disaster_type",
        "query_damage_label",
        "query_patch_path",
        "rank",
        "score",
        "is_match",
    ]
    results = pd.read_csv(results_path, usecols=columns)
    hits = results[
        (pd.to_numeric(results["rank"], errors="coerce") <= 10)
        & _is_true_series(results["is_match"])
        & results["query_damage_label"].astype(str).str.lower().isin(SAMPLE_DAMAGE_LABELS)
        & ~results["query_disaster"].map(_is_excluded_sample_disaster)
        & ~results["query_disaster_type"].map(_is_excluded_sample_disaster)
    ].copy()
    if hits.empty:
        return pd.DataFrame()

    hits = hits.sort_values(["rank", "score", "query_disaster", "query_positive_id"], ascending=[True, False, True, True])
    hits = hits.drop_duplicates(subset=["query_positive_id"], keep="first")
    metadata = hits.rename(
        columns={
            "query_positive_id": "positive_id",
            "query_building_id": "building_id",
            "query_tile_id": "tile_id",
            "query_disaster": "disaster",
            "query_disaster_type": "disaster_type",
            "query_damage_label": "damage_label",
            "rank": "correct_rank",
            "score": "correct_score",
        }
    )
    metadata = enrich_metadata(metadata, DEFAULT_FULL_CSV, source)
    if "query_patch_path" in metadata.columns:
        metadata["post_patch_path"] = metadata["post_patch_path"].where(
            _is_non_empty(metadata["post_patch_path"]),
            metadata["query_patch_path"],
        )
    metadata = metadata[
        metadata["split"].astype(str).str.lower().eq(SAMPLE_SPLIT)
        & _is_non_empty(metadata["post_patch_path"])
        & _is_hold_post_patch_path(metadata["post_patch_path"])
    ].copy()
    return metadata.reset_index(drop=True)


def enrich_metadata(metadata: pd.DataFrame, source_csv: Path, source: pd.DataFrame | None = None) -> pd.DataFrame:
    metadata = metadata.copy()
    source = source if source is not None else load_source_metadata(source_csv)
    if not source.empty and "positive_id" in metadata.columns and "positive_id" in source.columns:
        metadata = metadata.merge(source, on="positive_id", how="left", suffixes=("", "__source"))
        for column in SOURCE_METADATA_COLUMNS:
            source_column = f"{column}__source"
            if source_column not in metadata.columns:
                continue
            if column not in metadata.columns:
                metadata[column] = metadata[source_column]
            else:
                metadata[column] = metadata[column].where(_is_non_empty(metadata[column]), metadata[source_column])
            metadata = metadata.drop(columns=[source_column])

    if "pre_patch_path" not in metadata.columns and "image_path" in metadata.columns:
        metadata["pre_patch_path"] = metadata["image_path"]
    if "image_path" not in metadata.columns and "pre_patch_path" in metadata.columns:
        metadata["image_path"] = metadata["pre_patch_path"]

    for column in SOURCE_METADATA_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = ""

    metadata["tile_id"] = metadata["tile_id"].astype(str)
    metadata["disaster"] = metadata["disaster"].astype(str)
    metadata["disaster_type"] = metadata["disaster_type"].astype(str)
    metadata["positive_id"] = metadata["positive_id"].astype(str)
    metadata["building_uid"] = metadata["building_uid"].astype(str)
    metadata["view"] = "pre"
    return metadata.reset_index(drop=True)


def write_metadata_sqlite(metadata: pd.DataFrame, sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        column
        for column in SOURCE_METADATA_COLUMNS + ["image_path", "view"]
        if column in metadata.columns
    ]
    with sqlite3.connect(sqlite_path) as connection:
        metadata[columns].to_sql("gallery_items", connection, if_exists="replace", index=False)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_gallery_disaster_type ON gallery_items(disaster_type)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_gallery_disaster ON gallery_items(disaster)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_gallery_tile ON gallery_items(tile_id)")
        connection.commit()


def axis_positions(length: int, window_size: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("query stride must be positive")
    if window_size <= 0:
        raise ValueError("query window size must be positive")
    if length <= window_size:
        return [0]

    last = length - window_size
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def patch_quality(crop: Image.Image) -> float:
    array = np.asarray(crop.convert("RGB"), dtype=np.float32)
    gray = 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
    gray_std = float(gray.std())
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return gray_std
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_density = float((np.mean(dx > 20.0) + np.mean(dy > 20.0)) / 2.0)
    return gray_std + 50.0 * edge_density


@dataclass(frozen=True)
class QueryPatch:
    image: Image.Image
    x1: int
    y1: int
    x2: int
    y2: int
    quality: float


class DisasterRetrievalService:
    def __init__(self, settings: RetrievalSettings | None = None) -> None:
        self.settings = settings or RetrievalSettings.from_env()
        self.device = get_device(self.settings.device_preference)
        self.amp_enabled = bool(self.settings.amp and self.device.type == "cuda")
        self.transform = build_clip_transform(image_size=self.settings.image_size, train=False)
        self.model: torch.nn.Module | None = None
        self.checkpoint: dict[str, Any] = {}
        self.features: np.ndarray | None = None
        self.metadata: pd.DataFrame | None = None
        self.sample_metadata: pd.DataFrame | None = None
        self._index_cache: dict[tuple[str, str, str], tuple[np.ndarray, Any]] = {}
        self.load_error: str | None = None
        self.loaded_at: float | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.features is not None and self.metadata is not None

    def load(self) -> None:
        artifacts = self.settings.artifact_paths
        missing = [
            str(path)
            for path in (self.settings.checkpoint_path, artifacts.features_path, artifacts.metadata_path)
            if not path.exists()
        ]
        if missing:
            self.load_error = (
                "Missing retrieval artifacts: "
                + ", ".join(missing)
                + ". Build them with scripts/build_api_index.py."
            )
            return

        started_at = time.perf_counter()
        try:
            self.features = np.load(artifacts.features_path).astype(np.float32)
            raw_metadata = pd.read_csv(artifacts.metadata_path)
            source_metadata = load_source_metadata(artifacts.source_csv)
            self.metadata = enrich_metadata(raw_metadata, artifacts.source_csv, source_metadata)
            self.sample_metadata = load_verified_sample_metadata(self.settings.checkpoint_path, source_metadata)
            if len(self.metadata) != self.features.shape[0]:
                raise ValueError(
                    f"Feature rows ({self.features.shape[0]}) do not match metadata rows ({len(self.metadata)})"
                )

            self.model, self.checkpoint = load_model_checkpoint(
                self.settings.checkpoint_path,
                map_location=self.device,
                model_name=self.settings.model_name,
                local_files_only=self.settings.local_files_only,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            write_metadata_sqlite(self.metadata, artifacts.sqlite_path)
            self._index_cache.clear()
            self.loaded_at = started_at
            self.load_error = None
        except Exception as error:  # pragma: no cover - returned through /api/health
            self.load_error = str(error)
            self.model = None
            self.features = None
            self.metadata = None
            self.sample_metadata = None

    def available_checkpoints(self) -> list[dict[str, Any]]:
        checkpoint_dir = REPO_ROOT / "checkpoints"
        rows = []
        if not checkpoint_dir.exists():
            return rows
        for filename, display_name in ALLOWED_APP_CHECKPOINTS.items():
            path = checkpoint_dir / filename
            if not path.exists():
                continue
            rows.append(
                {
                    "name": path.name,
                    "display_name": display_name,
                    "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "absolute_path": str(path),
                    "active": path.resolve() == self.settings.checkpoint_path.resolve(),
                    "size_bytes": path.stat().st_size,
                }
            )
        return rows

    def switch_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        raw_path = Path(str(checkpoint_path).strip())
        if raw_path.is_absolute():
            resolved = raw_path
        elif len(raw_path.parts) == 1:
            resolved = (REPO_ROOT / "checkpoints" / raw_path).resolve()
        else:
            resolved = (REPO_ROOT / raw_path).resolve()
        try:
            resolved.relative_to(REPO_ROOT / "checkpoints")
        except ValueError as error:
            raise PermissionError("Checkpoint must be inside the checkpoints directory.") from error
        if resolved.name not in ALLOWED_APP_CHECKPOINTS:
            allowed = ", ".join(ALLOWED_APP_CHECKPOINTS)
            raise PermissionError(f"Checkpoint must be one of: {allowed}.")
        if not resolved.exists():
            raise FileNotFoundError(str(resolved))
        self.settings = replace(self.settings, checkpoint_path=resolved)
        self.load()
        return self.status()

    def status(self) -> dict[str, Any]:
        metadata = self.metadata
        return {
            "ready": self.ready,
            "error": self.load_error,
            "scope": self.settings.artifact_paths.scope,
            "device": str(self.device),
            "amp": self.amp_enabled,
            "checkpoint": str(self.settings.checkpoint_path),
            "features_path": str(self.settings.artifact_paths.features_path),
            "metadata_path": str(self.settings.artifact_paths.metadata_path),
            "sqlite_path": str(self.settings.artifact_paths.sqlite_path),
            "gallery_patch_count": int(self.features.shape[0]) if self.features is not None else 0,
            "gallery_tile_count": int(metadata["tile_id"].nunique()) if metadata is not None else 0,
            "disaster_type_count": int(metadata["disaster_type"].nunique()) if metadata is not None else 0,
            "disaster_count": int(metadata["disaster"].nunique()) if metadata is not None else 0,
            "model_name": self.settings.model_name,
            "local_files_only": self.settings.local_files_only,
            "image_size": self.settings.image_size,
            "query_window_size": self.settings.query_window_size,
            "query_stride": self.settings.query_stride,
            "patch_top_k": self.settings.patch_top_k,
            "aggregation_top_m": self.settings.aggregation_top_m,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_version": torch.__version__,
            "index_cache_size": len(self._index_cache),
        }

    def diagnostics(self) -> dict[str, Any]:
        status = self.status()
        artifacts = self.settings.artifact_paths
        artifact_rows = []
        for label, path in {
            "checkpoint": self.settings.checkpoint_path,
            "features": artifacts.features_path,
            "metadata": artifacts.metadata_path,
            "source_csv": artifacts.source_csv,
            "sqlite": artifacts.sqlite_path,
        }.items():
            artifact_rows.append(
                {
                    "name": label,
                    "path": str(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                }
            )
        status["artifacts"] = artifact_rows
        if torch.cuda.is_available():
            status["cuda_device_count"] = torch.cuda.device_count()
            status["cuda_device_name"] = torch.cuda.get_device_name(0)
        else:
            status["cuda_device_count"] = 0
            status["cuda_device_name"] = None
        return status

    def options(self) -> dict[str, Any]:
        self._require_ready()
        assert self.metadata is not None
        metadata = self.metadata
        disaster_types = sorted(metadata["disaster_type"].dropna().astype(str).unique().tolist())
        disasters = sorted(metadata["disaster"].dropna().astype(str).unique().tolist())
        disasters_by_type = {
            disaster_type: sorted(
                metadata.loc[metadata["disaster_type"] == disaster_type, "disaster"].dropna().astype(str).unique().tolist()
            )
            for disaster_type in disaster_types
        }
        counts = (
            metadata.groupby(["disaster_type", "disaster"], sort=True)
            .agg(patches=("positive_id", "count"), tiles=("tile_id", "nunique"))
            .reset_index()
            .to_dict("records")
        )
        return {
            "disaster_types": disaster_types,
            "disasters": disasters,
            "disasters_by_type": disasters_by_type,
            "counts": counts,
        }

    def samples(self, disaster_type: str | None = None, disaster: str | None = None, limit: int = 12) -> list[dict[str, Any]]:
        self._require_ready()
        frame = self.sample_metadata if self.sample_metadata is not None else pd.DataFrame()
        if frame.empty:
            return []
        mask = pd.Series(True, index=frame.index)
        filter_mode = self._normalize_filter_mode("both")
        use_type = filter_mode in {"both", "type"}
        use_disaster = filter_mode in {"both", "disaster"}
        if use_type and disaster_type and disaster_type.lower() not in {"all", "any", "none"}:
            mask &= frame["disaster_type"].astype(str).to_numpy() == disaster_type
        if use_disaster and disaster and disaster.lower() not in {"all", "any", "none"}:
            mask &= frame["disaster"].astype(str).to_numpy() == disaster
        candidates = frame.loc[mask].copy()
        candidates = candidates[
            candidates["damage_label"].astype(str).str.lower().isin(SAMPLE_DAMAGE_LABELS)
            & ~candidates["disaster"].map(_is_excluded_sample_disaster)
            & ~candidates["disaster_type"].map(_is_excluded_sample_disaster)
        ]
        limit = max(1, min(limit, 100))
        candidates = candidates.sort_values(
            ["correct_rank", "correct_score", "disaster_type", "disaster", "tile_id", "positive_id"],
            ascending=[True, False, True, True, True, True],
        )
        diverse_rows = candidates.drop_duplicates(subset=["disaster"], keep="first")
        if len(diverse_rows) < limit:
            remaining = candidates.loc[~candidates.index.isin(diverse_rows.index)]
            rows = pd.concat([diverse_rows, remaining], ignore_index=False).head(limit)
        else:
            rows = diverse_rows.head(limit)
        samples: list[dict[str, Any]] = []
        for _, row in rows.iterrows():
            post_path = row.get("post_patch_path")
            pre_path = row.get("pre_patch_path") or row.get("pre_image_path") or row.get("image_path")
            if not post_path:
                continue
            samples.append(
                {
                    "positive_id": str(row.get("positive_id", "")),
                    "tile_id": str(row.get("tile_id", "")),
                    "disaster": str(row.get("disaster", "")),
                    "disaster_type": str(row.get("disaster_type", "")),
                    "damage_label": str(row.get("damage_label", "")),
                    "correct_rank": int(row.get("correct_rank", 0) or 0),
                    "correct_score": float(row.get("correct_score", 0.0) or 0.0),
                    "post_patch_path": str(post_path) if post_path is not None else "",
                    "post_patch_url": public_asset_url(post_path),
                    "pre_patch_path": str(pre_path) if pre_path is not None else "",
                    "pre_patch_url": public_asset_url(pre_path),
                }
            )
        return samples

    def resolve_asset_path(self, raw_path: str) -> Path:
        resolved = resolve_data_path(raw_path)
        if resolved is None:
            raise FileNotFoundError(raw_path)
        try:
            resolved.relative_to(REPO_ROOT)
        except ValueError as error:
            raise PermissionError(f"Asset path is outside the repository: {resolved}") from error
        if not resolved.exists():
            raise FileNotFoundError(str(resolved))
        return resolved

    def search_bytes(
        self,
        image_bytes: bytes,
        disaster_type: str | None,
        disaster: str | None,
        top_k: int = 5,
        filter_mode: str = "both",
        aggregation: str = "top_m",
    ) -> dict[str, Any]:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return self.search_image(
                image.convert("RGB"),
                disaster_type=disaster_type,
                disaster=disaster,
                top_k=top_k,
                filter_mode=filter_mode,
                aggregation=aggregation,
            )

    def search_path(
        self,
        image_path: str | Path,
        disaster_type: str | None,
        disaster: str | None,
        top_k: int = 5,
        filter_mode: str = "both",
        aggregation: str = "top_m",
    ) -> dict[str, Any]:
        resolved = resolve_data_path(image_path)
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(str(image_path))
        with Image.open(resolved) as image:
            result = self.search_image(
                image.convert("RGB"),
                disaster_type=disaster_type,
                disaster=disaster,
                top_k=top_k,
                filter_mode=filter_mode,
                aggregation=aggregation,
            )
        result["query"]["source_path"] = str(image_path)
        result["query"]["source_url"] = public_asset_url(image_path)
        return result

    def search_image(
        self,
        image: Image.Image,
        disaster_type: str | None,
        disaster: str | None,
        top_k: int = 5,
        filter_mode: str = "both",
        aggregation: str = "top_m",
    ) -> dict[str, Any]:
        self._require_ready()
        assert self.features is not None and self.metadata is not None

        started_at = time.perf_counter()
        top_k = max(1, min(int(top_k), 50))
        filter_mode = self._normalize_filter_mode(filter_mode)
        aggregation = self._normalize_aggregation(aggregation)
        query_patches = self._select_query_patches(image)
        query_features = self._encode_query_patches(query_patches)

        candidate_indices, index = self._candidate_index(disaster_type, disaster, filter_mode)
        if candidate_indices.size == 0:
            raise ValueError("No gallery items match the selected filters.")

        patch_top_k = min(max(self.settings.patch_top_k, top_k), int(candidate_indices.size))
        scores, local_indices = search_index(index, query_features, top_k=patch_top_k)
        tile_scores: dict[str, list[float]] = {}
        tile_best: dict[str, tuple[float, int]] = {}

        for patch_scores, patch_indices in zip(scores, local_indices, strict=True):
            for score, local_index in zip(patch_scores, patch_indices, strict=True):
                if int(local_index) < 0:
                    continue
                global_index = int(candidate_indices[int(local_index)])
                row = self.metadata.iloc[global_index]
                tile_id = str(row["tile_id"])
                score_value = float(score)
                tile_scores.setdefault(tile_id, []).append(score_value)
                current_best = tile_best.get(tile_id)
                if current_best is None or score_value > current_best[0]:
                    tile_best[tile_id] = (score_value, global_index)

        ranked_tiles: list[tuple[str, float, float, int, int]] = []
        for tile_id, scores_for_tile in tile_scores.items():
            ordered = sorted(scores_for_tile, reverse=True)
            aggregate_score = self._aggregate_scores(ordered, aggregation)
            best_score, best_index = tile_best[tile_id]
            ranked_tiles.append((tile_id, aggregate_score, best_score, len(ordered), best_index))

        ranked_tiles.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        results = [self._format_result(rank, item) for rank, item in enumerate(ranked_tiles[:top_k], start=1)]

        return {
            "query": {
                "width": image.width,
                "height": image.height,
                "patch_count": len(query_patches),
                "disaster_type": disaster_type,
                "disaster": disaster,
                "filter_mode": filter_mode,
                "aggregation": aggregation,
            },
            "gallery": {
                "candidate_patch_count": int(candidate_indices.size),
                "candidate_tile_count": int(self.metadata.iloc[candidate_indices]["tile_id"].nunique()),
                "scope": self.settings.artifact_paths.scope,
            },
            "results": results,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
        }

    def _require_ready(self) -> None:
        if not self.ready:
            raise RuntimeError(self.load_error or "Retrieval service is not ready.")

    def _normalize_filter_mode(self, filter_mode: str | None) -> str:
        value = (filter_mode or "both").strip().lower()
        if value not in {"both", "type", "disaster", "none"}:
            raise ValueError("filter_mode must be one of: both, type, disaster, none.")
        return value

    def _normalize_aggregation(self, aggregation: str | None) -> str:
        value = (aggregation or "top_m").strip().lower()
        if value not in {"top_m", "max", "mean", "sum", "vote"}:
            raise ValueError("aggregation must be one of: top_m, max, mean, sum, vote.")
        return value

    def _filter_mask(self, disaster_type: str | None, disaster: str | None, filter_mode: str = "both") -> np.ndarray:
        assert self.metadata is not None
        mask = np.ones(len(self.metadata), dtype=bool)
        filter_mode = self._normalize_filter_mode(filter_mode)
        use_type = filter_mode in {"both", "type"}
        use_disaster = filter_mode in {"both", "disaster"}
        if use_type and disaster_type and disaster_type.lower() not in {"all", "any", "none"}:
            mask &= self.metadata["disaster_type"].astype(str).to_numpy() == disaster_type
        if use_disaster and disaster and disaster.lower() not in {"all", "any", "none"}:
            mask &= self.metadata["disaster"].astype(str).to_numpy() == disaster
        return mask

    def _candidate_index(self, disaster_type: str | None, disaster: str | None, filter_mode: str) -> tuple[np.ndarray, Any]:
        assert self.features is not None
        key = (disaster_type or "", disaster or "", filter_mode)
        cached = self._index_cache.get(key)
        if cached is not None:
            return cached

        candidate_indices = np.flatnonzero(self._filter_mask(disaster_type, disaster, filter_mode)).astype(np.int64)
        if candidate_indices.size == 0:
            self._index_cache[key] = (candidate_indices, None)
            return candidate_indices, None

        index = build_index(self.features[candidate_indices])
        self._index_cache[key] = (candidate_indices, index)
        return candidate_indices, index

    def _aggregate_scores(self, ordered_scores: list[float], aggregation: str) -> float:
        if not ordered_scores:
            return 0.0
        if aggregation == "max":
            return float(ordered_scores[0])
        if aggregation == "mean":
            return float(np.mean(ordered_scores))
        if aggregation == "sum":
            return float(np.sum(ordered_scores))
        if aggregation == "vote":
            return float(len(ordered_scores))
        top_m = max(1, min(self.settings.aggregation_top_m, len(ordered_scores)))
        return float(np.mean(ordered_scores[:top_m]))

    def _select_query_patches(self, image: Image.Image) -> list[QueryPatch]:
        width, height = image.size
        if width <= self.settings.query_window_size + 32 and height <= self.settings.query_window_size + 32:
            return [
                QueryPatch(
                    image=image.copy(),
                    x1=0,
                    y1=0,
                    x2=width,
                    y2=height,
                    quality=patch_quality(image),
                )
            ]

        windows: list[QueryPatch] = []
        for y1 in axis_positions(height, self.settings.query_window_size, self.settings.query_stride):
            for x1 in axis_positions(width, self.settings.query_window_size, self.settings.query_stride):
                x2 = min(x1 + self.settings.query_window_size, width)
                y2 = min(y1 + self.settings.query_window_size, height)
                crop = image.crop((x1, y1, x2, y2))
                windows.append(QueryPatch(crop, x1=x1, y1=y1, x2=x2, y2=y2, quality=patch_quality(crop)))

        if self.settings.max_query_patches > 0 and len(windows) > self.settings.max_query_patches:
            windows = sorted(windows, key=lambda item: item.quality, reverse=True)[: self.settings.max_query_patches]
        return sorted(windows, key=lambda item: (item.y1, item.x1))

    def _encode_query_patches(self, patches: list[QueryPatch]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        feature_blocks: list[np.ndarray] = []
        pending: list[torch.Tensor] = []
        for patch in patches:
            pending.append(self.transform(patch.image))
            if len(pending) >= self.settings.batch_size:
                feature_blocks.append(self._encode_tensor_batch(pending))
                pending.clear()
        if pending:
            feature_blocks.append(self._encode_tensor_batch(pending))
        return np.concatenate(feature_blocks, axis=0)

    def _encode_tensor_batch(self, tensors: list[torch.Tensor]) -> np.ndarray:
        assert self.model is not None
        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        context = torch.amp.autocast("cuda", enabled=self.amp_enabled) if self.device.type == "cuda" else nullcontext()
        with torch.no_grad():
            with context:
                embeddings = self.model.encode_image(batch)
        return embeddings.detach().cpu().numpy().astype(np.float32)

    def _format_result(self, rank: int, ranked_item: tuple[str, float, float, int, int]) -> dict[str, Any]:
        assert self.metadata is not None
        tile_id, aggregate_score, best_score, patch_hit_count, best_index = ranked_item
        best_row = self.metadata.iloc[best_index]
        tile_rows = self.metadata.loc[self.metadata["tile_id"].astype(str) == str(tile_id)]
        first_row = tile_rows.iloc[0] if not tile_rows.empty else best_row

        pre_image_path = first_row.get("pre_image_path") or best_row.get("pre_image_path") or best_row.get("image_path")
        pre_patch_path = best_row.get("pre_patch_path") or best_row.get("image_path")
        post_patch_path = best_row.get("post_patch_path")

        return {
            "rank": rank,
            "score": round(float(aggregate_score), 6),
            "best_patch_score": round(float(best_score), 6),
            "patch_hit_count": int(patch_hit_count),
            "tile_id": str(tile_id),
            "positive_id": str(best_row.get("positive_id", "")),
            "building_uid": str(best_row.get("building_uid", "")),
            "disaster": str(best_row.get("disaster", "")),
            "disaster_type": str(best_row.get("disaster_type", "")),
            "damage_label": str(best_row.get("damage_label", "")),
            "pre_image_path": str(pre_image_path) if pre_image_path is not None else "",
            "pre_image_url": public_asset_url(pre_image_path),
            "pre_patch_path": str(pre_patch_path) if pre_patch_path is not None else "",
            "pre_patch_url": public_asset_url(pre_patch_path),
            "paired_post_patch_path": str(post_patch_path) if post_patch_path is not None else "",
            "paired_post_patch_url": public_asset_url(post_patch_path),
            "tile_patch_count": int(len(tile_rows)),
        }


__all__ = [
    "DisasterRetrievalService",
    "RetrievalSettings",
    "REPO_ROOT",
    "enrich_metadata",
    "public_asset_url",
    "resolve_data_path",
    "write_metadata_sqlite",
]
