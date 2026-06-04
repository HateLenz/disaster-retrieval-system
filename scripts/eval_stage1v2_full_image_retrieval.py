from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.cuda.amp import autocast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.transforms import build_clip_transform
from src.datasets.xbd_building_dataset import load_building_patch_table, resolve_repo_path
from src.models.clip_visual_encoder import DEFAULT_CLIP_MODEL_NAME, load_model_checkpoint
from src.retrieval.faiss_index import build_index, search_index
from src.retrieval.metrics import compute_grouped_metrics, compute_macro_metrics, compute_metrics
from src.utils.device import get_device


DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches_smoke_stage1.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "clip_visual_baseline_smoke_stage1_v2.pt"
DEFAULT_RESULTS = REPO_ROOT / "outputs" / "predictions" / "full_image_retrieval_results_smoke_stage1_v2.csv"
DEFAULT_METRICS = REPO_ROOT / "outputs" / "predictions" / "full_image_retrieval_metrics_smoke_stage1_v2.json"
DEFAULT_PATCH_META = REPO_ROOT / "outputs" / "predictions" / "full_image_patch_metadata_smoke_stage1_v2.csv"


@dataclass(frozen=True)
class WindowPatch:
    x1: int
    y1: int
    x2: int
    y2: int
    quality: float
    gray_std: float
    edge_density: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full-image post-to-pre retrieval with the stage1v2 visual model. "
            "Each full image is represented by sliding-window patches, then patch matches "
            "are aggregated into full-image/tile rankings."
        ),
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", type=str, default="hold")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--window-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument(
        "--max-patches-per-image",
        type=int,
        default=64,
        help="Keep the highest-texture windows per image. Use 0 to keep all windows.",
    )
    parser.add_argument(
        "--min-gray-std",
        type=float,
        default=0.0,
        help="Drop nearly blank windows below this grayscale standard deviation.",
    )
    parser.add_argument("--edge-threshold", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--include-unclassified", action="store_true")
    parser.add_argument("--model-name", type=str, default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--prior-field",
        type=str,
        default="none",
        help="Optional gallery restriction field, e.g. disaster or disaster_type. Use none for full-gallery search.",
    )
    parser.add_argument(
        "--patch-top-k",
        type=int,
        default=100,
        help="Number of gallery patches retrieved for each query patch before image-level aggregation.",
    )
    parser.add_argument("--image-top-k", type=int, default=10)
    parser.add_argument("--top-m", type=int, default=20)
    parser.add_argument(
        "--aggregation",
        type=str,
        default="top_m",
        choices=("top_m", "max", "mean", "sum", "vote"),
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--patch-metadata-csv",
        type=Path,
        default=DEFAULT_PATCH_META,
        help="Optional metadata for generated full-image sliding-window patches.",
    )
    return parser.parse_args()


def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return autocast(enabled=enabled)
    return nullcontext()


def metric_ks(max_k: int) -> tuple[int, ...]:
    values = [1]
    if max_k >= 5:
        values.append(5)
    if max_k not in values:
        values.append(max_k)
    return tuple(values)


def load_model(args: argparse.Namespace, device: torch.device):
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model, checkpoint = load_model_checkpoint(
        args.checkpoint,
        map_location=device,
        model_name=args.model_name,
        local_files_only=args.local_files_only,
    )
    return model.to(device), checkpoint


def load_tile_table(args: argparse.Namespace) -> pd.DataFrame:
    table = load_building_patch_table(
        csv_path=args.csv,
        split=args.split,
        include_unclassified=args.include_unclassified,
    )
    required_columns = [
        "tile_id",
        "pre_image_path",
        "post_image_path",
        "disaster",
        "disaster_type",
        "split",
    ]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required full-image columns: {missing_columns}")

    tile_table = (
        table[required_columns]
        .drop_duplicates(subset=["tile_id"], keep="first")
        .sort_values(["split", "disaster_type", "tile_id"], kind="stable")
        .reset_index(drop=True)
    )
    if tile_table.empty:
        raise ValueError(f"No full-image tiles found for split={args.split!r}")
    return tile_table


def axis_positions(length: int, window_size: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    if window_size <= 0:
        raise ValueError("window-size must be positive")
    if length <= window_size:
        return [0]

    last = length - window_size
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def patch_quality(crop: Image.Image, edge_threshold: float) -> tuple[float, float, float]:
    array = np.asarray(crop.convert("RGB"), dtype=np.float32)
    gray = 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
    gray_std = float(gray.std())
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        edge_density = 0.0
    else:
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        edge_density = float((np.mean(dx > edge_threshold) + np.mean(dy > edge_threshold)) / 2.0)
    quality = gray_std + 50.0 * edge_density
    return quality, gray_std, edge_density


def select_windows(image: Image.Image, args: argparse.Namespace) -> list[WindowPatch]:
    width, height = image.size
    x_positions = axis_positions(width, args.window_size, args.stride)
    y_positions = axis_positions(height, args.window_size, args.stride)

    windows: list[WindowPatch] = []
    best_window: WindowPatch | None = None
    for y1 in y_positions:
        for x1 in x_positions:
            x2 = min(x1 + args.window_size, width)
            y2 = min(y1 + args.window_size, height)
            crop = image.crop((x1, y1, x2, y2))
            quality, gray_std, edge_density = patch_quality(crop, edge_threshold=args.edge_threshold)
            window = WindowPatch(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                quality=quality,
                gray_std=gray_std,
                edge_density=edge_density,
            )
            if best_window is None or window.quality > best_window.quality:
                best_window = window
            if gray_std >= args.min_gray_std:
                windows.append(window)

    if not windows and best_window is not None:
        windows.append(best_window)

    if args.max_patches_per_image > 0 and len(windows) > args.max_patches_per_image:
        windows = sorted(windows, key=lambda item: item.quality, reverse=True)[: args.max_patches_per_image]
    return sorted(windows, key=lambda item: (item.y1, item.x1))


def encode_batch(
    model: torch.nn.Module,
    tensors: list[torch.Tensor],
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    if not tensors:
        return np.empty((0, 0), dtype=np.float32)
    batch = torch.stack(tensors).to(device, non_blocking=True)
    with torch.no_grad():
        with amp_context(device, amp_enabled):
            embeddings = model.encode_image(batch)
    return embeddings.cpu().numpy().astype(np.float32)


def encode_full_image_patches(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tile_table: pd.DataFrame,
    view: str,
    device: torch.device,
) -> tuple[np.ndarray, pd.DataFrame]:
    if view not in {"pre", "post"}:
        raise ValueError("view must be one of {'pre', 'post'}")

    path_column = "pre_image_path" if view == "pre" else "post_image_path"
    transform = build_clip_transform(image_size=args.image_size, train=False)
    amp_enabled = bool(args.amp and device.type == "cuda")
    feature_blocks: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []

    pending_tensors: list[torch.Tensor] = []
    pending_rows: list[dict[str, Any]] = []

    def flush_pending() -> None:
        if not pending_tensors:
            return
        feature_blocks.append(
            encode_batch(
                model=model,
                tensors=pending_tensors,
                device=device,
                amp_enabled=amp_enabled,
            )
        )
        metadata_rows.extend(pending_rows)
        pending_tensors.clear()
        pending_rows.clear()

    model.eval()
    for tile_index, row in tile_table.iterrows():
        image_path = resolve_repo_path(row[path_column], REPO_ROOT)
        if not image_path.exists():
            raise FileNotFoundError(f"Full image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            windows = select_windows(image, args)
            for patch_index, window in enumerate(windows):
                crop = image.crop((window.x1, window.y1, window.x2, window.y2))
                pending_tensors.append(transform(crop))
                pending_rows.append(
                    {
                        "patch_id": f"{row['tile_id']}::{view}::{patch_index:04d}",
                        "tile_id": str(row["tile_id"]),
                        "tile_index": int(tile_index),
                        "view": view,
                        "split": str(row["split"]),
                        "disaster": str(row["disaster"]),
                        "disaster_type": str(row["disaster_type"]),
                        "image_path": str(image_path),
                        "x1": window.x1,
                        "y1": window.y1,
                        "x2": window.x2,
                        "y2": window.y2,
                        "quality": window.quality,
                        "gray_std": window.gray_std,
                        "edge_density": window.edge_density,
                    }
                )
                if len(pending_tensors) >= args.batch_size:
                    flush_pending()

    flush_pending()
    if not feature_blocks:
        raise ValueError(f"No {view} full-image patches were generated")
    features = np.concatenate(feature_blocks, axis=0)
    return features, pd.DataFrame(metadata_rows)


def score_candidate(scores: list[float], aggregation: str, top_m: int) -> tuple[float, float, int]:
    if not scores:
        return 0.0, 0.0, 0

    ordered = sorted((float(score) for score in scores), reverse=True)
    best_score = ordered[0]
    hit_count = len(ordered)
    if aggregation == "max":
        return best_score, best_score, hit_count
    if aggregation == "mean":
        return float(np.mean(ordered)), best_score, hit_count
    if aggregation == "sum":
        return float(np.sum(ordered)), best_score, hit_count
    if aggregation == "vote":
        return float(hit_count), best_score, hit_count

    selected = ordered[: max(1, min(top_m, len(ordered)))]
    return float(np.mean(selected)), best_score, hit_count


def normalized_prior_field(raw_prior_field: str) -> str | None:
    prior_field = str(raw_prior_field).strip()
    if not prior_field or prior_field.lower() in {"none", "null", "false", "off"}:
        return None
    return prior_field


def search_patch_candidates(
    args: argparse.Namespace,
    query_features: np.ndarray,
    query_meta: pd.DataFrame,
    gallery_features: np.ndarray,
    gallery_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prior_field = normalized_prior_field(args.prior_field)
    top_k = args.patch_top_k
    if top_k <= 0:
        raise ValueError("patch-top-k must be positive")

    all_scores = np.full((query_features.shape[0], top_k), -np.inf, dtype=np.float32)
    all_indices = np.full((query_features.shape[0], top_k), -1, dtype=np.int64)
    candidate_counts = np.full(query_features.shape[0], gallery_features.shape[0], dtype=np.int64)

    if prior_field is None:
        index = build_index(gallery_features)
        scores, indices = search_index(index, query_features, top_k=min(top_k, gallery_features.shape[0]))
        all_scores[:, : scores.shape[1]] = scores
        all_indices[:, : indices.shape[1]] = indices
        return all_scores, all_indices, {
            "field": None,
            "candidate_patch_count": summarize_counts(candidate_counts.tolist()),
        }

    if prior_field not in query_meta.columns:
        raise ValueError(f"Prior field is not available in query metadata: {prior_field}")
    if prior_field not in gallery_meta.columns:
        raise ValueError(f"Prior field is not available in gallery metadata: {prior_field}")

    gallery_groups: dict[str, np.ndarray] = {}
    for value, group in gallery_meta.groupby(prior_field, sort=True):
        gallery_groups[str(value)] = group.index.to_numpy(dtype=np.int64)

    for value, query_group in query_meta.groupby(prior_field, sort=True):
        query_indices = query_group.index.to_numpy(dtype=np.int64)
        gallery_indices = gallery_groups.get(str(value))
        if gallery_indices is None or gallery_indices.size == 0:
            candidate_counts[query_indices] = 0
            continue

        candidate_counts[query_indices] = int(gallery_indices.size)
        index = build_index(gallery_features[gallery_indices])
        scores, local_indices = search_index(
            index,
            query_features[query_indices],
            top_k=min(top_k, gallery_indices.size),
        )
        global_indices = gallery_indices[local_indices]
        all_scores[np.ix_(query_indices, np.arange(scores.shape[1]))] = scores
        all_indices[np.ix_(query_indices, np.arange(global_indices.shape[1]))] = global_indices

    return all_scores, all_indices, {
        "field": prior_field,
        "candidate_patch_count": summarize_counts(candidate_counts.tolist()),
    }


def aggregate_image_rankings(
    args: argparse.Namespace,
    query_meta: pd.DataFrame,
    gallery_meta: pd.DataFrame,
    patch_scores: np.ndarray,
    patch_indices: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str], list[str]]:
    query_groups = query_meta.groupby("tile_id", sort=True).indices
    gallery_patch_counts = gallery_meta["tile_id"].astype(str).value_counts().to_dict()
    gallery_image_paths = gallery_meta.groupby("tile_id", sort=True)["image_path"].first().to_dict()
    gallery_disasters = gallery_meta.groupby("tile_id", sort=True)["disaster"].first().to_dict()
    gallery_disaster_types = gallery_meta.groupby("tile_id", sort=True)["disaster_type"].first().to_dict()

    result_rows: list[dict[str, Any]] = []
    retrieved_rows: list[list[str]] = []
    query_tile_ids: list[str] = []
    query_disasters: list[str] = []
    query_disaster_types: list[str] = []

    for query_tile_id in sorted(query_groups):
        query_indices = list(query_groups[query_tile_id])
        query_rows = query_meta.iloc[query_indices]
        candidate_scores: dict[str, list[float]] = defaultdict(list)

        for query_patch_index in query_indices:
            for score, gallery_patch_index in zip(
                patch_scores[query_patch_index],
                patch_indices[query_patch_index],
                strict=True,
            ):
                gallery_patch_index = int(gallery_patch_index)
                if gallery_patch_index < 0:
                    continue
                gallery_tile_id = str(gallery_meta.iloc[gallery_patch_index]["tile_id"])
                candidate_scores[gallery_tile_id].append(float(score))

        scored_candidates: list[tuple[str, float, float, int]] = []
        for gallery_tile_id, scores in candidate_scores.items():
            aggregate_score, best_patch_score, hit_count = score_candidate(
                scores=scores,
                aggregation=args.aggregation,
                top_m=args.top_m,
            )
            scored_candidates.append((gallery_tile_id, aggregate_score, best_patch_score, hit_count))

        scored_candidates.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        ranked = scored_candidates[: args.image_top_k]
        retrieved_tile_ids = [item[0] for item in ranked]
        if len(retrieved_tile_ids) < args.image_top_k:
            retrieved_tile_ids.extend([""] * (args.image_top_k - len(retrieved_tile_ids)))

        query_tile_ids.append(str(query_tile_id))
        query_disaster = str(query_rows["disaster"].iloc[0])
        query_disaster_type = str(query_rows["disaster_type"].iloc[0])
        query_disasters.append(query_disaster)
        query_disaster_types.append(query_disaster_type)
        retrieved_rows.append(retrieved_tile_ids)

        for rank_index, (gallery_tile_id, aggregate_score, best_patch_score, hit_count) in enumerate(ranked, start=1):
            result_rows.append(
                {
                    "query_tile_id": str(query_tile_id),
                    "query_disaster": query_disaster,
                    "query_disaster_type": query_disaster_type,
                    "query_image_path": str(query_rows["image_path"].iloc[0]),
                    "query_patch_count": len(query_indices),
                    "rank": rank_index,
                    "score": aggregate_score,
                    "best_patch_score": best_patch_score,
                    "patch_hit_count": hit_count,
                    "retrieved_tile_id": gallery_tile_id,
                    "retrieved_disaster": str(gallery_disasters.get(gallery_tile_id, "")),
                    "retrieved_disaster_type": str(gallery_disaster_types.get(gallery_tile_id, "")),
                    "retrieved_image_path": str(gallery_image_paths.get(gallery_tile_id, "")),
                    "retrieved_patch_count": int(gallery_patch_counts.get(gallery_tile_id, 0)),
                    "is_match": str(query_tile_id) == gallery_tile_id,
                }
            )

    return (
        pd.DataFrame(result_rows),
        np.asarray(retrieved_rows, dtype=object),
        query_tile_ids,
        query_disasters,
        query_disaster_types,
    )


def summarize_counts(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    array = np.asarray(values, dtype=np.float32)
    return {
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def patch_count_summary(metadata: pd.DataFrame) -> dict[str, float | int]:
    counts = metadata["tile_id"].astype(str).value_counts().to_numpy(dtype=np.int64)
    return summarize_counts(counts.tolist())


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    tile_table = load_tile_table(args)
    model, checkpoint = load_model(args, device)

    gallery_features, gallery_meta = encode_full_image_patches(
        args=args,
        model=model,
        tile_table=tile_table,
        view="pre",
        device=device,
    )
    query_features, query_meta = encode_full_image_patches(
        args=args,
        model=model,
        tile_table=tile_table,
        view="post",
        device=device,
    )
    if gallery_features.shape[1] != query_features.shape[1]:
        raise ValueError("Query and gallery feature dimensions do not match")

    patch_scores, patch_indices, prior_report = search_patch_candidates(
        args=args,
        query_features=query_features,
        query_meta=query_meta,
        gallery_features=gallery_features,
        gallery_meta=gallery_meta,
    )

    result_frame, retrieved_tiles, query_tile_ids, query_disasters, query_disaster_types = aggregate_image_rankings(
        args=args,
        query_meta=query_meta,
        gallery_meta=gallery_meta,
        patch_scores=patch_scores,
        patch_indices=patch_indices,
    )

    ks = metric_ks(args.image_top_k)
    overall_metrics = compute_metrics(query_tile_ids, retrieved_tiles, ks=ks)
    by_disaster = compute_grouped_metrics(query_tile_ids, retrieved_tiles, groups=query_disasters, ks=ks)
    by_disaster_type = compute_grouped_metrics(query_tile_ids, retrieved_tiles, groups=query_disaster_types, ks=ks)

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(args.results_csv, index=False)
    args.patch_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([gallery_meta, query_meta], axis=0, ignore_index=True).to_csv(args.patch_metadata_csv, index=False)

    metrics = {
        "overall": overall_metrics,
        "macro_by_disaster": compute_macro_metrics(by_disaster),
        "macro_by_disaster_type": compute_macro_metrics(by_disaster_type),
        "by_disaster": by_disaster,
        "by_disaster_type": by_disaster_type,
        "split": args.split,
        "query_image_count": len(query_tile_ids),
        "gallery_image_count": int(gallery_meta["tile_id"].astype(str).nunique()),
        "query_patch_count": int(query_features.shape[0]),
        "gallery_patch_count": int(gallery_features.shape[0]),
        "query_patches_per_image": patch_count_summary(query_meta),
        "gallery_patches_per_image": patch_count_summary(gallery_meta),
        "window": {
            "image_size": args.image_size,
            "window_size": args.window_size,
            "stride": args.stride,
            "max_patches_per_image": args.max_patches_per_image,
            "min_gray_std": args.min_gray_std,
            "edge_threshold": args.edge_threshold,
        },
        "retrieval": {
            "patch_top_k": args.patch_top_k,
            "image_top_k": args.image_top_k,
            "aggregation": args.aggregation,
            "top_m": args.top_m,
            "prior": prior_report,
        },
        "checkpoint": str(args.checkpoint),
        "model_config": checkpoint.get("model_config", {}),
        "outputs": {
            "results_csv": str(args.results_csv),
            "metrics_json": str(args.metrics_json),
            "patch_metadata_csv": str(args.patch_metadata_csv),
        },
    }

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_json.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "overall": overall_metrics,
                "macro_by_disaster_type": metrics["macro_by_disaster_type"],
                "query_image_count": metrics["query_image_count"],
                "gallery_image_count": metrics["gallery_image_count"],
                "query_patch_count": metrics["query_patch_count"],
                "gallery_patch_count": metrics["gallery_patch_count"],
                "prior": prior_report,
            },
            ensure_ascii=False,
        )
    )
    print(f"Full-image retrieval results saved to: {args.results_csv}")
    print(f"Metrics saved to: {args.metrics_json}")
    print(f"Patch metadata saved to: {args.patch_metadata_csv}")


if __name__ == "__main__":
    main()
