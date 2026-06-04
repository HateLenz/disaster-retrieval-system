from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.transforms import build_clip_transform
from src.datasets.xbd_building_dataset import (
    XBDBuildingPatchDataset,
    collate_single_view,
    load_building_patch_table,
)
from src.models.clip_visual_encoder import DEFAULT_CLIP_MODEL_NAME, load_model_checkpoint
from src.retrieval.faiss_index import l2_normalize
from src.retrieval.metrics import compute_grouped_metrics, compute_macro_metrics, compute_metrics
from src.utils.device import get_device


DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches_smoke_stage1.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "clip_visual_baseline_smoke_stage1_v2.pt"
DEFAULT_FEATURES = REPO_ROOT / "indexes" / "pre_clip_features_smoke_stage1_v2.npy"
DEFAULT_PATCH_RESULTS = (
    REPO_ROOT / "outputs" / "predictions" / "clip_retrieval_results_smoke_stage1_v2_disaster_prior.csv"
)
DEFAULT_TILE_RESULTS = (
    REPO_ROOT / "outputs" / "predictions" / "clip_tile_retrieval_results_smoke_stage1_v2_disaster_prior.csv"
)
DEFAULT_METRICS = (
    REPO_ROOT / "outputs" / "predictions" / "clip_retrieval_metrics_smoke_stage1_v2_disaster_prior.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate stage1v2 retrieval with a known-disaster prior. "
            "For each post-disaster query, gallery candidates are restricted "
            "to the same value of --prior-field, defaulting to exact disaster."
        ),
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--features-meta", type=Path, default=None)
    parser.add_argument("--split", type=str, default="hold")
    parser.add_argument("--prior-field", type=str, default="disaster")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--include-unclassified", action="store_true")
    parser.add_argument("--model-name", type=str, default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--patch-top-k",
        type=int,
        default=100,
        help="Patch candidates kept per query before tile aggregation.",
    )
    parser.add_argument(
        "--patch-metric-top-k",
        type=int,
        default=10,
        help="Top-k used for patch-level recall metrics.",
    )
    parser.add_argument(
        "--patch-results-top-k",
        type=int,
        default=10,
        help="Number of patch candidates written per query.",
    )
    parser.add_argument("--tile-top-k", type=int, default=10)
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument(
        "--aggregation",
        type=str,
        default="top_m",
        choices=("top_m", "max", "mean", "sum", "vote"),
    )
    parser.add_argument("--patch-results-csv", type=Path, default=DEFAULT_PATCH_RESULTS)
    parser.add_argument("--tile-results-csv", type=Path, default=DEFAULT_TILE_RESULTS)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
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


def load_gallery_metadata(args: argparse.Namespace) -> pd.DataFrame:
    metadata_path = args.features_meta or args.features.with_suffix(".csv")
    if metadata_path.exists():
        return pd.read_csv(metadata_path)

    table = load_building_patch_table(
        csv_path=args.csv,
        split=args.split,
        include_unclassified=args.include_unclassified,
    )
    return table[
        [
            "positive_id",
            "building_id",
            "building_uid",
            "tile_id",
            "disaster",
            "disaster_type",
            "damage_label",
            "split",
            "pre_patch_path",
        ]
    ].rename(columns={"pre_patch_path": "image_path"})


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


def extract_post_embeddings(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    dataset = XBDBuildingPatchDataset(
        csv_path=args.csv,
        split=args.split,
        transform=build_clip_transform(image_size=args.image_size, train=False),
        include_unclassified=args.include_unclassified,
        view="post",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_single_view,
    )

    embeddings: list[np.ndarray] = []
    metadata_rows: list[dict[str, str]] = []
    amp_enabled = bool(args.amp and device.type == "cuda")

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            with amp_context(device, amp_enabled):
                batch_embeddings = model.encode_image(images)
            embeddings.append(batch_embeddings.cpu().numpy().astype(np.float32))

            for index in range(len(batch["positive_id"])):
                metadata_rows.append(
                    {
                        "positive_id": str(batch["positive_id"][index]),
                        "building_id": str(batch["building_id"][index]),
                        "building_uid": str(batch["building_uid"][index]),
                        "tile_id": str(batch["tile_id"][index]),
                        "disaster": str(batch["disaster"][index]),
                        "disaster_type": str(batch["disaster_type"][index]),
                        "damage_label": str(batch["damage_label"][index]),
                        "split": str(batch["split"][index]),
                        "query_patch_path": str(batch["image_path"][index]),
                    }
                )

    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)
    return matrix, metadata_rows


def validate_prior_field(
    prior_field: str,
    query_meta: list[dict[str, str]],
    gallery_meta: pd.DataFrame,
) -> None:
    if not query_meta:
        raise ValueError("No query samples were loaded")
    if prior_field not in query_meta[0]:
        raise ValueError(f"Prior field is not available in query metadata: {prior_field}")
    if prior_field not in gallery_meta.columns:
        raise ValueError(f"Prior field is not available in gallery metadata: {prior_field}")


def topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if scores.size == 0 or top_k <= 0:
        return np.empty((0,), dtype=np.int64)

    top_k = min(top_k, scores.size)
    if top_k == scores.size:
        selected = np.arange(scores.size)
    else:
        selected = np.argpartition(-scores, kth=top_k - 1)[:top_k]
    return selected[np.argsort(-scores[selected])]


def search_with_prior(
    query_features: np.ndarray,
    query_meta: list[dict[str, str]],
    gallery_features: np.ndarray,
    gallery_meta: pd.DataFrame,
    prior_field: str,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    query_vectors = l2_normalize(query_features.astype(np.float32))
    gallery_vectors = l2_normalize(gallery_features.astype(np.float32))

    prior_to_gallery_indices: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(gallery_meta[prior_field].astype(str).tolist()):
        prior_to_gallery_indices[str(value)].append(index)

    all_scores = np.full((len(query_meta), top_k), -np.inf, dtype=np.float32)
    all_indices = np.full((len(query_meta), top_k), -1, dtype=np.int64)
    candidate_counts: list[int] = []

    for query_index, query_row in enumerate(query_meta):
        prior_value = str(query_row[prior_field])
        candidate_indices = prior_to_gallery_indices.get(prior_value, [])
        candidate_counts.append(len(candidate_indices))
        if not candidate_indices:
            continue

        candidate_array = np.asarray(candidate_indices, dtype=np.int64)
        similarities = gallery_vectors[candidate_array] @ query_vectors[query_index]
        local_top_indices = topk_indices(similarities, top_k=top_k)
        global_top_indices = candidate_array[local_top_indices]

        count = len(global_top_indices)
        all_indices[query_index, :count] = global_top_indices
        all_scores[query_index, :count] = similarities[local_top_indices].astype(np.float32)

    return all_scores, all_indices, candidate_counts


def retrieved_values_from_indices(
    indices: np.ndarray,
    gallery_meta: pd.DataFrame,
    column: str,
) -> np.ndarray:
    values = gallery_meta[column].astype(str).to_numpy()
    retrieved = np.full(indices.shape, "", dtype=object)
    valid = indices >= 0
    if np.any(valid):
        retrieved[valid] = values[indices[valid]]
    return retrieved


def write_patch_results(
    args: argparse.Namespace,
    query_meta: list[dict[str, str]],
    gallery_meta: pd.DataFrame,
    scores: np.ndarray,
    indices: np.ndarray,
    candidate_counts: list[int],
) -> None:
    rows: list[dict[str, Any]] = []
    result_top_k = min(args.patch_results_top_k, indices.shape[1])

    for query_index, query_row in enumerate(query_meta):
        for rank_index in range(result_top_k):
            gallery_index = int(indices[query_index, rank_index])
            if gallery_index < 0:
                continue
            gallery_row = gallery_meta.iloc[gallery_index]
            rows.append(
                {
                    "query_positive_id": query_row["positive_id"],
                    "query_building_id": query_row["building_id"],
                    "query_tile_id": query_row["tile_id"],
                    "query_disaster": query_row["disaster"],
                    "query_disaster_type": query_row["disaster_type"],
                    "query_damage_label": query_row["damage_label"],
                    "query_patch_path": query_row["query_patch_path"],
                    "prior_field": args.prior_field,
                    "prior_value": query_row[args.prior_field],
                    "candidate_patch_count": candidate_counts[query_index],
                    "rank": rank_index + 1,
                    "score": float(scores[query_index, rank_index]),
                    "retrieved_positive_id": str(gallery_row["positive_id"]),
                    "retrieved_building_id": str(gallery_row["building_id"]),
                    "retrieved_tile_id": str(gallery_row["tile_id"]),
                    "retrieved_disaster": str(gallery_row["disaster"]),
                    "retrieved_disaster_type": str(gallery_row["disaster_type"]),
                    "retrieved_damage_label": str(gallery_row["damage_label"]),
                    "retrieved_patch_path": str(gallery_row["image_path"]),
                    "is_match": query_row["positive_id"] == str(gallery_row["positive_id"]),
                }
            )

    args.patch_results_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.patch_results_csv, index=False)


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


def first_value(rows: list[dict[str, str]], key: str) -> str:
    values = [str(row[key]) for row in rows if key in row]
    return values[0] if values else ""


def aggregate_tile_rankings(
    args: argparse.Namespace,
    query_meta: list[dict[str, str]],
    gallery_meta: pd.DataFrame,
    patch_scores: np.ndarray,
    patch_indices: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str], list[str]]:
    query_groups: dict[str, list[int]] = defaultdict(list)
    for query_index, row in enumerate(query_meta):
        query_groups[str(row["tile_id"])].append(query_index)

    gallery_tile_patch_counts = gallery_meta["tile_id"].astype(str).value_counts().to_dict()
    result_rows: list[dict[str, Any]] = []
    retrieved_tile_rows: list[list[str]] = []
    query_tile_ids: list[str] = []
    query_disasters: list[str] = []
    query_disaster_types: list[str] = []

    for query_tile_id in sorted(query_groups):
        query_indices = query_groups[query_tile_id]
        query_rows = [query_meta[index] for index in query_indices]
        candidate_scores: dict[str, list[float]] = defaultdict(list)
        candidate_disaster: dict[str, str] = {}
        candidate_disaster_type: dict[str, str] = {}

        for query_index in query_indices:
            for score, gallery_index in zip(patch_scores[query_index], patch_indices[query_index], strict=True):
                gallery_index = int(gallery_index)
                if gallery_index < 0:
                    continue
                gallery_row = gallery_meta.iloc[gallery_index]
                retrieved_tile_id = str(gallery_row["tile_id"])
                candidate_scores[retrieved_tile_id].append(float(score))
                candidate_disaster.setdefault(retrieved_tile_id, str(gallery_row.get("disaster", "")))
                candidate_disaster_type.setdefault(retrieved_tile_id, str(gallery_row.get("disaster_type", "")))

        scored_candidates: list[tuple[str, float, float, int]] = []
        for retrieved_tile_id, scores in candidate_scores.items():
            aggregate_score, best_patch_score, hit_count = score_candidate(
                scores=scores,
                aggregation=args.aggregation,
                top_m=args.top_m,
            )
            scored_candidates.append((retrieved_tile_id, aggregate_score, best_patch_score, hit_count))

        scored_candidates.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        ranked = scored_candidates[: args.tile_top_k]
        retrieved_tile_ids = [item[0] for item in ranked]
        if len(retrieved_tile_ids) < args.tile_top_k:
            retrieved_tile_ids.extend([""] * (args.tile_top_k - len(retrieved_tile_ids)))

        query_tile_ids.append(query_tile_id)
        query_disasters.append(first_value(query_rows, "disaster"))
        query_disaster_types.append(first_value(query_rows, "disaster_type"))
        retrieved_tile_rows.append(retrieved_tile_ids)

        for rank_index, (retrieved_tile_id, aggregate_score, best_patch_score, hit_count) in enumerate(ranked, start=1):
            result_rows.append(
                {
                    "query_tile_id": query_tile_id,
                    "query_disaster": first_value(query_rows, "disaster"),
                    "query_disaster_type": first_value(query_rows, "disaster_type"),
                    "prior_field": args.prior_field,
                    "prior_value": first_value(query_rows, args.prior_field),
                    "query_patch_count": len(query_indices),
                    "rank": rank_index,
                    "score": aggregate_score,
                    "best_patch_score": best_patch_score,
                    "patch_hit_count": hit_count,
                    "retrieved_tile_id": retrieved_tile_id,
                    "retrieved_disaster": candidate_disaster.get(retrieved_tile_id, ""),
                    "retrieved_disaster_type": candidate_disaster_type.get(retrieved_tile_id, ""),
                    "retrieved_tile_patch_count": int(gallery_tile_patch_counts.get(retrieved_tile_id, 0)),
                    "is_match": retrieved_tile_id == query_tile_id,
                }
            )

    return (
        pd.DataFrame(result_rows),
        np.asarray(retrieved_tile_rows, dtype=object),
        query_tile_ids,
        query_disasters,
        query_disaster_types,
    )


def candidate_count_summary(candidate_counts: list[int]) -> dict[str, float | int]:
    if not candidate_counts:
        return {"min": 0, "max": 0, "mean": 0.0}
    values = np.asarray(candidate_counts, dtype=np.float32)
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    if not args.features.exists():
        raise FileNotFoundError(
            f"Gallery feature file not found: {args.features}. "
            "Run extract_pre_clip_features.py with the stage1v2 checkpoint first."
        )

    gallery_features = np.load(args.features).astype(np.float32)
    gallery_meta = load_gallery_metadata(args).reset_index(drop=True)
    if gallery_features.shape[0] != len(gallery_meta):
        raise ValueError("Gallery feature count does not match gallery metadata rows")

    model, checkpoint = load_model(args, device)
    query_features, query_meta = extract_post_embeddings(args=args, model=model, device=device)
    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError("Query and gallery feature dimensions do not match")

    validate_prior_field(args.prior_field, query_meta=query_meta, gallery_meta=gallery_meta)
    patch_scores, patch_indices, candidate_counts = search_with_prior(
        query_features=query_features,
        query_meta=query_meta,
        gallery_features=gallery_features,
        gallery_meta=gallery_meta,
        prior_field=args.prior_field,
        top_k=args.patch_top_k,
    )

    patch_metric_top_k = min(args.patch_metric_top_k, args.patch_top_k)
    patch_retrieved_ids = retrieved_values_from_indices(
        patch_indices[:, :patch_metric_top_k],
        gallery_meta=gallery_meta,
        column="positive_id",
    )
    query_positive_ids = [row["positive_id"] for row in query_meta]
    query_disasters = [row["disaster"] for row in query_meta]
    query_disaster_types = [row["disaster_type"] for row in query_meta]
    query_damage_labels = [row["damage_label"] for row in query_meta]

    patch_metrics = compute_metrics(
        query_positive_ids,
        patch_retrieved_ids,
        ks=metric_ks(patch_metric_top_k),
    )
    patch_by_disaster = compute_grouped_metrics(
        query_positive_ids,
        patch_retrieved_ids,
        groups=query_disasters,
        ks=metric_ks(patch_metric_top_k),
    )
    patch_by_disaster_type = compute_grouped_metrics(
        query_positive_ids,
        patch_retrieved_ids,
        groups=query_disaster_types,
        ks=metric_ks(patch_metric_top_k),
    )
    patch_by_damage = compute_grouped_metrics(
        query_positive_ids,
        patch_retrieved_ids,
        groups=query_damage_labels,
        ks=metric_ks(patch_metric_top_k),
    )

    write_patch_results(
        args=args,
        query_meta=query_meta,
        gallery_meta=gallery_meta,
        scores=patch_scores,
        indices=patch_indices,
        candidate_counts=candidate_counts,
    )

    tile_frame, retrieved_tiles, query_tile_ids, tile_query_disasters, tile_query_disaster_types = aggregate_tile_rankings(
        args=args,
        query_meta=query_meta,
        gallery_meta=gallery_meta,
        patch_scores=patch_scores,
        patch_indices=patch_indices,
    )
    args.tile_results_csv.parent.mkdir(parents=True, exist_ok=True)
    tile_frame.to_csv(args.tile_results_csv, index=False)

    tile_metrics = compute_metrics(
        query_tile_ids,
        retrieved_tiles,
        ks=metric_ks(args.tile_top_k),
    )
    tile_by_disaster = compute_grouped_metrics(
        query_tile_ids,
        retrieved_tiles,
        groups=tile_query_disasters,
        ks=metric_ks(args.tile_top_k),
    )
    tile_by_disaster_type = compute_grouped_metrics(
        query_tile_ids,
        retrieved_tiles,
        groups=tile_query_disaster_types,
        ks=metric_ks(args.tile_top_k),
    )

    metrics = {
        "prior": {
            "field": args.prior_field,
            "candidate_patch_count": candidate_count_summary(candidate_counts),
        },
        "patch": {
            "overall": patch_metrics,
            "macro_by_disaster": compute_macro_metrics(patch_by_disaster),
            "macro_by_disaster_type": compute_macro_metrics(patch_by_disaster_type),
            "macro_by_damage_label": compute_macro_metrics(patch_by_damage),
            "by_disaster": patch_by_disaster,
            "by_disaster_type": patch_by_disaster_type,
            "by_damage_label": patch_by_damage,
            "query_patch_count": len(query_meta),
            "gallery_patch_count": int(gallery_features.shape[0]),
            "metric_top_k": patch_metric_top_k,
            "search_top_k": args.patch_top_k,
        },
        "tile": {
            "overall": tile_metrics,
            "macro_by_disaster": compute_macro_metrics(tile_by_disaster),
            "macro_by_disaster_type": compute_macro_metrics(tile_by_disaster_type),
            "by_disaster": tile_by_disaster,
            "by_disaster_type": tile_by_disaster_type,
            "query_tile_count": len(query_tile_ids),
            "gallery_tile_count": int(gallery_meta["tile_id"].astype(str).nunique()),
            "tile_top_k": args.tile_top_k,
            "aggregation": args.aggregation,
            "top_m": args.top_m,
        },
        "checkpoint": str(args.checkpoint),
        "features": str(args.features),
        "split": args.split,
        "model_config": checkpoint.get("model_config", {}),
        "outputs": {
            "patch_results_csv": str(args.patch_results_csv),
            "tile_results_csv": str(args.tile_results_csv),
            "metrics_json": str(args.metrics_json),
        },
    }

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_json.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "prior": metrics["prior"],
                "patch": metrics["patch"]["overall"],
                "tile": metrics["tile"]["overall"],
            },
            ensure_ascii=False,
        )
    )
    print(f"Patch-level retrieval results saved to: {args.patch_results_csv}")
    print(f"Tile-level retrieval results saved to: {args.tile_results_csv}")
    print(f"Metrics saved to: {args.metrics_json}")


if __name__ == "__main__":
    main()
