from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.transforms import build_clip_transform
from src.datasets.xbd_building_dataset import XBDBuildingPatchDataset, collate_single_view, load_building_patch_table
from src.models.clip_visual_encoder import DEFAULT_CLIP_MODEL_NAME, load_model_checkpoint
from src.retrieval.faiss_index import build_index, load_index, save_index, search_index
from src.retrieval.metrics import compute_grouped_metrics, compute_macro_metrics, compute_metrics
from src.utils.device import get_device


DEFAULT_PATCH_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "clip_visual_baseline.pt"
DEFAULT_FEATURES = REPO_ROOT / "indexes" / "pre_clip_features.npy"
DEFAULT_INDEX = REPO_ROOT / "indexes" / "pre_clip_tile.faiss"
DEFAULT_RESULTS = REPO_ROOT / "outputs" / "predictions" / "clip_tile_retrieval_results.csv"
DEFAULT_METRICS = REPO_ROOT / "outputs" / "predictions" / "clip_tile_retrieval_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image/tile-level retrieval by aggregating patch-level CLIP retrieval results.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_PATCH_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="Pre-disaster gallery patch feature matrix.")
    parser.add_argument("--features-meta", type=Path, default=None, help="Optional metadata CSV saved during extraction.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--split", type=str, default="hold")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch-top-k", type=int, default=100, help="Number of pre patches retrieved per post patch.")
    parser.add_argument("--tile-top-k", type=int, default=10, help="Number of pre tiles returned per post tile.")
    parser.add_argument("--top-m", type=int, default=5, help="Number of strongest patch scores averaged for top_m aggregation.")
    parser.add_argument(
        "--aggregation",
        type=str,
        default="top_m",
        choices=("top_m", "max", "mean", "sum", "vote"),
        help="How patch-level scores are aggregated into tile-level scores.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--include-unclassified", action="store_true")
    parser.add_argument("--use-existing-index", action="store_true")
    parser.add_argument("--model-name", type=str, default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return autocast(enabled=enabled)
    return nullcontext()


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
    model, checkpoint = load_model_checkpoint(
        args.checkpoint,
        map_location=device,
        model_name=args.model_name,
        local_files_only=args.local_files_only,
    )
    return model.to(device), checkpoint


def extract_post_patch_embeddings(
    args: argparse.Namespace,
    model,
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
    amp_enabled = bool(args.amp and device.type == "cuda")

    embeddings: list[np.ndarray] = []
    metadata_rows: list[dict[str, str]] = []
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


def build_or_load_index(args: argparse.Namespace, gallery_features: np.ndarray):
    if args.use_existing_index and args.index.exists():
        return load_index(args.index)
    index = build_index(gallery_features)
    save_index(index, args.index)
    return index


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
    query_meta: list[dict[str, str]],
    gallery_meta: pd.DataFrame,
    patch_scores: np.ndarray,
    patch_indices: np.ndarray,
    tile_top_k: int,
    aggregation: str,
    top_m: int,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    query_groups: dict[str, list[int]] = defaultdict(list)
    for query_index, row in enumerate(query_meta):
        query_groups[str(row["tile_id"])].append(query_index)

    gallery_tile_patch_counts = gallery_meta["tile_id"].astype(str).value_counts().to_dict()
    result_rows: list[dict[str, object]] = []
    retrieved_tile_rows: list[list[str]] = []
    query_tile_ids: list[str] = []
    query_disaster_types: list[str] = []

    for query_tile_id in sorted(query_groups):
        query_indices = query_groups[query_tile_id]
        query_rows = [query_meta[index] for index in query_indices]
        candidate_scores: dict[str, list[float]] = defaultdict(list)
        candidate_disaster: dict[str, str] = {}
        candidate_disaster_type: dict[str, str] = {}

        for query_index in query_indices:
            for score, gallery_index in zip(patch_scores[query_index], patch_indices[query_index], strict=True):
                gallery_row = gallery_meta.iloc[int(gallery_index)]
                retrieved_tile_id = str(gallery_row["tile_id"])
                candidate_scores[retrieved_tile_id].append(float(score))
                candidate_disaster.setdefault(retrieved_tile_id, str(gallery_row.get("disaster", "")))
                candidate_disaster_type.setdefault(retrieved_tile_id, str(gallery_row.get("disaster_type", "")))

        scored_candidates: list[tuple[str, float, float, int]] = []
        for retrieved_tile_id, scores in candidate_scores.items():
            aggregate_score, best_patch_score, hit_count = score_candidate(
                scores=scores,
                aggregation=aggregation,
                top_m=top_m,
            )
            scored_candidates.append((retrieved_tile_id, aggregate_score, best_patch_score, hit_count))

        scored_candidates.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        ranked = scored_candidates[:tile_top_k]
        retrieved_tile_ids = [item[0] for item in ranked]
        if len(retrieved_tile_ids) < tile_top_k:
            retrieved_tile_ids.extend([""] * (tile_top_k - len(retrieved_tile_ids)))

        query_tile_ids.append(query_tile_id)
        query_disaster_types.append(first_value(query_rows, "disaster_type"))
        retrieved_tile_rows.append(retrieved_tile_ids)

        for rank_index, (retrieved_tile_id, aggregate_score, best_patch_score, hit_count) in enumerate(ranked, start=1):
            result_rows.append(
                {
                    "query_tile_id": query_tile_id,
                    "query_disaster": first_value(query_rows, "disaster"),
                    "query_disaster_type": first_value(query_rows, "disaster_type"),
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

    retrieved_matrix = np.asarray(retrieved_tile_rows, dtype=object)
    return pd.DataFrame(result_rows), retrieved_matrix, query_tile_ids, query_disaster_types


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    gallery_features = np.load(args.features).astype(np.float32)
    gallery_meta = load_gallery_metadata(args).reset_index(drop=True)
    if gallery_features.shape[0] != len(gallery_meta):
        raise ValueError("Gallery feature count does not match gallery metadata rows")

    model, checkpoint = load_model(args, device)
    query_features, query_meta = extract_post_patch_embeddings(args=args, model=model, device=device)
    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError("Query and gallery feature dimensions do not match")

    index = build_or_load_index(args, gallery_features)
    patch_scores, patch_indices = search_index(index, query_features, top_k=args.patch_top_k)
    result_frame, retrieved_tiles, query_tile_ids, query_disaster_types = aggregate_tile_rankings(
        query_meta=query_meta,
        gallery_meta=gallery_meta,
        patch_scores=patch_scores,
        patch_indices=patch_indices,
        tile_top_k=args.tile_top_k,
        aggregation=args.aggregation,
        top_m=args.top_m,
    )

    metrics = compute_metrics(query_tile_ids, retrieved_tiles, ks=(1, 5, args.tile_top_k))
    grouped_metrics = compute_grouped_metrics(
        query_tile_ids,
        retrieved_tiles,
        groups=query_disaster_types,
        ks=(1, 5, args.tile_top_k),
    )
    macro_metrics = compute_macro_metrics(grouped_metrics)

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(args.results_csv, index=False)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "overall": metrics,
                "macro": macro_metrics,
                "by_disaster_type": grouped_metrics,
                "query_tile_count": len(query_tile_ids),
                "gallery_tile_count": int(gallery_meta["tile_id"].astype(str).nunique()),
                "query_patch_count": len(query_meta),
                "gallery_patch_count": int(gallery_features.shape[0]),
                "patch_top_k": args.patch_top_k,
                "tile_top_k": args.tile_top_k,
                "aggregation": args.aggregation,
                "top_m": args.top_m,
                "checkpoint": str(args.checkpoint),
                "features": str(args.features),
                "index_path": str(args.index),
                "model_config": checkpoint.get("model_config", {}),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        json.dumps(
            {
                "overall": metrics,
                "macro": macro_metrics,
                "by_disaster_type": grouped_metrics,
                "query_tile_count": len(query_tile_ids),
                "gallery_tile_count": int(gallery_meta["tile_id"].astype(str).nunique()),
                "aggregation": args.aggregation,
                "top_m": args.top_m,
            },
            ensure_ascii=False,
        )
    )
    print(f"Tile-level retrieval results saved to: {args.results_csv}")
    print(f"Tile-level metrics saved to: {args.metrics_json}")


if __name__ == "__main__":
    main()
