from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.transforms import build_clip_transform
from src.datasets.xbd_building_dataset import (
    XBDBuildingPatchDataset,
    collate_single_view,
    load_building_patch_table,
)
from src.models.clip_visual_encoder import (
    CLIPVisualEncoder,
    DEFAULT_CLIP_MODEL_NAME,
    load_model_checkpoint,
    resolve_clip_model_source,
)
from src.retrieval.faiss_index import build_index, load_index, save_index, search_index
from src.retrieval.metrics import compute_grouped_metrics, compute_macro_metrics, compute_metrics
from src.utils.device import get_device


DEFAULT_PATCH_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "clip_visual_baseline.pt"
DEFAULT_FEATURES = REPO_ROOT / "indexes" / "pre_clip_features.npy"
DEFAULT_INDEX = REPO_ROOT / "indexes" / "pre_clip.faiss"
DEFAULT_RESULTS = REPO_ROOT / "outputs" / "predictions" / "clip_retrieval_results.csv"
DEFAULT_METRICS = REPO_ROOT / "outputs" / "predictions" / "clip_retrieval_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLIP building retrieval with FAISS.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_PATCH_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="Pre-disaster gallery feature matrix.")
    parser.add_argument("--features-meta", type=Path, default=None, help="Optional metadata CSV saved during extraction.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--split", type=str, default="hold")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--include-unclassified", action="store_true")
    parser.add_argument("--use-existing-index", action="store_true")
    parser.add_argument("--model-name", type=str, default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--semantic-field", type=str, default=None)
    parser.add_argument(
        "--semantic-fields",
        nargs="+",
        action="append",
        default=None,
        help=(
            "One or more query text fields. Values may also be comma-separated, "
            "e.g. disaster_type,damage_label. When omitted, checkpoint semantic_fields "
            "or --semantic-field is used."
        ),
    )
    parser.add_argument("--text-template", type=str, default=None)
    parser.add_argument(
        "--disable-semantic-query",
        action="store_true",
        help="Evaluate query as image-only even if the checkpoint supports semantic query fusion.",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[CLIPVisualEncoder, dict[str, object]]:
    model, checkpoint = load_model_checkpoint(
        args.checkpoint,
        map_location=device,
        model_name=args.model_name,
        local_files_only=args.local_files_only,
    )
    return model.to(device), checkpoint


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
    return table[["positive_id", "building_id", "tile_id", "disaster", "disaster_type", "damage_label", "split", "pre_patch_path"]].rename(
        columns={"pre_patch_path": "image_path"}
    )


def humanize_label(label: str) -> str:
    return str(label).replace("_", " ").replace("-", " ")


def parse_semantic_fields(raw_fields: list[list[str]] | None) -> list[str] | None:
    if not raw_fields:
        return None

    fields: list[str] = []
    for field_group in raw_fields:
        for raw_field in field_group:
            fields.extend(
                field.strip()
                for field in str(raw_field).split(",")
                if field.strip()
            )
    return fields or None


def semantic_field_name(semantic_fields: list[str]) -> str:
    return "+".join(semantic_fields)


def semantic_label_from_values(values: tuple[str, ...]) -> str:
    return " | ".join(values)


def format_text_prompt(text_template: str, semantic_fields: list[str], values: tuple[str, ...]) -> str:
    if len(values) != len(semantic_fields):
        raise ValueError("Semantic prompt values must align with semantic_fields")

    label = semantic_label_from_values(values)
    format_values = {
        "label": humanize_label(label),
        "raw_label": label,
        "semantic_field": semantic_field_name(semantic_fields),
        "semantic_fields": semantic_field_name(semantic_fields),
    }
    for field, value in zip(semantic_fields, values):
        format_values[field] = humanize_label(value)
        format_values[f"raw_{field}"] = value
    return text_template.format(**format_values)


def load_tokenizer(model_name: str, local_files_only: bool) -> CLIPTokenizer:
    resolved_model_name, resolved_local_files_only = resolve_clip_model_source(
        model_name=model_name,
        local_files_only=local_files_only,
    )
    return CLIPTokenizer.from_pretrained(
        resolved_model_name,
        local_files_only=resolved_local_files_only,
    )


def resolve_semantic_query_config(
    args: argparse.Namespace,
    model: CLIPVisualEncoder,
    checkpoint: dict[str, object],
) -> tuple[bool, list[str], str | None]:
    semantic_config = checkpoint.get("semantic_config", {})
    semantic_config = semantic_config if isinstance(semantic_config, dict) else {}
    semantic_fields = parse_semantic_fields(args.semantic_fields)
    if semantic_fields is None:
        checkpoint_fields = semantic_config.get("semantic_fields")
        if isinstance(checkpoint_fields, list):
            semantic_fields = [str(field) for field in checkpoint_fields if str(field)]
    if semantic_fields is None:
        semantic_field = args.semantic_field or semantic_config.get("semantic_field")
        semantic_fields = [str(semantic_field)] if semantic_field else []

    text_template = args.text_template or semantic_config.get("text_template")
    use_semantic_query = (
        not args.disable_semantic_query
        and hasattr(model, "fuse_query_embeddings")
        and isinstance(text_template, str)
        and bool(semantic_fields)
        and bool(text_template)
    )
    return use_semantic_query, semantic_fields, str(text_template) if text_template else None


def extract_query_embeddings(
    args: argparse.Namespace,
    model: CLIPVisualEncoder,
    checkpoint: dict[str, object],
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
    use_semantic_query, semantic_fields, text_template = resolve_semantic_query_config(
        args=args,
        model=model,
        checkpoint=checkpoint,
    )
    tokenizer = load_tokenizer(args.model_name, args.local_files_only) if use_semantic_query else None
    embeddings: list[np.ndarray] = []
    metadata_rows: list[dict[str, str]] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            with amp_context(device, amp_enabled):
                if use_semantic_query:
                    missing_fields = [field for field in semantic_fields if field not in batch]
                    if missing_fields:
                        raise ValueError(f"Semantic field(s) are not available in query batch: {missing_fields}")
                    prompts = [
                        format_text_prompt(
                            text_template=str(text_template),
                            semantic_fields=semantic_fields,
                            values=tuple(str(batch[field][index]) for field in semantic_fields),
                        )
                        for index in range(len(batch["positive_id"]))
                    ]
                    tokenized = tokenizer(
                        prompts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_embeddings = model.encode_text(
                        input_ids=tokenized["input_ids"].to(device),
                        attention_mask=tokenized["attention_mask"].to(device),
                    )
                    image_embeddings = model.encode_image(images)
                    batch_embeddings = model.fuse_query_embeddings(image_embeddings, text_embeddings)
                else:
                    prompts = [""] * len(batch["positive_id"])
                    batch_embeddings = model.encode_image(images)
            embeddings.append(batch_embeddings.cpu().numpy().astype(np.float32))

            for index in range(len(batch["positive_id"])):
                metadata_rows.append(
                    {
                        "positive_id": batch["positive_id"][index],
                        "building_id": batch["building_id"][index],
                        "tile_id": batch["tile_id"][index],
                        "disaster": batch["disaster"][index],
                        "disaster_type": batch["disaster_type"][index],
                        "damage_label": batch["damage_label"][index],
                        "split": batch["split"][index],
                        "query_patch_path": batch["image_path"][index],
                        "query_text_prompt": prompts[index],
                    }
                )

    query_matrix = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)
    return query_matrix, metadata_rows


def build_or_load_index(args: argparse.Namespace, gallery_features: np.ndarray):
    if args.use_existing_index and args.index.exists():
        return load_index(args.index)

    index = build_index(gallery_features)
    save_index(index, args.index)
    return index


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    gallery_features = np.load(args.features).astype(np.float32)
    gallery_meta = load_gallery_metadata(args).reset_index(drop=True)
    if gallery_features.shape[0] != len(gallery_meta):
        raise ValueError("Gallery feature count does not match gallery metadata rows")

    model, checkpoint = load_model(args, device)
    query_features, query_meta = extract_query_embeddings(args, model, checkpoint, device)
    use_semantic_query, semantic_fields, text_template = resolve_semantic_query_config(
        args=args,
        model=model,
        checkpoint=checkpoint,
    )
    index = build_or_load_index(args, gallery_features)

    scores, indices = search_index(index, query_features, top_k=args.top_k)
    gallery_positive_ids = gallery_meta["positive_id"].astype(str).to_numpy()
    retrieved_ids = gallery_positive_ids[indices]
    query_positive_ids = [row["positive_id"] for row in query_meta]
    query_disaster_types = [row["disaster_type"] for row in query_meta]
    query_damage_labels = [row["damage_label"] for row in query_meta]

    metrics = compute_metrics(query_positive_ids, retrieved_ids, ks=(1, 5, args.top_k))
    grouped_metrics = compute_grouped_metrics(
        query_positive_ids,
        retrieved_ids,
        groups=query_disaster_types,
        ks=(1, 5, args.top_k),
    )
    macro_metrics = compute_macro_metrics(grouped_metrics)
    damage_grouped_metrics = compute_grouped_metrics(
        query_positive_ids,
        retrieved_ids,
        groups=query_damage_labels,
        ks=(1, 5, args.top_k),
    )
    damage_macro_metrics = compute_macro_metrics(damage_grouped_metrics)

    result_rows: list[dict[str, str | int | float | bool]] = []
    for query_index, query_row in enumerate(query_meta):
        for rank_index in range(indices.shape[1]):
            gallery_index = int(indices[query_index, rank_index])
            gallery_row = gallery_meta.iloc[gallery_index]
            result_rows.append(
                {
                    "query_positive_id": query_row["positive_id"],
                    "query_building_id": query_row["building_id"],
                    "query_tile_id": query_row["tile_id"],
                    "query_disaster": query_row["disaster"],
                    "query_disaster_type": query_row["disaster_type"],
                    "query_damage_label": query_row["damage_label"],
                    "query_patch_path": query_row["query_patch_path"],
                    "query_text_prompt": query_row.get("query_text_prompt", ""),
                    "rank": rank_index + 1,
                    "score": float(scores[query_index, rank_index]),
                    "retrieved_positive_id": str(gallery_row["positive_id"]),
                    "retrieved_building_id": str(gallery_row["building_id"]),
                    "retrieved_tile_id": str(gallery_row["tile_id"]),
                    "retrieved_disaster": str(gallery_row["disaster"]),
                    "retrieved_disaster_type": str(gallery_row["disaster_type"]),
                    "retrieved_damage_label": str(gallery_row["damage_label"]),
                    "retrieved_patch_path": str(gallery_row["image_path"]),
                    "is_match": bool(query_row["positive_id"] == str(gallery_row["positive_id"])),
                }
            )

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(args.results_csv, index=False)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    semantic_field_value = None
    if semantic_fields:
        semantic_field_value = semantic_fields[0] if len(semantic_fields) == 1 else semantic_field_name(semantic_fields)

    with args.metrics_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "overall": metrics,
                "macro": macro_metrics,
                "by_disaster_type": grouped_metrics,
                "by_damage_label": damage_grouped_metrics,
                "macro_by_damage_label": damage_macro_metrics,
                "gallery_size": int(gallery_features.shape[0]),
                "query_size": int(query_features.shape[0]),
                "index_path": str(args.index),
                "query_mode": "post_image_plus_disaster_text" if use_semantic_query else "post_image_only",
                "semantic_field": semantic_field_value,
                "semantic_fields": semantic_fields,
                "text_template": text_template,
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
                "by_damage_label": damage_grouped_metrics,
            },
            ensure_ascii=False,
        )
    )
    print(f"Retrieval results saved to: {args.results_csv}")
    print(f"Metrics saved to: {args.metrics_json}")


if __name__ == "__main__":
    main()
