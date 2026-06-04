from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.transforms import build_clip_transform
from src.datasets.xbd_building_dataset import (
    XBDBuildingPatchDataset,
    collate_building_pairs,
)
from src.models.clip_visual_encoder import (
    CLIPSemanticEncoder,
    DEFAULT_CLIP_MODEL_NAME,
    load_torch_checkpoint,
    resolve_clip_model_source,
)
from src.retrieval.faiss_index import build_index, search_index
from src.retrieval.metrics import compute_grouped_metrics, compute_macro_metrics, compute_metrics
from src.utils.device import get_device, setup_torch_runtime


DEFAULT_PATCH_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches_smoke_stage1.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "clip_semantic_stage2_smoke.pt"
DEFAULT_TEXT_TEMPLATE = "a satellite image of a building after a {label} disaster"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train stage-2 CLIP retrieval with semantic disaster-text supervision.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_PATCH_CSV, help="Building patch CSV.")
    parser.add_argument("--train-split", type=str, default="tier3", help="Training split name.")
    parser.add_argument("--eval-split", type=str, default="hold", help="Validation split name.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Checkpoint path.")
    parser.add_argument(
        "--init-visual-checkpoint",
        type=Path,
        default=None,
        help="Optional stage-1 visual checkpoint used to initialize the vision backbone and projection head.",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--semantic-temperature", type=float, default=0.07)
    parser.add_argument("--semantic-loss-weight", type=float, default=0.05)
    parser.add_argument("--visual-loss-weight", type=float, default=1.0)
    parser.add_argument("--semantic-field", type=str, default="disaster_type")
    parser.add_argument(
        "--semantic-fields",
        nargs="+",
        action="append",
        default=None,
        help=(
            "One or more CSV fields used to build semantic labels and text prompts. "
            "Values may also be comma-separated, e.g. disaster_type,damage_label. "
            "When omitted, --semantic-field is used for backward compatibility."
        ),
    )
    parser.add_argument("--text-template", type=str, default=DEFAULT_TEXT_TEMPLATE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--amp", action="store_true", help="Enable AMP when CUDA is available.")
    parser.add_argument("--include-unclassified", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--random-init", action="store_true", help="Do not load original pretrained CLIP weights.")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Train the CLIP vision backbone.")
    parser.add_argument("--unfreeze-text-backbone", action="store_true", help="Train the CLIP text backbone.")
    parser.add_argument(
        "--freeze-image-projection",
        action="store_true",
        help="Freeze the image projection head after optional stage-1 visual initialization.",
    )
    parser.add_argument(
        "--disable-balanced-sampling",
        action="store_true",
        help="Disable inverse-frequency disaster-type sampling for the training loader.",
    )
    parser.add_argument(
        "--loss-weighting",
        type=str,
        default="inverse_freq",
        choices=("none", "inverse_freq"),
        help="Optional class-weighted losses using disaster-type inverse-frequency weights.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="macro_recall@1",
        choices=("macro_recall@1", "overall_recall@1"),
        help="Metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--query-fusion-mode",
        type=str,
        default="residual_gate",
        choices=("residual_gate", "concat_mlp"),
        help="How to inject disaster text into the post-image query embedding.",
    )
    parser.add_argument(
        "--detach-gallery-for-query-loss",
        action="store_true",
        help=(
            "Detach pre-disaster gallery embeddings in the semantic query-to-gallery loss. "
            "This keeps disaster text supervision on the post/query side while the pre-gallery "
            "space is maintained by the visual auxiliary loss."
        ),
    )
    parser.add_argument(
        "--evaluate-before-training",
        action="store_true",
        help="Evaluate and checkpoint the initialized model as epoch 0 before any semantic fine-tuning.",
    )
    parser.add_argument("--train-log-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pair_loader(
    csv_path: Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    include_unclassified: bool,
    train: bool,
    balanced_sampling: bool,
    sampler_seed: int,
) -> tuple[DataLoader, dict[str, int], dict[str, float]]:
    dataset = XBDBuildingPatchDataset(
        csv_path=csv_path,
        split=split,
        transform=build_clip_transform(image_size=image_size, train=train),
        include_unclassified=include_unclassified,
        view="pair",
    )

    disaster_types = [str(record["disaster_type"]) for record in dataset.records]
    class_counts = dict(Counter(disaster_types))
    total_samples = float(sum(class_counts.values()))
    class_weights = {
        disaster_type: total_samples / (len(class_counts) * count)
        for disaster_type, count in class_counts.items()
    } if class_counts else {}

    sampler = None
    shuffle = train
    if train and balanced_sampling and disaster_types:
        generator = torch.Generator()
        generator.manual_seed(sampler_seed)
        sample_weights = [class_weights[disaster_type] for disaster_type in disaster_types]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_building_pairs,
    )
    return dataloader, class_counts, class_weights


def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return autocast(enabled=enabled)
    return nullcontext()


def build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


def humanize_label(label: str) -> str:
    return str(label).replace("_", " ").replace("-", " ")


def parse_semantic_fields(raw_fields: list[list[str]] | None, fallback_field: str) -> list[str]:
    if not raw_fields:
        return [fallback_field]

    fields: list[str] = []
    for field_group in raw_fields:
        for raw_field in field_group:
            fields.extend(
                field.strip()
                for field in str(raw_field).split(",")
                if field.strip()
            )
    if not fields:
        raise ValueError("--semantic-fields did not contain any valid field names")
    return fields


def semantic_field_name(semantic_fields: list[str]) -> str:
    return "+".join(semantic_fields)


def semantic_label_from_values(values: tuple[str, ...]) -> str:
    return " | ".join(values)


def semantic_label_from_record(record: dict[str, object], semantic_fields: list[str]) -> str:
    missing_fields = [field for field in semantic_fields if field not in record]
    if missing_fields:
        raise ValueError(f"Semantic field(s) missing from record: {missing_fields}")
    return semantic_label_from_values(tuple(str(record[field]) for field in semantic_fields))


def semantic_label_from_batch(batch: dict[str, object], semantic_fields: list[str], index: int) -> str:
    values: list[str] = []
    missing_fields: list[str] = []
    for field in semantic_fields:
        field_values = batch.get(field)
        if field_values is None:
            missing_fields.append(field)
            continue
        values.append(str(field_values[index]))
    if missing_fields:
        raise ValueError(f"Semantic field(s) missing from batch: {missing_fields}")
    return semantic_label_from_values(tuple(values))


def format_semantic_prompt(
    text_template: str,
    semantic_fields: list[str],
    values: tuple[str, ...],
) -> str:
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


def build_semantic_classes(
    records: list[dict[str, object]],
    semantic_fields: list[str],
    text_template: str,
) -> tuple[list[str], dict[str, int], list[str]]:
    missing_fields = sorted(
        {
            field
            for field in semantic_fields
            if any(field not in record for record in records)
        }
    )
    if missing_fields:
        raise ValueError(f"Semantic field(s) missing from records: {missing_fields}")

    label_values = sorted(
        {
            tuple(str(record[field]) for field in semantic_fields)
            for record in records
        }
    )
    if not label_values:
        raise ValueError(f"No semantic labels found for fields: {semantic_fields}")

    prompts = [
        format_semantic_prompt(
            text_template=text_template,
            semantic_fields=semantic_fields,
            values=values,
        )
        for values in label_values
    ]
    labels = [semantic_label_from_values(values) for values in label_values]
    return labels, {label: index for index, label in enumerate(labels)}, prompts


def load_tokenizer(model_name: str, local_files_only: bool) -> CLIPTokenizer:
    resolved_model_name, resolved_local_files_only = resolve_clip_model_source(
        model_name=model_name,
        local_files_only=local_files_only,
    )
    return CLIPTokenizer.from_pretrained(
        resolved_model_name,
        local_files_only=resolved_local_files_only,
    )


def weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    if sample_weights is None:
        return values.mean()
    if sample_weights.ndim != 1 or sample_weights.shape[0] != values.shape[0]:
        raise ValueError("sample_weights must be rank-1 and aligned with the batch dimension")
    weights = sample_weights.to(device=values.device, dtype=values.dtype)
    weights = weights / weights.mean().clamp_min(1e-12)
    return (values * weights).mean()


def query_to_gallery_loss(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    temperature: float,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if query_embeddings.ndim != 2 or gallery_embeddings.ndim != 2:
        raise ValueError("query_embeddings and gallery_embeddings must be rank-2 tensors")
    if query_embeddings.shape != gallery_embeddings.shape:
        raise ValueError("query_embeddings and gallery_embeddings must have the same shape")

    logits = query_embeddings @ gallery_embeddings.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
    loss = weighted_mean(per_sample_loss, sample_weights)

    with torch.no_grad():
        top1 = (logits.argmax(dim=1) == labels).float().mean()

    return {
        "loss": loss,
        "retrieval_acc": top1.detach(),
    }


def initialize_visual_branch_from_checkpoint(
    model: CLIPSemanticEncoder,
    checkpoint_path: Path,
    map_location: torch.device,
) -> dict[str, object]:
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=map_location)
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid visual checkpoint format: {checkpoint_path}")

    model_state = model.state_dict()
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    for key, value in state.items():
        if not isinstance(key, str) or not key.startswith(("backbone.", "projection.")):
            continue
        if key not in model_state or tuple(model_state[key].shape) != tuple(value.shape):
            skipped_keys.append(key)
            continue
        model_state[key] = value
        loaded_keys.append(key)

    if not loaded_keys:
        raise ValueError(f"No compatible visual keys found in checkpoint: {checkpoint_path}")

    model.load_state_dict(model_state)
    return {
        "checkpoint": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "loaded_key_count": len(loaded_keys),
        "skipped_key_count": len(skipped_keys),
        "loaded_prefixes": ["backbone.", "projection."],
    }


def semantic_alignment_loss(
    post_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    post_logits = post_embeddings @ text_embeddings.t() / temperature
    loss = weighted_mean(F.cross_entropy(post_logits, targets, reduction="none"), sample_weights)

    with torch.no_grad():
        post_acc = (post_logits.argmax(dim=1) == targets).float().mean()

    return {
        "loss": loss,
        "loss_post_to_text": loss.detach(),
        "post_text_acc": post_acc.detach(),
        "text_acc": post_acc.detach(),
    }


def encode_eval_embeddings(
    model: CLIPSemanticEncoder,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    semantic_fields: list[str],
    semantic_label_to_id: dict[str, int],
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    gallery_embeddings: list[np.ndarray] = []
    query_embeddings: list[np.ndarray] = []
    positive_ids: list[str] = []
    disaster_types: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            pre_images = batch["pre_image"].to(device, non_blocking=True)
            post_images = batch["post_image"].to(device, non_blocking=True)
            semantic_targets = torch.tensor(
                [
                    semantic_label_to_id[semantic_label_from_batch(batch, semantic_fields, index)]
                    for index in range(len(batch["positive_id"]))
                ],
                dtype=torch.long,
                device=device,
            )
            with amp_context(device, amp_enabled):
                outputs = model(
                    pre_pixel_values=pre_images,
                    post_pixel_values=post_images,
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                )
                batch_text_embeddings = outputs["text_embeddings"].index_select(0, semantic_targets)
                batch_query_embeddings = model.fuse_query_embeddings(
                    outputs["post_embeddings"],
                    batch_text_embeddings,
                )

            gallery_embeddings.append(outputs["pre_embeddings"].cpu().numpy().astype(np.float32))
            query_embeddings.append(batch_query_embeddings.cpu().numpy().astype(np.float32))
            positive_ids.extend(batch["positive_id"])
            disaster_types.extend(batch["disaster_type"])

    gallery_matrix = np.concatenate(gallery_embeddings, axis=0) if gallery_embeddings else np.empty((0, 0), dtype=np.float32)
    query_matrix = np.concatenate(query_embeddings, axis=0) if query_embeddings else np.empty((0, 0), dtype=np.float32)
    return gallery_matrix, query_matrix, positive_ids, disaster_types


def evaluate_retrieval(
    model: CLIPSemanticEncoder,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    top_k: int,
    semantic_fields: list[str],
    semantic_label_to_id: dict[str, int],
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float]]:
    gallery_matrix, query_matrix, positive_ids, disaster_types = encode_eval_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        amp_enabled=amp_enabled,
        semantic_fields=semantic_fields,
        semantic_label_to_id=semantic_label_to_id,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
    )
    if len(positive_ids) == 0:
        return {}, {}, {}

    index = build_index(gallery_matrix)
    _, indices = search_index(index, query_matrix, top_k=top_k)
    gallery_ids = np.asarray(positive_ids)
    retrieved_ids = gallery_ids[indices]

    metrics = compute_metrics(positive_ids, retrieved_ids, ks=(1, 5, top_k))
    grouped = compute_grouped_metrics(
        positive_ids,
        retrieved_ids,
        groups=disaster_types,
        ks=(1, 5, top_k),
    )
    macro = compute_macro_metrics(grouped)
    return metrics, grouped, macro


def select_checkpoint_metric(
    overall_metrics: dict[str, float],
    macro_metrics: dict[str, float],
    selection_metric: str,
) -> float:
    if selection_metric == "macro_recall@1":
        return macro_metrics.get("recall@1", overall_metrics.get("recall@1", 0.0))
    return overall_metrics.get("recall@1", 0.0)


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: CLIPSemanticEncoder,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    args: argparse.Namespace,
    semantic_labels: list[str],
    semantic_prompts: list[str],
    overall_metrics: dict[str, float],
    grouped_metrics: dict[str, dict[str, float]],
    macro_metrics: dict[str, float],
    checkpoint_metric: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_config": model.get_config(),
        "semantic_config": {
            "semantic_field": args.semantic_field,
            "semantic_fields": args.semantic_fields_resolved,
            "text_template": args.text_template,
            "semantic_labels": semantic_labels,
            "semantic_prompts": semantic_prompts,
            "semantic_temperature": args.semantic_temperature,
            "semantic_loss_weight": args.semantic_loss_weight,
            "visual_loss_weight": args.visual_loss_weight,
            "query_fusion_mode": args.query_fusion_mode,
            "detach_gallery_for_query_loss": args.detach_gallery_for_query_loss,
            "init_visual_checkpoint": str(args.init_visual_checkpoint) if args.init_visual_checkpoint else None,
            "freeze_image_projection": args.freeze_image_projection,
            "query_mode": "post_image_plus_disaster_text",
        },
        "args": vars(args),
        "metrics": overall_metrics,
        "grouped_metrics": grouped_metrics,
        "macro_metrics": macro_metrics,
        "checkpoint_metric": checkpoint_metric,
    }
    torch.save(payload, checkpoint_path)


def append_jsonl(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, ensure_ascii=False, indent=2)


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    args.semantic_fields_resolved = parse_semantic_fields(
        raw_fields=args.semantic_fields,
        fallback_field=args.semantic_field,
    )
    setup_torch_runtime()
    set_seed(args.seed)

    if "query_fusion_mode" not in inspect.signature(CLIPSemanticEncoder).parameters:
        raise RuntimeError(
            "The imported CLIPSemanticEncoder does not support query_fusion_mode. "
            "Please sync src/models/clip_visual_encoder.py with the stage2 semantic code before training."
        )

    device = get_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    train_loader, train_class_counts, train_class_weights = build_pair_loader(
        csv_path=args.csv,
        split=args.train_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_unclassified=args.include_unclassified,
        train=True,
        balanced_sampling=not args.disable_balanced_sampling,
        sampler_seed=args.seed,
    )
    eval_loader, _, _ = build_pair_loader(
        csv_path=args.csv,
        split=args.eval_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_unclassified=args.include_unclassified,
        train=False,
        balanced_sampling=False,
        sampler_seed=args.seed,
    )

    train_records = list(getattr(train_loader.dataset, "records", []))
    semantic_labels, semantic_label_to_id, semantic_prompts = build_semantic_classes(
        records=train_records,
        semantic_fields=args.semantic_fields_resolved,
        text_template=args.text_template,
    )
    tokenizer = load_tokenizer(model_name=args.model_name, local_files_only=args.local_files_only)
    tokenized_prompts = tokenizer(
        semantic_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    prompt_input_ids = tokenized_prompts["input_ids"].to(device)
    prompt_attention_mask = tokenized_prompts["attention_mask"].to(device)

    model = CLIPSemanticEncoder(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        freeze_backbone=not args.unfreeze_backbone,
        freeze_text_backbone=not args.unfreeze_text_backbone,
        pretrained=not args.random_init,
        local_files_only=args.local_files_only,
        query_fusion_mode=args.query_fusion_mode,
    ).to(device)
    visual_init_report = None
    if args.init_visual_checkpoint is not None:
        visual_init_report = initialize_visual_branch_from_checkpoint(
            model=model,
            checkpoint_path=args.init_visual_checkpoint,
            map_location=device,
        )
    if args.freeze_image_projection:
        for parameter in model.projection.parameters():
            parameter.requires_grad = False

    pretrained_source = "stage1_visual_checkpoint" if args.init_visual_checkpoint else "original_clip"
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = build_grad_scaler(enabled=amp_enabled)

    print(
        json.dumps(
            {
                "stage": "stage2_semantic",
                "pretrained_source": pretrained_source,
                "train_disaster_type_counts": train_class_counts,
                "train_sampling_weights": train_class_weights,
                "balanced_sampling": not args.disable_balanced_sampling,
                "loss_weighting": args.loss_weighting,
                "selection_metric": args.selection_metric,
                "semantic_field": args.semantic_field,
                "semantic_fields": args.semantic_fields_resolved,
                "semantic_labels": semantic_labels,
                "semantic_prompts": semantic_prompts,
                "semantic_loss_weight": args.semantic_loss_weight,
                "visual_loss_weight": args.visual_loss_weight,
                "query_fusion_mode": args.query_fusion_mode,
                "detach_gallery_for_query_loss": args.detach_gallery_for_query_loss,
                "init_visual_checkpoint": str(args.init_visual_checkpoint) if args.init_visual_checkpoint else None,
                "visual_init_report": visual_init_report,
                "freeze_image_projection": args.freeze_image_projection,
                "trainable_parameter_count": sum(parameter.numel() for parameter in trainable_params),
                "query_mode": "post_image_plus_disaster_text",
            },
            ensure_ascii=False,
        )
    )

    best_recall = -1.0
    best_epoch = 0
    best_eval_metrics: dict[str, float] = {}
    best_eval_macro: dict[str, float] = {}
    best_eval_grouped: dict[str, dict[str, float]] = {}
    if args.evaluate_before_training:
        eval_metrics, eval_grouped, eval_macro = evaluate_retrieval(
            model=model,
            dataloader=eval_loader,
            device=device,
            amp_enabled=amp_enabled,
            top_k=args.top_k,
            semantic_fields=args.semantic_fields_resolved,
            semantic_label_to_id=semantic_label_to_id,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )
        checkpoint_metric = select_checkpoint_metric(
            overall_metrics=eval_metrics,
            macro_metrics=eval_macro,
            selection_metric=args.selection_metric,
        )
        best_recall = checkpoint_metric
        best_epoch = 0
        best_eval_metrics = dict(eval_metrics)
        best_eval_macro = dict(eval_macro)
        best_eval_grouped = dict(eval_grouped)
        save_checkpoint(
            checkpoint_path=args.checkpoint,
            epoch=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            semantic_labels=semantic_labels,
            semantic_prompts=semantic_prompts,
            overall_metrics=eval_metrics,
            grouped_metrics=eval_grouped,
            macro_metrics=eval_macro,
            checkpoint_metric=checkpoint_metric,
        )
        initial_payload = {
            "epoch": 0,
            "phase": "initialized_before_training",
            "train": {},
            "eval": eval_metrics,
            "eval_macro": eval_macro,
            "eval_by_disaster_type": eval_grouped,
            "checkpoint_metric": checkpoint_metric,
            "best_checkpoint_metric": best_recall,
            "best_epoch": best_epoch,
        }
        append_jsonl(args.train_log_jsonl, initial_payload)
        print(json.dumps(initial_payload, ensure_ascii=False))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_query_loss = 0.0
        running_visual_loss = 0.0
        running_semantic_loss = 0.0
        running_query_acc = 0.0
        running_visual_acc = 0.0
        running_text_acc = 0.0
        num_batches = 0

        for batch in train_loader:
            pre_images = batch["pre_image"].to(device, non_blocking=True)
            post_images = batch["post_image"].to(device, non_blocking=True)
            missing_fields = [field for field in args.semantic_fields_resolved if field not in batch]
            if missing_fields:
                raise ValueError(f"Semantic field(s) are not available in batch: {missing_fields}")
            semantic_targets = torch.tensor(
                [
                    semantic_label_to_id[semantic_label_from_batch(batch, args.semantic_fields_resolved, index)]
                    for index in range(len(batch["positive_id"]))
                ],
                dtype=torch.long,
                device=device,
            )

            sample_weights = None
            if args.loss_weighting == "inverse_freq" and train_class_weights:
                sample_weights = torch.tensor(
                    [train_class_weights[str(disaster_type)] for disaster_type in batch["disaster_type"]],
                    dtype=torch.float32,
                    device=device,
                )

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp_enabled):
                outputs = model(
                    pre_pixel_values=pre_images,
                    post_pixel_values=post_images,
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                )
                batch_text_embeddings = outputs["text_embeddings"].index_select(0, semantic_targets)
                query_embeddings = model.fuse_query_embeddings(
                    outputs["post_embeddings"],
                    batch_text_embeddings,
                )
                query_gallery_embeddings = (
                    outputs["pre_embeddings"].detach()
                    if args.detach_gallery_for_query_loss
                    else outputs["pre_embeddings"]
                )
                query_loss_dict = query_to_gallery_loss(
                    query_embeddings=query_embeddings,
                    gallery_embeddings=query_gallery_embeddings,
                    temperature=args.temperature,
                    sample_weights=sample_weights,
                )
                visual_loss_dict = query_to_gallery_loss(
                    query_embeddings=outputs["post_embeddings"],
                    gallery_embeddings=outputs["pre_embeddings"],
                    temperature=args.temperature,
                    sample_weights=sample_weights,
                )
                semantic_loss_dict = semantic_alignment_loss(
                    post_embeddings=outputs["post_embeddings"],
                    text_embeddings=outputs["text_embeddings"],
                    targets=semantic_targets,
                    temperature=args.semantic_temperature,
                    sample_weights=sample_weights,
                )
                loss = (
                    query_loss_dict["loss"]
                    + args.visual_loss_weight * visual_loss_dict["loss"]
                    + args.semantic_loss_weight * semantic_loss_dict["loss"]
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            running_query_loss += float(query_loss_dict["loss"].item())
            running_visual_loss += float(visual_loss_dict["loss"].item())
            running_semantic_loss += float(semantic_loss_dict["loss"].item())
            running_query_acc += float(query_loss_dict["retrieval_acc"].item())
            running_visual_acc += float(visual_loss_dict["retrieval_acc"].item())
            running_text_acc += float(semantic_loss_dict["text_acc"].item())
            num_batches += 1

        scheduler.step()
        train_metrics = {
            "loss": running_loss / max(num_batches, 1),
            "query_loss": running_query_loss / max(num_batches, 1),
            "visual_loss": running_visual_loss / max(num_batches, 1),
            "semantic_loss": running_semantic_loss / max(num_batches, 1),
            "query_retrieval_acc": running_query_acc / max(num_batches, 1),
            "visual_retrieval_acc": running_visual_acc / max(num_batches, 1),
            "semantic_text_acc": running_text_acc / max(num_batches, 1),
        }
        eval_metrics, eval_grouped, eval_macro = evaluate_retrieval(
            model=model,
            dataloader=eval_loader,
            device=device,
            amp_enabled=amp_enabled,
            top_k=args.top_k,
            semantic_fields=args.semantic_fields_resolved,
            semantic_label_to_id=semantic_label_to_id,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )
        checkpoint_metric = select_checkpoint_metric(
            overall_metrics=eval_metrics,
            macro_metrics=eval_macro,
            selection_metric=args.selection_metric,
        )

        if checkpoint_metric >= best_recall:
            best_recall = checkpoint_metric
            best_epoch = epoch
            best_eval_metrics = dict(eval_metrics)
            best_eval_macro = dict(eval_macro)
            best_eval_grouped = dict(eval_grouped)
            save_checkpoint(
                checkpoint_path=args.checkpoint,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                semantic_labels=semantic_labels,
                semantic_prompts=semantic_prompts,
                overall_metrics=eval_metrics,
                grouped_metrics=eval_grouped,
                macro_metrics=eval_macro,
                checkpoint_metric=checkpoint_metric,
            )

        epoch_payload = {
            "epoch": epoch,
            "train": train_metrics,
            "eval": eval_metrics,
            "eval_macro": eval_macro,
            "eval_by_disaster_type": eval_grouped,
            "checkpoint_metric": checkpoint_metric,
            "best_checkpoint_metric": best_recall,
            "best_epoch": best_epoch,
        }
        append_jsonl(args.train_log_jsonl, epoch_payload)
        print(json.dumps(epoch_payload, ensure_ascii=False))

    write_json(
        args.summary_json,
        {
            "checkpoint": str(args.checkpoint),
            "best_epoch": best_epoch,
            "best_checkpoint_metric": best_recall,
            "best_eval": best_eval_metrics,
            "best_eval_macro": best_eval_macro,
            "best_eval_by_disaster_type": best_eval_grouped,
            "train_disaster_type_counts": train_class_counts,
            "train_sampling_weights": train_class_weights,
            "balanced_sampling": not args.disable_balanced_sampling,
            "loss_weighting": args.loss_weighting,
            "selection_metric": args.selection_metric,
            "semantic_field": args.semantic_field,
            "semantic_fields": args.semantic_fields_resolved,
            "semantic_labels": semantic_labels,
            "semantic_prompts": semantic_prompts,
            "semantic_loss_weight": args.semantic_loss_weight,
            "visual_loss_weight": args.visual_loss_weight,
            "query_fusion_mode": args.query_fusion_mode,
            "detach_gallery_for_query_loss": args.detach_gallery_for_query_loss,
            "init_visual_checkpoint": str(args.init_visual_checkpoint) if args.init_visual_checkpoint else None,
            "visual_init_report": visual_init_report,
            "freeze_image_projection": args.freeze_image_projection,
            "evaluate_before_training": args.evaluate_before_training,
            "query_mode": "post_image_plus_disaster_text",
            "pretrained_source": pretrained_source,
            "args": vars(args),
        },
    )
    print(f"Best checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
