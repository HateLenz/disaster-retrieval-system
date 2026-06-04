from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.transforms import build_clip_transform
from src.datasets.xbd_building_dataset import (
    XBDBuildingPatchDataset,
    collate_building_pairs,
)
from src.losses.contrastive import SymmetricInfoNCELoss
from src.models.clip_visual_encoder import CLIPVisualEncoder, DEFAULT_CLIP_MODEL_NAME
from src.retrieval.faiss_index import build_index, search_index
from src.retrieval.metrics import compute_grouped_metrics, compute_macro_metrics, compute_metrics
from src.utils.device import get_device, setup_torch_runtime


DEFAULT_PATCH_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "clip_visual_baseline.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CLIP-based building-level cross-temporal retrieval baseline.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_PATCH_CSV, help="Building patch CSV.")
    parser.add_argument("--train-split", type=str, default="tier3", help="Training split name.")
    parser.add_argument("--eval-split", type=str, default="hold", help="Validation split name.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Checkpoint path.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--amp", action="store_true", help="Enable AMP when CUDA is available.")
    parser.add_argument("--include-unclassified", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--random-init", action="store_true", help="Do not load pretrained CLIP weights.")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Train the CLIP backbone as well.")
    parser.add_argument(
        "--disable-balanced-sampling",
        action="store_true",
        help="Disable inverse-frequency disaster-type sampling for the training loader.",
    )
    parser.add_argument(
        "--loss-weighting",
        type=str,
        default="none",
        choices=("none", "inverse_freq"),
        help="Optional class-weighted InfoNCE using disaster-type inverse-frequency weights.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="macro_recall@1",
        choices=("macro_recall@1", "overall_recall@1"),
        help="Metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--train-log-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL path for per-epoch training/eval records.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional final experiment summary path.",
    )
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
        persistent_workers=num_workers,
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


def encode_eval_embeddings(
    model: CLIPVisualEncoder,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
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
            with amp_context(device, amp_enabled):
                outputs = model(pre_pixel_values=pre_images, post_pixel_values=post_images)

            gallery_embeddings.append(outputs["pre_embeddings"].cpu().numpy().astype(np.float32))
            query_embeddings.append(outputs["post_embeddings"].cpu().numpy().astype(np.float32))
            positive_ids.extend(batch["positive_id"])
            disaster_types.extend(batch["disaster_type"])

    gallery_matrix = np.concatenate(gallery_embeddings, axis=0) if gallery_embeddings else np.empty((0, 0), dtype=np.float32)
    query_matrix = np.concatenate(query_embeddings, axis=0) if query_embeddings else np.empty((0, 0), dtype=np.float32)
    return gallery_matrix, query_matrix, positive_ids, disaster_types


def evaluate_retrieval(
    model: CLIPVisualEncoder,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    top_k: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float]]:
    gallery_matrix, query_matrix, positive_ids, disaster_types = encode_eval_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        amp_enabled=amp_enabled,
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
    model: CLIPVisualEncoder,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    args: argparse.Namespace,
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
    setup_torch_runtime()
    set_seed(args.seed)

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

    model = CLIPVisualEncoder(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        freeze_backbone=not args.unfreeze_backbone,
        pretrained=not args.random_init,
        local_files_only=args.local_files_only,
    ).to(device)
    criterion = SymmetricInfoNCELoss(temperature=args.temperature)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = build_grad_scaler(enabled=amp_enabled)

    print(
        json.dumps(
            {
                "train_disaster_type_counts": train_class_counts,
                "train_sampling_weights": train_class_weights,
                "balanced_sampling": not args.disable_balanced_sampling,
                "loss_weighting": args.loss_weighting,
                "selection_metric": args.selection_metric,
            },
            ensure_ascii=False,
        )
    )

    best_recall = -1.0
    best_epoch = 0
    best_eval_metrics: dict[str, float] = {}
    best_eval_macro: dict[str, float] = {}
    best_eval_grouped: dict[str, dict[str, float]] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0

        for batch in train_loader:
            pre_images = batch["pre_image"].to(device, non_blocking=True)
            post_images = batch["post_image"].to(device, non_blocking=True)
            sample_weights = None
            if args.loss_weighting == "inverse_freq" and train_class_weights:
                sample_weights = torch.tensor(
                    [train_class_weights[str(disaster_type)] for disaster_type in batch["disaster_type"]],
                    dtype=torch.float32,
                    device=device,
                )

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp_enabled):
                outputs = model(pre_pixel_values=pre_images, post_pixel_values=post_images)
                loss_dict = criterion(
                    outputs["post_embeddings"],
                    outputs["pre_embeddings"],
                    sample_weights=sample_weights,
                )
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            running_acc += float(loss_dict["retrieval_acc"].item())
            num_batches += 1

        scheduler.step()
        train_metrics = {
            "loss": running_loss / max(num_batches, 1),
            "retrieval_acc": running_acc / max(num_batches, 1),
        }
        eval_metrics, eval_grouped, eval_macro = evaluate_retrieval(
            model=model,
            dataloader=eval_loader,
            device=device,
            amp_enabled=amp_enabled,
            top_k=args.top_k,
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
        print(
            json.dumps(
                epoch_payload,
                ensure_ascii=False,
            )
        )

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
            "args": vars(args),
        },
    )
    print(f"Best checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
