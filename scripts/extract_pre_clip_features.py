from __future__ import annotations

import argparse
import sys
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
from src.datasets.xbd_building_dataset import (
    XBDBuildingPatchDataset,
    collate_single_view,
)
from src.models.clip_visual_encoder import (
    CLIPVisualEncoder,
    DEFAULT_CLIP_MODEL_NAME,
    load_model_checkpoint,
)
from src.utils.device import get_device


DEFAULT_PATCH_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "clip_visual_baseline.pt"
DEFAULT_OUTPUT = REPO_ROOT / "indexes" / "pre_clip_features.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract gallery features for pre-disaster building patches.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_PATCH_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", type=str, default="hold")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--include-unclassified", action="store_true")
    parser.add_argument("--view", type=str, default="pre", choices=["pre", "post"])
    parser.add_argument("--model-name", type=str, default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--random-init", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> CLIPVisualEncoder:
    if args.checkpoint.exists():
        model, _ = load_model_checkpoint(
            args.checkpoint,
            map_location=device,
            model_name=args.model_name,
            local_files_only=args.local_files_only,
        )
        return model.to(device)

    model = CLIPVisualEncoder(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        freeze_backbone=True,
        pretrained=not args.random_init,
        local_files_only=args.local_files_only,
    )
    return model.to(device)


def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return autocast(enabled=enabled)
    return nullcontext()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    dataset = XBDBuildingPatchDataset(
        csv_path=args.csv,
        split=args.split,
        transform=build_clip_transform(image_size=args.image_size, train=False),
        include_unclassified=args.include_unclassified,
        view=args.view,
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

    model = load_model(args, device)
    model.eval()

    embeddings: list[np.ndarray] = []
    metadata_rows: list[dict[str, str | int]] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            with amp_context(device, amp_enabled):
                batch_embeddings = model.encode_image(images)
            embeddings.append(batch_embeddings.cpu().numpy().astype(np.float32))

            for index in range(len(batch["positive_id"])):
                metadata_rows.append(
                    {
                        "positive_id": batch["positive_id"][index],
                        "building_id": batch["building_id"][index],
                        "building_uid": batch["building_uid"][index],
                        "tile_id": batch["tile_id"][index],
                        "disaster": batch["disaster"][index],
                        "disaster_type": batch["disaster_type"][index],
                        "damage_label": batch["damage_label"][index],
                        "damage_id": int(batch["damage_id"][index].item()),
                        "split": batch["split"][index],
                        "image_path": batch["image_path"][index],
                        "view": args.view,
                    }
                )

    feature_matrix = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, feature_matrix)

    metadata_path = args.output.with_suffix(".csv")
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

    print(f"Features saved to: {args.output}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Feature matrix shape: {feature_matrix.shape}")


if __name__ == "__main__":
    main()
