from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.retrieval_service import write_metadata_sqlite
from src.datasets.transforms import build_clip_transform
from src.datasets.xbd_building_dataset import XBDBuildingPatchDataset, collate_single_view
from src.models.clip_visual_encoder import DEFAULT_CLIP_MODEL_NAME, load_model_checkpoint
from src.retrieval.faiss_index import build_index, save_index
from src.utils.device import get_device


DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "xbd_building_patches.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "clip_visual_baseline_smoke_stage1_v2.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "indexes" / "api_hold_stage1_v2"
DEFAULT_SQLITE = REPO_ROOT / "db" / "retrieval_hold_stage1_v2.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the local hold pre-disaster retrieval database used by the FastAPI service.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    parser.add_argument("--split", type=str, default="hold")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--include-unclassified", action="store_true")
    parser.add_argument("--model-name", type=str, default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--no-local-files-only", dest="local_files_only", action="store_false")
    return parser.parse_args()


def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
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
        view="pre",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_single_view,
    )

    model, checkpoint = load_model_checkpoint(
        args.checkpoint,
        map_location=device,
        model_name=args.model_name,
        local_files_only=args.local_files_only,
    )
    model = model.to(device)
    model.eval()

    feature_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            with amp_context(device, amp_enabled):
                embeddings = model.encode_image(images)
            feature_blocks.append(embeddings.cpu().numpy().astype(np.float32))

    features = np.concatenate(feature_blocks, axis=0) if feature_blocks else np.empty((0, 0), dtype=np.float32)
    metadata = pd.DataFrame(dataset.records)
    metadata["image_path"] = metadata["pre_patch_path"]
    metadata["view"] = "pre"

    if len(metadata) != features.shape[0]:
        raise ValueError(f"Metadata rows ({len(metadata)}) do not match feature rows ({features.shape[0]})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.output_dir / "pre_features.npy"
    metadata_path = args.output_dir / "pre_metadata.csv"
    faiss_path = args.output_dir / "pre_features.faiss"

    np.save(features_path, features)
    metadata.to_csv(metadata_path, index=False)
    save_index(build_index(features), faiss_path)
    write_metadata_sqlite(metadata, args.sqlite)

    print(f"Features saved to: {features_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"FAISS index saved to: {faiss_path}")
    print(f"SQLite metadata saved to: {args.sqlite}")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")


if __name__ == "__main__":
    main()
