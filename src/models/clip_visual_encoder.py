from __future__ import annotations

import re
import os
import pathlib as pathlib_module
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextConfig, CLIPTextModel, CLIPVisionConfig, CLIPVisionModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIP_REPO_ID = "openai/clip-vit-base-patch32"
DEFAULT_LOCAL_CLIP_MODEL_DIR = REPO_ROOT / "pretrained" / DEFAULT_CLIP_REPO_ID.replace("/", "-")
DEFAULT_CLIP_MODEL_NAME = str(DEFAULT_LOCAL_CLIP_MODEL_DIR.relative_to(REPO_ROOT))


def resolve_clip_model_source(
    model_name: str | Path,
    local_files_only: bool = False,
) -> tuple[str, bool]:
    raw_model_name = str(model_name)
    if not raw_model_name:
        return raw_model_name, local_files_only

    candidate_paths: list[Path] = []
    raw_path = Path(raw_model_name)
    if raw_path.is_absolute():
        candidate_paths.append(raw_path)
    else:
        candidate_paths.append((REPO_ROOT / raw_path).resolve())
    candidate_paths.append((REPO_ROOT / "pretrained" / raw_model_name.replace("\\", "-").replace("/", "-")).resolve())

    seen: set[str] = set()
    for candidate in candidate_paths:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.is_dir():
            return str(candidate), True if local_files_only or candidate.is_relative_to(REPO_ROOT / "pretrained") else local_files_only

    return raw_model_name, local_files_only


def normalize_clip_model_name(model_name: str | Path) -> str:
    resolved_model_name, _ = resolve_clip_model_source(model_name=model_name, local_files_only=False)
    resolved_path = Path(resolved_model_name)
    if resolved_path.is_absolute() and resolved_path.exists():
        try:
            return str(resolved_path.relative_to(REPO_ROOT))
        except ValueError:
            return str(resolved_path)
    return str(model_name)


def infer_patch_size(model_name: str) -> int:
    match = re.search(r"patch(\d+)", model_name)
    if match is None:
        return 32
    return int(match.group(1))


def build_clip_config(model_name: str) -> CLIPVisionConfig:
    patch_size = infer_patch_size(model_name)
    return CLIPVisionConfig(
        image_size=224,
        patch_size=patch_size,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        projection_dim=512,
    )


def build_clip_text_config() -> CLIPTextConfig:
    return CLIPTextConfig(
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        projection_dim=512,
    )


def load_clip_vision_backbone(
    model_name: str,
    pretrained: bool = True,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> CLIPVisionModel:
    resolved_model_name, resolved_local_files_only = resolve_clip_model_source(
        model_name=model_name,
        local_files_only=local_files_only,
    )

    if pretrained:
        return CLIPVisionModel.from_pretrained(
            resolved_model_name,
            local_files_only=resolved_local_files_only,
            cache_dir=cache_dir,
        )

    try:
        config = CLIPVisionConfig.from_pretrained(
            resolved_model_name,
            local_files_only=resolved_local_files_only,
            cache_dir=cache_dir,
        )
    except Exception:
        config = build_clip_config(resolved_model_name)
    return CLIPVisionModel(config)


def load_clip_text_backbone(
    model_name: str,
    pretrained: bool = True,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> CLIPTextModel:
    resolved_model_name, resolved_local_files_only = resolve_clip_model_source(
        model_name=model_name,
        local_files_only=local_files_only,
    )

    if pretrained:
        return CLIPTextModel.from_pretrained(
            resolved_model_name,
            local_files_only=resolved_local_files_only,
            cache_dir=cache_dir,
        )

    try:
        config = CLIPTextConfig.from_pretrained(
            resolved_model_name,
            local_files_only=resolved_local_files_only,
            cache_dir=cache_dir,
        )
    except Exception:
        config = build_clip_text_config()
    return CLIPTextModel(config)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class ResidualTextFusion(nn.Module):
    def __init__(self, embedding_dim: int, initial_gate_logit: float = -2.0) -> None:
        super().__init__()
        self.text_delta = nn.Linear(embedding_dim, embedding_dim)
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate_logit)))
        nn.init.zeros_(self.text_delta.weight)
        nn.init.zeros_(self.text_delta.bias)

    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.gate_logit)
        return image_embeddings + scale * self.text_delta(text_embeddings)


class CLIPVisualEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL_NAME,
        embedding_dim: int = 256,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = normalize_clip_model_name(model_name)
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.local_files_only = local_files_only
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.freeze_backbone = freeze_backbone
        self.dropout = dropout

        self.backbone = load_clip_vision_backbone(
            model_name=self.model_name,
            pretrained=pretrained,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.projection = ProjectionHead(
            input_dim=hidden_size,
            output_dim=embedding_dim,
            dropout=dropout,
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values)

        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        embeddings = self.projection(pooled)
        return F.normalize(embeddings, dim=-1)

    def forward(
        self,
        pre_pixel_values: torch.Tensor | None = None,
        post_pixel_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        if pre_pixel_values is not None:
            outputs["pre_embeddings"] = self.encode_image(pre_pixel_values)
        if post_pixel_values is not None:
            outputs["post_embeddings"] = self.encode_image(post_pixel_values)
        return outputs

    def get_config(self) -> dict[str, Any]:
        return {
            "architecture": "clip_visual_encoder",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "freeze_backbone": self.freeze_backbone,
            "pretrained": self.pretrained,
            "local_files_only": self.local_files_only,
            "cache_dir": self.cache_dir,
            "dropout": self.dropout,
        }


class CLIPSemanticEncoder(CLIPVisualEncoder):
    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL_NAME,
        embedding_dim: int = 256,
        freeze_backbone: bool = True,
        freeze_text_backbone: bool = True,
        pretrained: bool = True,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
        dropout: float = 0.1,
        query_fusion_mode: str = "residual_gate",
    ) -> None:
        super().__init__(
            model_name=model_name,
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            dropout=dropout,
        )
        self.freeze_text_backbone = freeze_text_backbone
        self.query_fusion_mode = query_fusion_mode
        self.text_backbone = load_clip_text_backbone(
            model_name=self.model_name,
            pretrained=pretrained,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        text_hidden_size = int(self.text_backbone.config.hidden_size)
        self.text_projection = ProjectionHead(
            input_dim=text_hidden_size,
            output_dim=embedding_dim,
            dropout=dropout,
        )
        if query_fusion_mode == "concat_mlp":
            self.query_fusion = ProjectionHead(
                input_dim=embedding_dim * 2,
                output_dim=embedding_dim,
                dropout=dropout,
            )
        elif query_fusion_mode == "residual_gate":
            self.query_fusion = ResidualTextFusion(embedding_dim=embedding_dim)
        else:
            raise ValueError("query_fusion_mode must be one of {'residual_gate', 'concat_mlp'}")

        if freeze_text_backbone:
            for parameter in self.text_backbone.parameters():
                parameter.requires_grad = False

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.freeze_text_backbone:
            self.text_backbone.eval()
            with torch.no_grad():
                outputs = self.text_backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            outputs = self.text_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, -1]
        embeddings = self.text_projection(pooled)
        return F.normalize(embeddings, dim=-1)

    def fuse_query_embeddings(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError("image_embeddings and text_embeddings must have the same shape")
        if self.query_fusion_mode == "concat_mlp":
            fused = self.query_fusion(torch.cat([image_embeddings, text_embeddings], dim=-1))
        else:
            fused = self.query_fusion(image_embeddings, text_embeddings)
        return F.normalize(fused, dim=-1)

    def encode_query(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_embeddings = self.encode_image(pixel_values)
        text_embeddings = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.fuse_query_embeddings(image_embeddings, text_embeddings)

    def forward(
        self,
        pre_pixel_values: torch.Tensor | None = None,
        post_pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(
            pre_pixel_values=pre_pixel_values,
            post_pixel_values=post_pixel_values,
        )
        if input_ids is not None:
            outputs["text_embeddings"] = self.encode_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if "post_embeddings" in outputs and outputs["post_embeddings"].shape == outputs["text_embeddings"].shape:
                outputs["query_embeddings"] = self.fuse_query_embeddings(
                    outputs["post_embeddings"],
                    outputs["text_embeddings"],
                )
        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "architecture": "clip_semantic_encoder",
                "freeze_text_backbone": self.freeze_text_backbone,
                "query_fusion_mode": self.query_fusion_mode,
            }
        )
        return config


def build_model_from_config(config: dict[str, Any]) -> CLIPVisualEncoder:
    architecture = str(config.get("architecture", "clip_visual_encoder"))
    if architecture == "clip_semantic_encoder":
        return CLIPSemanticEncoder(
            model_name=str(config.get("model_name", DEFAULT_CLIP_MODEL_NAME)),
            embedding_dim=int(config.get("embedding_dim", 256)),
            freeze_backbone=bool(config.get("freeze_backbone", True)),
            freeze_text_backbone=bool(config.get("freeze_text_backbone", True)),
            pretrained=bool(config.get("pretrained", True)),
            local_files_only=bool(config.get("local_files_only", False)),
            cache_dir=config.get("cache_dir"),
            dropout=float(config.get("dropout", 0.1)),
            query_fusion_mode=str(config.get("query_fusion_mode", "concat_mlp")),
        )

    return CLIPVisualEncoder(
        model_name=str(config.get("model_name", DEFAULT_CLIP_MODEL_NAME)),
        embedding_dim=int(config.get("embedding_dim", 256)),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
        pretrained=bool(config.get("pretrained", True)),
        local_files_only=bool(config.get("local_files_only", False)),
        cache_dir=config.get("cache_dir"),
        dropout=float(config.get("dropout", 0.1)),
    )


def load_model_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    **config_overrides: Any,
) -> tuple[CLIPVisualEncoder, dict[str, Any]]:
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=map_location)
    model_config = dict(checkpoint.get("model_config", {}))
    model_config.update({key: value for key, value in config_overrides.items() if value is not None})
    model = build_model_from_config(model_config)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def load_torch_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except NotImplementedError as error:
        if os.name != "nt" or "PosixPath" not in str(error):
            raise
        original_posix_path = pathlib_module.PosixPath
        pathlib_module.PosixPath = pathlib_module.WindowsPath
        try:
            return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        finally:
            pathlib_module.PosixPath = original_posix_path


__all__ = [
    "CLIPSemanticEncoder",
    "CLIPVisualEncoder",
    "DEFAULT_CLIP_MODEL_NAME",
    "DEFAULT_CLIP_REPO_ID",
    "DEFAULT_LOCAL_CLIP_MODEL_DIR",
    "build_model_from_config",
    "load_clip_text_backbone",
    "load_clip_vision_backbone",
    "load_model_checkpoint",
    "load_torch_checkpoint",
    "normalize_clip_model_name",
    "resolve_clip_model_source",
]
