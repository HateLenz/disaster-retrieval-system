from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def build_clip_transform(
    image_size: int = 224,
    train: bool = False,
) -> transforms.Compose:
    ops: list[transforms.Transform] = []
    if train:
        ops.extend(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    else:
        ops.extend(
            [
                transforms.Resize(
                    image_size,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
            ]
        )

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_IMAGE_MEAN, std=CLIP_IMAGE_STD),
        ]
    )
    return transforms.Compose(ops)


__all__ = [
    "CLIP_IMAGE_MEAN",
    "CLIP_IMAGE_STD",
    "build_clip_transform",
]
