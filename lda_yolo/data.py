from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetConfig:
    yaml_path: Path
    root: Path
    train: Path
    val: Path
    names: list[str]


def load_dataset_config(path: str | Path) -> DatasetConfig:
    yaml_path = Path(path).resolve()
    content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    declared_root = Path(content.get("path", "."))
    root = declared_root if declared_root.is_absolute() else (yaml_path.parent / declared_root).resolve()
    names_value = content["names"]
    if isinstance(names_value, dict):
        names = [names_value[index] for index in sorted(names_value)]
    else:
        names = list(names_value)

    def resolve_split(value: str) -> Path:
        split = Path(value)
        return split if split.is_absolute() else (root / split).resolve()

    return DatasetConfig(
        yaml_path=yaml_path,
        root=root,
        train=resolve_split(content["train"]),
        val=resolve_split(content["val"]),
        names=names,
    )


def find_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() == ".txt":
        return [Path(line.strip()).resolve() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not source.exists():
        raise FileNotFoundError(f"Image split not found: {source}")
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lower = [part.lower() for part in parts]
    if "images" in lower:
        index = len(lower) - 1 - lower[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def read_labels(path: Path) -> Tensor:
    rows: list[list[float]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            values = line.strip().split()
            if len(values) >= 5:
                rows.append([float(value) for value in values[:5]])
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)


def pil_to_tensor(image: Image.Image) -> Tensor:
    byte_tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = byte_tensor.reshape(image.height, image.width, 3).permute(2, 0, 1)
    return tensor.float().div_(255.0)


def letterbox(image: Image.Image, labels: Tensor, size: int) -> tuple[Image.Image, Tensor]:
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    pad_x, pad_y = (size - resized_width) // 2, (size - resized_height) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    if labels.numel():
        adjusted = labels.clone()
        adjusted[:, 1] = (labels[:, 1] * width * scale + pad_x) / size
        adjusted[:, 2] = (labels[:, 2] * height * scale + pad_y) / size
        adjusted[:, 3] = labels[:, 3] * width * scale / size
        adjusted[:, 4] = labels[:, 4] * height * scale / size
        labels = adjusted
    return canvas, labels


def random_scale(image: Image.Image, labels: Tensor, size: int, factor: float) -> tuple[Image.Image, Tensor]:
    scaled_size = max(16, round(size * factor))
    scaled = image.resize((scaled_size, scaled_size), Image.Resampling.BILINEAR)
    if scaled_size >= size:
        offset_x = random.randint(0, scaled_size - size)
        offset_y = random.randint(0, scaled_size - size)
        output = scaled.crop((offset_x, offset_y, offset_x + size, offset_y + size))
        shift_x, shift_y = -offset_x, -offset_y
    else:
        offset_x = random.randint(0, size - scaled_size)
        offset_y = random.randint(0, size - scaled_size)
        output = Image.new("RGB", (size, size), (114, 114, 114))
        output.paste(scaled, (offset_x, offset_y))
        shift_x, shift_y = offset_x, offset_y
    if not labels.numel():
        return output, labels
    xyxy = torch.empty_like(labels[:, 1:5])
    xyxy[:, 0] = (labels[:, 1] - labels[:, 3] / 2) * size * factor + shift_x
    xyxy[:, 1] = (labels[:, 2] - labels[:, 4] / 2) * size * factor + shift_y
    xyxy[:, 2] = (labels[:, 1] + labels[:, 3] / 2) * size * factor + shift_x
    xyxy[:, 3] = (labels[:, 2] + labels[:, 4] / 2) * size * factor + shift_y
    xyxy.clamp_(0, size)
    widths, heights = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
    keep = (widths > 2) & (heights > 2)
    result = labels[keep].clone()
    xyxy = xyxy[keep]
    if result.numel():
        result[:, 1] = (xyxy[:, 0] + xyxy[:, 2]) / (2 * size)
        result[:, 2] = (xyxy[:, 1] + xyxy[:, 3]) / (2 * size)
        result[:, 3] = (xyxy[:, 2] - xyxy[:, 0]) / size
        result[:, 4] = (xyxy[:, 3] - xyxy[:, 1]) / size
    return output, result


class YoloDetectionDataset(Dataset[tuple[Tensor, Tensor, str]]):
    def __init__(
        self,
        source: Path,
        image_size: int = 640,
        augment: bool = False,
        mosaic_probability: float = 0.5,
        mixup_probability: float = 0.1,
    ) -> None:
        self.images = find_images(source)
        if not self.images:
            raise RuntimeError(f"No images found under {source}")
        self.image_size = image_size
        self.augment = augment
        self.mosaic_probability = mosaic_probability
        self.mixup_probability = mixup_probability

    def __len__(self) -> int:
        return len(self.images)

    def _load(self, index: int) -> tuple[Image.Image, Tensor]:
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        labels = read_labels(infer_label_path(path))
        return letterbox(image, labels, self.image_size)

    def _mosaic(self, index: int) -> tuple[Image.Image, Tensor]:
        indices = [index] + random.choices(range(len(self.images)), k=3)
        half = self.image_size // 2
        canvas = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
        merged: list[Tensor] = []
        for slot, source_index in enumerate(indices):
            image, labels = self._load(source_index)
            image = image.resize((half, half), Image.Resampling.BILINEAR)
            column, row = slot % 2, slot // 2
            canvas.paste(image, (column * half, row * half))
            if labels.numel():
                labels = labels.clone()
                labels[:, 1] = labels[:, 1] / 2 + column / 2
                labels[:, 2] = labels[:, 2] / 2 + row / 2
                labels[:, 3:5] /= 2
                merged.append(labels)
        return canvas, torch.cat(merged, dim=0) if merged else torch.zeros((0, 5))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        if self.augment and random.random() < self.mosaic_probability:
            image, labels = self._mosaic(index)
        else:
            image, labels = self._load(index)

        if self.augment and random.random() < self.mixup_probability:
            second, second_labels = self._load(random.randrange(len(self.images)))
            image = Image.blend(image, second, alpha=0.5)
            labels = torch.cat((labels, second_labels), dim=0)

        if self.augment:
            image, labels = random_scale(
                image, labels, self.image_size, random.uniform(0.8, 1.2)
            )
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if labels.numel():
                    labels[:, 1] = 1.0 - labels[:, 1]
            if random.random() < 0.1:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                if labels.numel():
                    labels[:, 2] = 1.0 - labels[:, 2]
        return pil_to_tensor(image), labels, str(self.images[index])


def detection_collate(
    batch: Sequence[tuple[Tensor, Tensor, str]],
) -> tuple[Tensor, list[Tensor], list[str]]:
    images, labels, paths = zip(*batch)
    return torch.stack(images), list(labels), list(paths)

