from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = [
    ("red_object", (220, 45, 65)),
    ("blue_object", (35, 105, 220)),
    ("yellow_object", (238, 188, 35)),
]


def generate_split(root: Path, split: str, count: int, size: int, rng: random.Random) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image_index in range(count):
        background = rng.randint(65, 130)
        image = Image.new("RGB", (size, size), (background, background + 5, background + 10))
        draw = ImageDraw.Draw(image)
        labels = []
        for _ in range(rng.randint(2, 6)):
            class_id = rng.randrange(len(CLASSES))
            _, color = CLASSES[class_id]
            width = rng.randint(max(5, size // 24), max(8, size // 7))
            height = rng.randint(max(5, size // 24), max(8, size // 7))
            x1 = rng.randint(0, size - width - 1)
            y1 = rng.randint(0, size - height - 1)
            x2, y2 = x1 + width, y1 + height
            if class_id == 1:
                draw.ellipse((x1, y1, x2, y2), fill=color)
            else:
                draw.rectangle((x1, y1, x2, y2), fill=color)
            center_x = (x1 + x2) / (2 * size)
            center_y = (y1 + y2) / (2 * size)
            labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width / size:.6f} {height / size:.6f}")
        image.save(image_dir / f"{image_index:04d}.jpg", quality=92)
        (label_dir / f"{image_index:04d}.txt").write_text("\n".join(labels), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic YOLO dataset")
    parser.add_argument("--output", type=Path, default=Path("demo_data"))
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--val", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    generate_split(args.output, "train", args.train, args.size, rng)
    generate_split(args.output, "val", args.val, args.size, rng)
    print(f"generated train={args.train} val={args.val} under {args.output}")


if __name__ == "__main__":
    main()

