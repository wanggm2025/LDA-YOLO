from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from lda_yolo.boxes import decode_outputs
from lda_yolo.data import letterbox, pil_to_tensor
from lda_yolo.model import build_model


COLORS = [
    (239, 71, 111),
    (17, 138, 178),
    (6, 214, 160),
    (255, 209, 102),
    (131, 56, 236),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LDA-YOLO inference on one image")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/predict/result.jpg"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 640))
    names = checkpoint.get("names", [str(i) for i in range(checkpoint["num_classes"])])
    model = build_model(int(checkpoint["num_classes"]), int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    original = Image.open(args.source).convert("RGB")
    prepared, _ = letterbox(original, torch.zeros((0, 5)), image_size)
    tensor = pil_to_tensor(prepared).unsqueeze(0).to(device)
    with torch.no_grad():
        detections = decode_outputs(model(tensor), model.strides, image_size, args.conf, args.iou)[0].cpu()

    draw = ImageDraw.Draw(prepared)
    font = ImageFont.load_default()
    for x1, y1, x2, y2, score, class_id in detections.tolist():
        class_index = int(class_id)
        color = COLORS[class_index % len(COLORS)]
        box = (x1 * image_size, y1 * image_size, x2 * image_size, y2 * image_size)
        draw.rectangle(box, outline=color, width=2)
        label = f"{names[class_index]} {score:.2f}"
        label_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(label_box, fill=color)
        draw.text((box[0], box[1]), label, fill="white", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.save(args.output)
    print(f"saved {args.output} detections={len(detections)}")


if __name__ == "__main__":
    main()

