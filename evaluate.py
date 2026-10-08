from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lda_yolo.boxes import decode_outputs
from lda_yolo.data import YoloDetectionDataset, detection_collate, load_dataset_config
from lda_yolo.metrics import detection_metrics
from lda_yolo.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LDA-YOLO with COCO-style mAP")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    config = load_dataset_config(args.data)
    image_size = int(checkpoint.get("image_size", 640))
    model = build_model(int(checkpoint["num_classes"]), int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = YoloDetectionDataset(config.val, image_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch, collate_fn=detection_collate)
    predictions, targets = [], []
    with torch.no_grad():
        for images, batch_targets, _ in loader:
            outputs = model(images.to(device))
            predictions.extend(decode_outputs(outputs, model.strides, image_size, args.conf, args.iou))
            targets.extend(target.to(device) for target in batch_targets)
    metrics = detection_metrics(predictions, targets, len(config.names))
    print(" ".join(f"{name}={value:.4f}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()

