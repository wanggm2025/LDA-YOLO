from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lda_yolo.data import YoloDetectionDataset, detection_collate, load_dataset_config
from lda_yolo.loss import DetectionLoss
from lda_yolo.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the runnable LDA-YOLO reproduction")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML file")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--final-lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/train/lda_yolo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


@torch.no_grad()
def validation_loss(model, loader, criterion, device, image_size: int) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for images, targets, _ in loader:
        images = images.to(device)
        result = criterion(model(images), targets, image_size)
        total += float(result.total)
        batches += 1
    return total / max(batches, 1)


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, names: list[str], value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": value,
            "num_classes": len(names),
            "names": names,
            "base_channels": args.base_channels,
            "image_size": args.img_size,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    config = load_dataset_config(args.data)
    train_dataset = YoloDetectionDataset(config.train, args.img_size, augment=True)
    val_dataset = YoloDetectionDataset(config.val, args.img_size, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
    )

    model = build_model(len(config.names), args.base_channels).to(device)
    criterion = DetectionLoss(len(config.names), model.strides)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.final_lr
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_loss = 0, math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("val_loss", math.inf))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "arguments.json").write_text(
        json.dumps(vars(args), default=str, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"device={device} train={len(train_dataset)} val={len(val_dataset)} "
        f"classes={len(config.names)} parameters={model.parameter_count():,}"
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {"loss": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
        for batch_index, (images, targets, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                result = criterion(model(images), targets, args.img_size)
            scaler.scale(result.total).backward()
            scaler.step(optimizer)
            scaler.update()
            values = result.scalars()
            for key in running:
                running[key] += values[key]
            if batch_index == 1 or batch_index % 10 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} batch {batch_index}/{len(train_loader)} "
                    f"loss={values['loss']:.4f} positives={result.positives}"
                )
        scheduler.step()
        val_loss = validation_loss(model, val_loader, criterion, device, args.img_size)
        denominator = max(len(train_loader), 1)
        summary = " ".join(f"{key}={value / denominator:.4f}" for key, value in running.items())
        print(f"epoch {epoch + 1}: {summary} val_loss={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f}")
        save_checkpoint(args.output / "last.pt", model, optimizer, epoch, args, config.names, val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(args.output / "best.pt", model, optimizer, epoch, args, config.names, val_loss)


if __name__ == "__main__":
    main()

