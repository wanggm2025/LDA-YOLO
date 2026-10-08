from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor

from .boxes import pairwise_iou, xywh_to_xyxy


def _average_precision(recall: Tensor, precision: Tensor) -> float:
    if recall.numel() == 0:
        return 0.0
    samples = torch.linspace(0, 1, 101, device=recall.device)
    values = []
    for sample in samples:
        valid = precision[recall >= sample]
        values.append(valid.max() if valid.numel() else precision.new_tensor(0.0))
    return float(torch.stack(values).mean())


def detection_metrics(
    predictions: list[Tensor],
    targets: list[Tensor],
    num_classes: int,
    iou_thresholds: Tensor | None = None,
) -> dict[str, float]:
    """Compute class-averaged precision, recall, mAP50 and mAP50:95."""

    thresholds = iou_thresholds if iou_thresholds is not None else torch.linspace(0.5, 0.95, 10)
    device = predictions[0].device if predictions else torch.device("cpu")
    thresholds = thresholds.to(device)
    all_ap = torch.zeros((num_classes, len(thresholds)), device=device)
    class_precision = torch.zeros(num_classes, device=device)
    class_recall = torch.zeros(num_classes, device=device)
    valid_classes = torch.zeros(num_classes, dtype=torch.bool, device=device)

    target_boxes_by_image: list[Tensor] = []
    target_classes_by_image: list[Tensor] = []
    for target in targets:
        target = target.to(device)
        target_boxes_by_image.append(xywh_to_xyxy(target[:, 1:5]) if target.numel() else target.new_zeros((0, 4)))
        target_classes_by_image.append(target[:, 0].long() if target.numel() else target.new_zeros((0,), dtype=torch.long))

    for class_id in range(num_classes):
        ground_truth: dict[int, Tensor] = {}
        total_ground_truth = 0
        candidates: list[tuple[float, int, Tensor]] = []
        for image_index, (prediction, boxes, classes) in enumerate(
            zip(predictions, target_boxes_by_image, target_classes_by_image)
        ):
            class_gt = boxes[classes == class_id]
            ground_truth[image_index] = class_gt
            total_ground_truth += len(class_gt)
            class_predictions = prediction[prediction[:, 5].long() == class_id]
            for row in class_predictions:
                candidates.append((float(row[4]), image_index, row[:4]))
        if total_ground_truth == 0:
            continue
        valid_classes[class_id] = True
        candidates.sort(key=lambda item: item[0], reverse=True)

        for threshold_index, threshold in enumerate(thresholds):
            matched = {
                image_index: torch.zeros(len(boxes), dtype=torch.bool, device=device)
                for image_index, boxes in ground_truth.items()
            }
            tp = torch.zeros(len(candidates), device=device)
            fp = torch.zeros(len(candidates), device=device)
            for prediction_index, (_, image_index, box) in enumerate(candidates):
                boxes = ground_truth[image_index]
                if boxes.numel() == 0:
                    fp[prediction_index] = 1
                    continue
                overlaps = pairwise_iou(box.unsqueeze(0), boxes)[0]
                overlaps[matched[image_index]] = -1
                best_iou, best_index = overlaps.max(dim=0)
                if best_iou >= threshold:
                    tp[prediction_index] = 1
                    matched[image_index][best_index] = True
                else:
                    fp[prediction_index] = 1
            tp_cumulative = tp.cumsum(0)
            fp_cumulative = fp.cumsum(0)
            recall = tp_cumulative / max(total_ground_truth, 1)
            precision = tp_cumulative / (tp_cumulative + fp_cumulative).clamp(min=1e-9)
            all_ap[class_id, threshold_index] = _average_precision(recall, precision)
            if threshold_index == 0 and len(candidates):
                class_precision[class_id] = precision[-1]
                class_recall[class_id] = recall[-1]

    if not valid_classes.any():
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}
    return {
        "precision": float(class_precision[valid_classes].mean()),
        "recall": float(class_recall[valid_classes].mean()),
        "map50": float(all_ap[valid_classes, 0].mean()),
        "map50_95": float(all_ap[valid_classes].mean()),
    }
