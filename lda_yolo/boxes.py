from __future__ import annotations

import torch
from torch import Tensor


def xywh_to_xyxy(boxes: Tensor) -> Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    half = size / 2
    return torch.cat((center - half, center + half), dim=-1)


def clip_boxes(boxes: Tensor, minimum: float = 0.0, maximum: float = 1.0) -> Tensor:
    return boxes.clamp(minimum, maximum)


def aligned_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    top_left = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=1)
    return intersection / (area1 + area2 - intersection + eps)


def pairwise_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=1)
    return intersection / (area1[:, None] + area2[None, :] - intersection + eps)


def nms(boxes: Tensor, scores: Tensor, threshold: float) -> Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[Tensor] = []
    while order.numel():
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        overlap = pairwise_iou(boxes[current].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][overlap <= threshold]
    return torch.stack(keep)


def decode_outputs(
    outputs: list[Tensor],
    strides: tuple[int, int, int],
    image_size: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
    pre_nms_topk: int = 3000,
) -> list[Tensor]:
    """Decode raw heads into normalized [x1,y1,x2,y2,score,class] detections."""

    batch_size = outputs[0].shape[0]
    per_image: list[list[Tensor]] = [[] for _ in range(batch_size)]
    for output, stride in zip(outputs, strides):
        batch, channels, height, width = output.shape
        prediction = output.permute(0, 2, 3, 1).contiguous()
        yy, xx = torch.meshgrid(
            torch.arange(height, device=output.device),
            torch.arange(width, device=output.device),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1).to(output.dtype)
        centers = (prediction[..., :2].sigmoid() * 2.0 - 0.5 + grid) * stride / image_size
        sizes = prediction[..., 2:4].clamp(-4.0, 4.0).exp() * stride / image_size
        boxes = xywh_to_xyxy(torch.cat((centers, sizes), dim=-1)).clamp(0.0, 1.0)
        objectness = prediction[..., 4].sigmoid()
        class_probability, class_id = prediction[..., 5:].sigmoid().max(dim=-1)
        scores = objectness * class_probability
        for batch_index in range(batch):
            mask = scores[batch_index] >= conf_threshold
            if not mask.any():
                continue
            selected_boxes = boxes[batch_index][mask]
            selected_scores = scores[batch_index][mask]
            selected_classes = class_id[batch_index][mask]
            if selected_scores.numel() > pre_nms_topk:
                top = selected_scores.topk(pre_nms_topk).indices
                selected_boxes = selected_boxes[top]
                selected_scores = selected_scores[top]
                selected_classes = selected_classes[top]
            per_image[batch_index].append(
                torch.cat(
                    (
                        selected_boxes,
                        selected_scores[:, None],
                        selected_classes[:, None].to(selected_boxes.dtype),
                    ),
                    dim=1,
                )
            )

    results: list[Tensor] = []
    for candidates in per_image:
        if not candidates:
            results.append(outputs[0].new_zeros((0, 6)))
            continue
        detections = torch.cat(candidates, dim=0)
        kept: list[Tensor] = []
        for class_id in detections[:, 5].unique():
            class_mask = detections[:, 5] == class_id
            class_detections = detections[class_mask]
            indices = nms(class_detections[:, :4], class_detections[:, 4], iou_threshold)
            kept.append(class_detections[indices])
        merged = torch.cat(kept, dim=0)
        merged = merged[merged[:, 4].argsort(descending=True)[:max_detections]]
        results.append(merged)
    return results

