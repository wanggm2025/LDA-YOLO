from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .boxes import aligned_iou, xywh_to_xyxy


@dataclass
class LossResult:
    total: Tensor
    box: Tensor
    objectness: Tensor
    classification: Tensor
    positives: int

    def scalars(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "box": float(self.box.detach()),
            "obj": float(self.objectness.detach()),
            "cls": float(self.classification.detach()),
            "positives": float(self.positives),
        }


class DetectionLoss(nn.Module):
    """A compact anchor-free YOLO-style objective for the three LDA-YOLO heads."""

    def __init__(
        self,
        num_classes: int,
        strides: tuple[int, int, int] = (4, 8, 16),
        box_weight: float = 7.5,
        objectness_weight: float = 1.0,
        class_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.box_weight = box_weight
        self.objectness_weight = objectness_weight
        self.class_weight = class_weight

    def _choose_scale(self, target: Tensor, image_size: int) -> int:
        object_size = torch.sqrt(target[3] * target[4]).item() * image_size
        references = [stride * 8 for stride in self.strides]
        return min(range(len(references)), key=lambda index: abs(references[index] - object_size))

    def forward(self, outputs: list[Tensor], targets: list[Tensor], image_size: int) -> LossResult:
        device = outputs[0].device
        box_loss = outputs[0].new_tensor(0.0)
        objectness_loss = outputs[0].new_tensor(0.0)
        class_loss = outputs[0].new_tensor(0.0)
        positive_count = 0

        assignments: list[list[tuple[int, int, int, Tensor]]] = [[] for _ in outputs]
        for batch_index, image_targets in enumerate(targets):
            for target in image_targets.to(device):
                if target.numel() != 5:
                    continue
                scale_index = self._choose_scale(target, image_size)
                _, _, height, width = outputs[scale_index].shape
                x_index = min(int(target[1].item() * width), width - 1)
                y_index = min(int(target[2].item() * height), height - 1)
                assignments[scale_index].append((batch_index, y_index, x_index, target))

        for scale_index, (output, stride) in enumerate(zip(outputs, self.strides)):
            batch, _, height, width = output.shape
            objectness_target = torch.zeros((batch, height, width), device=device)
            class_target = torch.zeros((batch, height, width, self.num_classes), device=device)
            box_target = torch.zeros((batch, height, width, 4), device=device)
            positive = torch.zeros((batch, height, width), dtype=torch.bool, device=device)

            for batch_index, y_index, x_index, target in assignments[scale_index]:
                class_id = int(target[0].item())
                if 0 <= class_id < self.num_classes:
                    objectness_target[batch_index, y_index, x_index] = 1.0
                    class_target[batch_index, y_index, x_index, class_id] = 1.0
                    box_target[batch_index, y_index, x_index] = target[1:5]
                    positive[batch_index, y_index, x_index] = True

            prediction = output.permute(0, 2, 3, 1).contiguous()
            positive_weight = output.new_tensor(5.0)
            objectness_loss = objectness_loss + F.binary_cross_entropy_with_logits(
                prediction[..., 4], objectness_target, pos_weight=positive_weight
            )

            if positive.any():
                indices = positive.nonzero(as_tuple=False)
                raw_boxes = prediction[..., :4][positive]
                grid_xy = indices[:, [2, 1]].to(raw_boxes.dtype)
                centers = (raw_boxes[:, :2].sigmoid() * 2.0 - 0.5 + grid_xy)
                centers[:, 0] /= width
                centers[:, 1] /= height
                sizes = raw_boxes[:, 2:4].clamp(-4.0, 4.0).exp() * stride / image_size
                decoded = xywh_to_xyxy(torch.cat((centers, sizes), dim=1))
                expected = xywh_to_xyxy(box_target[positive])
                box_loss = box_loss + (1.0 - aligned_iou(decoded, expected)).mean()
                class_loss = class_loss + F.binary_cross_entropy_with_logits(
                    prediction[..., 5:][positive], class_target[positive]
                )
                positive_count += int(positive.sum())

        number_of_scales = len(outputs)
        objectness_loss = objectness_loss / number_of_scales
        if positive_count == 0:
            box_loss = box_loss + sum(output.sum() * 0.0 for output in outputs)
            class_loss = class_loss + sum(output.sum() * 0.0 for output in outputs)
        total = (
            self.box_weight * box_loss
            + self.objectness_weight * objectness_loss
            + self.class_weight * class_loss
        )
        return LossResult(total, box_loss, objectness_loss, class_loss, positive_count)

