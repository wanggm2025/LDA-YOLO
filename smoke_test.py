from __future__ import annotations

import torch

from lda_yolo.boxes import decode_outputs
from lda_yolo.loss import DetectionLoss
from lda_yolo.model import build_model
from lda_yolo.modules import haar_dwt, haar_idwt


def main() -> None:
    torch.manual_seed(0)
    model = build_model(num_classes=3, base_channels=8)
    model.train()
    images = torch.randn(2, 3, 96, 96)
    targets = [
        torch.tensor([[0.0, 0.35, 0.40, 0.12, 0.10], [2.0, 0.72, 0.67, 0.20, 0.18]]),
        torch.tensor([[1.0, 0.50, 0.50, 0.08, 0.09]]),
    ]
    outputs = model(images)
    assert [tuple(output.shape[-2:]) for output in outputs] == [(24, 24), (12, 12), (6, 6)]
    criterion = DetectionLoss(3, model.strides)
    loss = criterion(outputs, targets, image_size=96)
    assert torch.isfinite(loss.total)
    loss.total.backward()

    wave = torch.randn(1, 4, 31, 29)
    subbands, size = haar_dwt(wave)
    restored = haar_idwt(subbands, size)
    assert torch.allclose(wave, restored, atol=1e-5)

    model.eval()
    with torch.no_grad():
        predictions = decode_outputs(model(images[:1]), model.strides, 96, conf_threshold=0.0)
    assert len(predictions) == 1 and predictions[0].shape[1] == 6
    print(
        f"smoke test passed: params={model.parameter_count():,} "
        f"loss={float(loss.total.detach()):.4f} detections={len(predictions[0])}"
    )


if __name__ == "__main__":
    main()
