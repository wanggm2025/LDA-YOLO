from __future__ import annotations

import unittest

import torch

from lda_yolo.loss import DetectionLoss
from lda_yolo.model import build_model
from lda_yolo.modules import DDAM, DCFM, IC2f, SGConv, haar_dwt, haar_idwt


class ModuleTests(unittest.TestCase):
    def test_modules_preserve_shape(self) -> None:
        tensor = torch.randn(2, 16, 15, 17)
        for module in (SGConv(16), IC2f(16, 16), DCFM(16), DDAM(16)):
            with self.subTest(module=type(module).__name__):
                self.assertEqual(module(tensor).shape, tensor.shape)

    def test_haar_round_trip_for_odd_shape(self) -> None:
        tensor = torch.randn(2, 5, 17, 19)
        subbands, size = haar_dwt(tensor)
        restored = haar_idwt(subbands, size)
        self.assertTrue(torch.allclose(tensor, restored, atol=1e-5))

    def test_model_and_loss_backward(self) -> None:
        model = build_model(2, base_channels=8)
        image = torch.randn(1, 3, 64, 64)
        outputs = model(image)
        self.assertEqual([tuple(item.shape[-2:]) for item in outputs], [(16, 16), (8, 8), (4, 4)])
        target = [torch.tensor([[1.0, 0.5, 0.5, 0.15, 0.12]])]
        result = DetectionLoss(2, model.strides)(outputs, target, 64)
        result.total.backward()
        self.assertTrue(torch.isfinite(result.total))


if __name__ == "__main__":
    unittest.main()

