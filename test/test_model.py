import unittest
import torch
from model import ViTVAE

class TestModel(unittest.TestCase):

    def test_backward(self):
        model = ViTVAE(2, 128, 512, 3, 8, 512, dropout=0.1, emb_dropout=0.1)
        xs = torch.ones(2, 3, 32, 32)

        losses, vss = model(xs)
        losses["loss"].backward()

        for name, params in model.named_parameters():
            self.assertIsNotNone(params.grad)
