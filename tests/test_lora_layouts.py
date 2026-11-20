import unittest

import torch

import Krea2_SVD_LoRA_Merger as merger


class FactorLayoutTests(unittest.TestCase):
    def test_linear_layout_round_trip(self):
        a = torch.randn(3, 5)
        b = torch.randn(7, 3)
        layout = merger.describe_factor_layout(a.shape, b.shape)
        af, bf = merger.flatten_factors(a, b, layout)
        ar, br = merger.restore_factors(af, bf, layout, 3)
        self.assertEqual(tuple(ar.shape), (3, 5))
        self.assertEqual(tuple(br.shape), (7, 3))
        self.assertTrue(torch.allclose(br @ ar, b @ a))

    def test_conv2d_down_kernel_layout(self):
        a = torch.randn(3, 4, 3, 3)
        b = torch.randn(6, 3, 1, 1)
        layout = merger.describe_factor_layout(a.shape, b.shape)
        af, bf = merger.flatten_factors(a, b, layout)
        ar, br = merger.restore_factors(af, bf, layout, 3)
        self.assertEqual(tuple(af.shape), (3, 36))
        self.assertEqual(tuple(bf.shape), (6, 3))
        self.assertEqual(tuple(ar.shape), tuple(a.shape))
        self.assertEqual(tuple(br.shape), tuple(b.shape))
        self.assertTrue(torch.allclose(bf @ af, b.reshape(6, 3) @ a.reshape(3, 36)))

    def test_conv2d_up_kernel_layout(self):
        a = torch.randn(3, 4, 1, 1)
        b = torch.randn(6, 3, 3, 3)
        layout = merger.describe_factor_layout(a.shape, b.shape)
        af, bf = merger.flatten_factors(a, b, layout)
        ar, br = merger.restore_factors(af, bf, layout, 3)
        self.assertEqual(tuple(af.shape), (3, 4))
        self.assertEqual(tuple(bf.shape), (54, 3))
        self.assertEqual(tuple(ar.shape), tuple(a.shape))
        self.assertEqual(tuple(br.shape), tuple(b.shape))
        expected = b.permute(0, 2, 3, 1).reshape(54, 3) @ a.reshape(3, 4)
        self.assertTrue(torch.allclose(bf @ af, expected))

    def test_rejects_two_spatial_kernels(self):
        with self.assertRaises(ValueError):
            merger.describe_factor_layout((3, 4, 3, 3), (6, 3, 3, 3))

    def test_kohya_and_peft_key_variants(self):
        self.assertEqual(merger.parse_key("layer.lora_down.weight")[:2], ("A", "layer"))
        self.assertEqual(merger.parse_key("layer.lora_up.weight")[:2], ("B", "layer"))
        self.assertEqual(merger.parse_key("layer.lora_A.default.weight")[:2], ("A", "layer"))
        self.assertEqual(merger.parse_key("layer.lora_B.custom.weight")[:2], ("B", "layer"))
        self.assertEqual(merger.parse_key("layer.lora_A.weight")[:2], ("A", "layer"))


if __name__ == "__main__":
    unittest.main()
