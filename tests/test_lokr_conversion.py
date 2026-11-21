import unittest

import torch

from lokr_to_svd_lora import (
    convert_lokr_state_dict,
    inspect_lokr_header,
    is_lokr_state_keys,
)


class LoKrConversionTests(unittest.TestCase):
    def test_full_rank_conversion_reconstructs_kronecker_product(self):
        torch.manual_seed(7)
        w1 = torch.randn(2, 2)
        w2 = torch.randn(3, 2)
        source = {
            "layer.lokr_w1": w1,
            "layer.lokr_w2": w2,
            "layer.alpha": torch.tensor(1.0),
        }

        converted = convert_lokr_state_dict(source, torch, target_rank=4)
        A = converted["layer.lora_down.weight"]
        B = converted["layer.lora_up.weight"]

        self.assertEqual(tuple(A.shape), (4, 4))
        self.assertEqual(tuple(B.shape), (6, 4))
        self.assertTrue(torch.allclose(B @ A, torch.kron(w1, w2), atol=1e-5, rtol=1e-5))
        self.assertEqual(float(converted["layer.alpha"]), 4.0)

    def test_truncated_conversion_has_requested_rank(self):
        torch.manual_seed(11)
        source = {
            "layer.lokr_w1": torch.randn(2, 2),
            "layer.lokr_w2": torch.randn(5, 4),
        }
        converted = convert_lokr_state_dict(source, torch, target_rank=2)
        self.assertEqual(tuple(converted["layer.lora_down.weight"].shape), (2, 8))
        self.assertEqual(tuple(converted["layer.lora_up.weight"].shape), (10, 2))
        self.assertEqual(float(converted["layer.alpha"]), 2.0)

    def test_header_inspection_reports_product_dimensions(self):
        header = {
            "layer.lokr_w1": {"shape": (2, 2), "dtype": "F32"},
            "layer.lokr_w2": {"shape": (5, 4), "dtype": "BF16"},
            "layer.alpha": {"shape": (), "dtype": "F32"},
        }
        info = inspect_lokr_header("sample.safetensors", {}, header)
        self.assertEqual(info["format"], "lokr")
        self.assertEqual(info["modules"]["layer"]["A"]["shape"], (1, 8))
        self.assertEqual(info["modules"]["layer"]["B"]["shape"], (10, 1))
        self.assertEqual(info["module_count"], 1)

    def test_lokr_detection(self):
        self.assertTrue(is_lokr_state_keys(["x.lokr_w1", "x.lokr_w2"]))
        self.assertFalse(is_lokr_state_keys(["x.lora_down.weight", "x.lora_up.weight", "x.alpha"]))

    def test_decomposed_or_tucker_layout_is_rejected_explicitly(self):
        source = {
            "layer.lokr_w1_a": torch.randn(2, 1),
            "layer.lokr_w1_b": torch.randn(1, 2),
            "layer.lokr_w2": torch.randn(3, 3),
        }
        with self.assertRaisesRegex(ValueError, "decomposed/Tucker"):
            convert_lokr_state_dict(source, torch, target_rank=2)

    def test_incomplete_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete LoKr pair"):
            convert_lokr_state_dict({"layer.lokr_w1": torch.randn(2, 2)}, torch, 2)


if __name__ == "__main__":
    unittest.main()
