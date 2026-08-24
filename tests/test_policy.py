from __future__ import annotations

import sys
import unittest

from lcgemm import qwen_policy
from lcgemm.scheme import load


class PolicyTests(unittest.TestCase):
    def test_scheme_files_are_self_contained_and_validated(self):
        gate = load("2x2_postsum_cse")
        down = load("2x2_locality_ordered")
        self.assertEqual(gate.shape, (2, 2, 2))
        self.assertEqual(down.shape, (2, 2, 2))
        self.assertEqual(gate.rank, 7)
        self.assertEqual(down.rank, 7)
        self.assertTrue(gate.has_postsum)
        self.assertFalse(down.has_postsum)

    def test_qwen_stock_policy_never_imports_patch(self):
        for tokens in (512, 1024):
            with self.subTest(tokens=tokens):
                sys.modules.pop("lcgemm.integrate.qwen_patch", None)
                self.assertEqual(qwen_policy.deployment_for(tokens), "stock")
                self.assertIsNone(qwen_policy.arm_selected_engine(tokens))
                self.assertNotIn("lcgemm.integrate.qwen_patch", sys.modules)

    def test_qwen_lc_policy_is_exact_shape(self):
        for tokens in (2048, 4096, 8192):
            with self.subTest(tokens=tokens):
                self.assertEqual(qwen_policy.deployment_for(tokens), "lc_chain")

    def test_qwen_policy_rejects_unvalidated_shapes(self):
        for tokens in (0, 513, 3072, 16384):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                qwen_policy.deployment_for(tokens)
