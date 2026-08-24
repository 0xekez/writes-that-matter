from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def function_arguments(path: Path, class_name: str, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == function_name:
                    return {arg.arg for arg in child.args.args + child.args.kwonlyargs}
    raise AssertionError(f"{class_name}.{function_name} not found in {path}")


class SourcePolicyTests(unittest.TestCase):
    def test_settled_sm100_knobs_are_not_arguments(self):
        arguments = function_arguments(
            ROOT / "lcgemm/kernels/sm100.py", "LcGemmSm100", "__init__"
        )
        self.assertTrue(
            arguments.isdisjoint({"ablate_chain", "plane_chunk", "plane_warps"})
        )

    def test_rank_split_has_only_identity_inputs(self):
        arguments = function_arguments(
            ROOT / "lcgemm/kernels/down_rank_split.py",
            "LcGemmDownRankSplit",
            "__init__",
        )
        self.assertEqual(arguments, {"self", "scheme", "mnk"})

    def test_deployment_plan_has_no_device_or_dtype_flags(self):
        for path, class_name in (
            (ROOT / "lcgemm/seams/gate_up.py", "GateUpPlan"),
            (ROOT / "lcgemm/seams/down.py", "DownPlan"),
        ):
            with self.subTest(class_name=class_name):
                arguments = function_arguments(path, class_name, "__init__")
                self.assertTrue(arguments.isdisjoint({"device", "dtype"}))

    def test_reproduction_cli_has_no_kernel_feature_flags(self):
        source = (ROOT / "lcgemm/bench/prefill.py").read_text()
        for flag in ("--scheme", "--split", "--persist", "--stage", "--boundary"):
            with self.subTest(flag=flag):
                self.assertNotIn(flag, source)

    def test_result_schema_keeps_untrimmed_samples(self):
        source = (ROOT / "lcgemm/bench/common.py").read_text()
        self.assertIn('"samples": samples', source)

    def test_experiment_modules_are_not_in_package(self):
        forbidden = {
            "attn.py",
            "oproj.py",
            "rank_split.py",
            "gate_planes.py",
            "adopt.py",
        }
        self.assertFalse(
            forbidden.intersection(
                path.name for path in (ROOT / "lcgemm").rglob("*.py")
            )
        )
