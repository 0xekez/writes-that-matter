from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from lcgemm.bench.headline import MODELS, TOKENS, build_headline


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/headline"


class HeadlineArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((RESULTS / "summary.json").read_text())

    def test_summary_recomputes_exactly_from_all_raw_arms(self):
        expected = build_headline(
            RESULTS / "raw",
            models=MODELS,
            tokens=TOKENS,
            rounds=self.summary["rounds"],
            runs=self.summary["cold_l2_passes_per_arm_per_round"],
        )
        self.assertEqual(expected, self.summary["headline"])

    def test_reference_checksums(self):
        rows = (RESULTS / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(rows), 162)
        for row in rows:
            expected, relative = row.split(maxsplit=1)
            path = RESULTS / relative.strip().lstrip("*")
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_headline_signs_and_round_counts(self):
        for model in MODELS:
            for tokens in TOKENS:
                case = self.summary["headline"][f"{model}_{tokens}"]
                values = case["speedup_percent"]["values"]
                self.assertEqual(len(values), 10)
                expected_positive = tokens >= 2048
                self.assertTrue(all((value > 0) == expected_positive for value in values))

    def test_paper_headline_points_match_summary(self):
        source = (ROOT / "tex/main.tex").read_text()
        origins = {"muse": "leftorigin", "qwen": "rightorigin"}
        positions = ("0.65", "2.15", "3.65", "5.15")
        for model in MODELS:
            for x, tokens in zip(positions, TOKENS):
                values = self.summary["headline"][f"{model}_{tokens}"]["speedup_percent"]
                expected = (
                    f"\\speeduppoint{{\\{origins[model]}}}{{{x}}}"
                    f"{{{values['q1']:.4f}}}{{{values['median']:.4f}}}"
                    f"{{{values['q3']:.4f}}}{{${values['median']:.1f}\\%$}}"
                )
                with self.subTest(model=model, tokens=tokens):
                    self.assertEqual(source.count(expected), 1)

    def test_main_is_the_only_tex_source(self):
        self.assertEqual(list(ROOT.rglob("*.tex")), [ROOT / "tex/main.tex"])

    def test_readme_table_matches_summary(self):
        readme = (ROOT / "README.md").read_text()
        labels = {"muse": "Muse Glimmer 30B", "qwen": "Qwen3.8 27B"}
        for model in MODELS:
            for tokens in TOKENS:
                values = self.summary["headline"][f"{model}_{tokens}"]["speedup_percent"]
                expected = (
                    f"| {labels[model]} | {tokens:,} | {values['median']:+.3f}% | "
                    f"[{values['q1']:+.3f}%, {values['q3']:+.3f}%] |"
                )
                with self.subTest(model=model, tokens=tokens):
                    self.assertIn(expected, readme)


if __name__ == "__main__":
    unittest.main()
