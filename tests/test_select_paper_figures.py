"""Synthetic-only selector tests. No models, experimental PNGs or score files."""
import ast
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from PIL import Image, ImageDraw

SCRIPT = Path(__file__).resolve().parents[1] / "tools/select_paper_figures.py"
SPEC = importlib.util.spec_from_file_location("selector", SCRIPT)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.root = self.work / "pool"
        self.scores = self.work / "per_image_scores.csv"
        self.differences = self.work / "paired_differences.csv"
        self.output = self.work / "result"
        self.rows, self.diffs = [], []
        # 24 real synthetic PNG pairs: varied structure/colours and metric signs.
        for sid in range(24):
            base = [4.0 + sid * .08, 4.5 + sid * .02, .20 + sid * .002]
            sign = -1 if sid < 8 else 0 if sid < 16 else 1
            deltas = np.array([.4, .20, .02]) * sign * (1 + (sid % 8) / 10)
            for pool in ("G0", "G1"):
                image = Image.new("RGB", (512, 512), (55 + sid * 2, 100, 165 - sid))
                draw = ImageDraw.Draw(image)
                for y in range(0, 512, 32):
                    for x in range(0, 512, 32):
                        if (x // 32 + y // 32) % 2:
                            draw.rectangle((x, y, x + 25, y + 25), fill=(210, 180 - sid, 65 + sid))
                draw.ellipse((100 + sid, 90, 380, 400), fill=(80, 40 + sid, 180))
                if pool == "G1":
                    draw.rectangle((180, 180, 300, 270), fill=(200, 135 + sid, 80))
                relative = f"{pool}/coco/HSQR/img_pil_wm/{sid}.png"
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                values = np.array(base) + (deltas if pool == "G1" else 0)
                self.rows.append({"sample_id": str(sid), "generation_pool": pool,
                                  "watermark_status": "wm", "image_path": "Z:\\old-machine\\" + relative.replace("/", "\\"),
                                  "image_sha256": selector.sha256(path),
                                  **dict(zip(selector.METRICS.values(), values))})
            for m, metric in enumerate(selector.METRICS):
                self.diffs.append({"sample_id": str(sid), "watermark_status": "wm",
                                   "metric": metric, "g0_score": base[m],
                                   "g1_score": base[m] + deltas[m], "difference": deltas[m],
                                   "pairing_confidence": "synthetic_test_only"})
        self.save()

    def save(self):
        write_csv(self.scores, self.rows)
        write_csv(self.differences, self.diffs)

    def args(self, output=None):
        return ["--data-root", str(self.root), "--scores", str(self.scores),
                "--differences", str(self.differences), "--output-dir", str(output or self.output),
                "--rotation", "30", "--num-overview", "6", "--seed", "42", "--expected-pairs", "24"]

    def invoke(self, args=None):
        stream = io.StringIO()
        with redirect_stderr(stream), redirect_stdout(stream):
            code = selector.main(args or self.args())
        return code, stream.getvalue()

    def load(self):
        return selector.load_inputs(self.root.resolve(), self.scores, self.differences, 24)

    def assert_invalid(self, text):
        code, message = self.invoke()
        self.assertEqual(code, 2, message)
        self.assertIn(text, message)
        self.assertFalse(self.output.exists())

    def test_standalone_cli_outputs_and_source_immutability(self):
        before = {str(p): selector.sha256(p) for p in [*self.root.rglob("*.png"), self.scores, self.differences]}
        copied = self.work / "standalone/select_paper_figures.py"
        copied.parent.mkdir()
        shutil.copyfile(SCRIPT, copied)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        process = subprocess.run([sys.executable, "-B", str(copied), *self.args()],
                                 cwd=self.work, env=environment, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual({p.name for p in self.output.iterdir()}, set(selector.OUTPUTS))
        for name in selector.OUTPUTS[:2]:
            with Image.open(self.output / name) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.height, 100)
        with (self.output / "selected_candidates.csv").open(newline="", encoding="utf-8") as handle:
            selected = list(csv.DictReader(handle))
        self.assertEqual(sum(r["selection"] == "overview" for r in selected), 6)
        for category in ("positive", "near_zero", "negative"):
            group = [r for r in selected if r["selection"] == category]
            self.assertEqual(len(group), 3)
            self.assertEqual(len({r["sample_id"] for r in group}), 3)
        for row in selected:
            for field in selector.METRICS.values():
                self.assertAlmostEqual(float(row[f"G1_{field}"]) - float(row[f"G0_{field}"]),
                                       float(row[f"delta_{field}"]))
        protocol = (self.output / "selection_protocol.md").read_text(encoding="utf-8")
        self.assertIn("NON-FORMAL SIZE OVERRIDE", protocol)
        self.assertIn("synthetic_test_only", protocol)
        self.assertIn("PREVIEW", protocol)
        self.assertEqual(before, {path: selector.sha256(Path(path)) for path in before})

    def test_repeatable_ids_and_pixels_independent_of_row_order(self):
        code, message = self.invoke()
        self.assertEqual(code, 0, message)
        self.rows.reverse()
        self.diffs.reverse()
        self.save()
        other = self.work / "other"
        code, message = self.invoke(self.args(other))
        self.assertEqual(code, 0, message)
        for name in selector.OUTPUTS[:3]:
            self.assertEqual((self.output / name).read_bytes(), (other / name).read_bytes())

    def test_default_requires_1000_and_cli_defaults(self):
        args = selector.parser().parse_args(self.args()[:-2])
        self.assertEqual(args.expected_pairs, 1000)
        self.assertEqual(args.num_freeu_per_category, 3)
        self.assert_invalid_default()
        ast.parse(SCRIPT.read_text(encoding="utf-8"), feature_version=(3, 10))

    def assert_invalid_default(self):
        code, message = self.invoke(self.args()[:-2])
        self.assertEqual(code, 2)
        self.assertIn("Expected 1000", message)

    def test_duplicate_score_rejected(self):
        self.rows.append(self.rows[0].copy())
        self.save()
        self.assert_invalid("duplicate wm score")

    def test_missing_pair_rejected(self):
        self.rows.pop()
        self.save()
        self.assert_invalid("complete wm pairs")

    def test_extra_unscored_png_excluded_missing_scored_png_rejected(self):
        source = self.root / "G0/coco/HSQR/img_pil_wm/0.png"
        shutil.copyfile(source, source.with_name("999.png"))
        pairs, audit = self.load()
        self.assertEqual(len(pairs), 24)
        self.assertEqual(audit["directory_inventory"]["G0"]["unscored_pngs_excluded"], 1)
        self.assertNotIn("999", {p["sample_id"] for p in pairs})
        source.unlink()
        self.assert_invalid("Missing or unsafe image")

    def test_image_hash_mismatch_rejected(self):
        self.rows[0]["image_sha256"] = "0" * 64
        self.save()
        self.assert_invalid("SHA256 mismatch")

    def test_cross_pool_path_and_traversal_rejected(self):
        self.rows[0]["image_path"] = "G1/coco/HSQR/img_pil_wm/0.png"
        self.save()
        self.assert_invalid("image_path does not match")
        self.rows[0]["sample_id"] = "../0"
        self.save()
        self.assert_invalid("Invalid canonical sample_id")

    def test_nonfinite_and_missing_metric_rejected(self):
        self.rows[0]["musiq_ava"] = "NaN"
        self.save()
        self.assert_invalid("Non-finite")
        self.rows[0]["musiq_ava"] = ""
        self.save()
        self.assert_invalid("Non-numeric")

    def test_difference_values_missing_and_duplicates_rejected(self):
        self.diffs[0]["difference"] = 123
        self.save()
        self.assert_invalid("Scores/differences disagree")
        self.diffs[0]["difference"] = -.4
        self.diffs.append(self.diffs[0].copy())
        self.save()
        self.assert_invalid("Unknown/duplicate wm difference")
        self.diffs = self.diffs[1:-1]
        self.save()
        self.assert_invalid("Missing wm paired differences")

    def test_split_membership_checked_when_available(self):
        manifest = self.root / "manifests/formal_splits.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({"splits": {"test": [{"sample_id": sid} for sid in range(24)]}}))
        self.assertEqual(self.load()[1]["wm_pairs"], 24)
        manifest.write_text(json.dumps({"splits": {"test": [{"sample_id": 999}]}}))
        self.assert_invalid("test membership differs")

    def test_no_wm_rows_are_not_used(self):
        ignored = self.rows[0].copy()
        ignored.update(watermark_status="no_wm", image_path="not-present.png", musiq_ava="")
        self.rows.append(ignored)
        ignored_diff = self.diffs[0].copy()
        ignored_diff.update(watermark_status="no_wm", difference="")
        self.diffs.append(ignored_diff)
        self.save()
        self.assertEqual(len(self.load()[0]), 24)

    def test_png_spec_corruption_rejected(self):
        path = self.root / "G0/coco/HSQR/img_pil_wm/0.png"
        Image.new("RGB", (64, 64)).save(path)
        self.rows[0]["image_sha256"] = selector.sha256(path)
        self.save()
        self.assert_invalid("Expected 512x512 RGB PNG")

    def test_fixed_thresholds_no_silent_relaxation(self):
        stats = np.zeros((24, 7))
        with self.assertRaisesRegex(ValueError, "Only 0 Overview"):
            selector.select_overview(stats, np.zeros((24, 3)), np.linspace(0, 1, 24),
                                     np.zeros(24), 6, np.random.default_rng(42))

    def test_quality_strata_are_balanced(self):
        pairs, _ = self.load()
        stats, descriptors, quality, readability = selector.prepare_features(pairs, 30)
        selected, info = selector.select_overview(stats, descriptors, quality, readability, 8,
                                                   np.random.default_rng(42))
        self.assertEqual(info["selected_by_quality_quartile"], [2, 2, 2, 2])
        self.assertTrue(all(.1 <= quality[i] <= .9 for i, *_ in selected))

    def test_freeu_quantiles_avoid_extremes_and_categories_are_unique(self):
        pairs, _ = self.load()
        selected, info, effect, _, _ = selector.select_freeu(pairs, 3, np.random.default_rng(42))
        self.assertEqual(len({i for i, *_ in selected}), 9)
        for category in ("positive", "negative"):
            ids = [i for i, cat, *_ in selected if cat == category]
            pool = range(16, 24) if category == "positive" else range(8)
            self.assertNotIn(min(pool, key=lambda i: effect[i]), ids)
            self.assertNotIn(max(pool, key=lambda i: effect[i]), ids)
            self.assertEqual(info["categories"][category]["target_quantiles"], [.25, .5, .75])
        self.assertEqual(len(info["warnings"]), 0)

    def test_no_fabricated_negative_or_zero_category(self):
        pairs, _ = self.load()
        for pair in pairs:
            for field in selector.METRICS.values():
                pair["G1"][field] = pair["G0"][field] + .5
        selected, info, *_ = selector.select_freeu(pairs, 3, np.random.default_rng(42))
        self.assertTrue(all(cat == "positive" for _, cat, *_ in selected))
        self.assertEqual(info["categories"]["negative"]["eligible"], 0)
        self.assertEqual(info["categories"]["near_zero"]["eligible"], 0)
        self.assertEqual(len(info["warnings"]), 2)

    def test_cancellation_is_not_near_zero(self):
        pairs, _ = self.load()
        for pair in pairs:
            for field, delta in zip(selector.METRICS.values(), [2, -2, 0]):
                pair["G1"][field] = pair["G0"][field] + delta
        _, info, *_ = selector.select_freeu(pairs, 3, np.random.default_rng(42))
        self.assertEqual(info["categories"]["near_zero"]["eligible"], 0)

    def test_rotation_matches_existing_attack_pixel_for_pixel(self):
        spec = importlib.util.spec_from_file_location("research_attacks", SCRIPT.parents[1] / "src/research_attacks.py")
        attacks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(attacks)
        rng = np.random.default_rng(42)
        for size in ((512, 512), (511, 527), (317, 221)):
            image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
            for angle in (30, -30, 0, 90, 17.5):
                expected = attacks.rotate_image(image, angle)
                actual = selector.rotate_preview(image, angle)
                self.assertEqual(actual.size, (512, 512))
                self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertNotIn("torch", selector.__dict__)

    def test_protect_inputs_and_existing_output(self):
        code, message = self.invoke(self.args(self.root / "G0/derived"))
        self.assertEqual(code, 2)
        self.assertIn("separate", message)
        self.output.mkdir()
        (self.output / "keep.txt").write_text("keep")
        code, message = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("never overwritten", message)
        self.assertEqual((self.output / "keep.txt").read_text(), "keep")

    def test_invalid_numeric_arguments_and_help(self):
        for extra in (["--rotation", "nan"], ["--seed", "-1"], ["--num-overview", "0"]):
            code, _ = self.invoke(self.args() + extra)
            self.assertEqual(code, 2)
        process = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0)
        self.assertIn("--num-overview", process.stdout)


if __name__ == "__main__":
    unittest.main()
