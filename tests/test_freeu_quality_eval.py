"""Synthetic protocol and resumability tests; no real evaluator downloads."""
from contextlib import closing, contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image
import paper_protocol as p
import freeu_quality_data as data
import freeu_quality_eval as quality


def save(path, value):
    Path(path).write_bytes(p.canonical(p.seal(value)) + b"\n")


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "manifests").mkdir()
        cls.metadata = cls.root / "metadata.json"
        cls.metadata.write_text(json.dumps({"annotations": [
            {"image_id": i, "caption": f"caption {i}"} for i in range(4500)]}), encoding="utf-8")
        cls.splits = p.create_splits(cls.metadata)
        cls.plan = p.generation_plan(cls.splits)
        save(cls.root / "manifests/formal_splits.json", cls.splits)
        save(cls.root / "manifests/generation_plan.json", cls.plan)
        cls.ids = [r["sample_id"] for r in cls.splits["splits"]["test"]]
        cls.png = io.BytesIO()
        Image.new("RGB", (4, 4), (20, 30, 40)).save(cls.png, format="PNG")
        cls.png = cls.png.getvalue()
        import hashlib
        image_hash = hashlib.sha256(cls.png).hexdigest()
        cls.manifests = {}
        for pool in ("G0", "G1"):
            directory = cls.root / pool / "coco/HSQR"
            for kind in ("img_pil", "img_pil_wm"):
                (directory / kind).mkdir(parents=True)
                for sid in cls.ids:
                    (directory / kind / f"{sid}.png").write_bytes(cls.png)
            config = {"protocol": p.PROTOCOL, "generation_protocol": p.GENERATION,
                "generation_plan_hash": cls.plan["sha256"], "generation_pool": pool,
                "freeu_enabled": pool == "G1", "freeu": p.FREEU, "model": {"id": "fixture"},
                "steps": 50, "guidance": 7.5, "model_dtype": "float32", "fft_dtype": "float32",
                "vae_slicing": True, "geometry": p.GEOMETRY, "batch_size": 2,
                "code": {"git_commit": "fixture"}, "environment": {"fixture": True}}
            save(directory / "generation_config.json", config)
            records = {}
            for row in cls.plan["records"]:
                sid = row["sample_id"]
                record = {**row, "config_hash": p.digest(config), "images": {}, "latents": {}}
                for status, folder in (("no_wm", "img_pil"), ("wm", "img_pil_wm")):
                    record["images"][status] = {"path": f"/old/machine/{pool}/coco/HSQR/{folder}/{sid}.png",
                                                "sha256": image_hash}
                    latent_hash = p.digest(["latent", sid, status])
                    record["latents"][status] = {"path": f"/not/copied/{sid}-{status}.npy", "sha256": latent_hash}
                    record[f"{status}_latent_hash"] = latent_hash
                records[str(sid)] = p.seal(record)
            manifest = p.seal({**config, "records": records,
                               "patterns": {"path": "/not/copied/patterns.pt", "sha256": "a" * 64}})
            cls.manifests[pool] = manifest
            save(directory / "generation_manifest.json", manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self, **kwargs):
        return SimpleNamespace(data_root=str(self.root), split="test", dataset_id="coco",
            metadata=str(self.metadata), split_manifest=None, generation_plan=None,
            legacy=kwargs.get("legacy", False))

    @contextmanager
    def changed_manifest(self, modify):
        path = self.root / "G1/coco/HSQR/generation_manifest.json"
        old = path.read_bytes()
        value = deepcopy(self.manifests["G1"])
        modify(value)
        save(path, value)
        try:
            yield
        finally:
            path.write_bytes(old)

    def test_formal_relocation_full_membership_without_latent_files(self):
        report, images = data.inspect_data(self.args())
        self.assertEqual(report["status"], "PASS", report["issues"][:3])
        self.assertEqual(len(images), 4000)
        self.assertEqual(report["relocated_images"], 4000)
        self.assertEqual(report["verification_counts"]["paired_samples"], 1000)
        self.assertEqual({r["sample_id"] for r in images}, set(self.ids))
        self.assertNotEqual(set(self.ids), set(range(1000)))
        self.assertFalse(list(self.root.rglob("*.npy")))

    def test_missing_image_and_changed_bytes_fail(self):
        path = self.root / f"G1/coco/HSQR/img_pil_wm/{self.ids[0]}.png"
        old = path.read_bytes()
        try:
            path.unlink()
            report, _ = data.inspect_data(self.args())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["groups"]["G1/wm"]["missing"], 1)
            path.write_bytes(old + b"changed")
            report, _ = data.inspect_data(self.args())
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(any("SHA256" in i["message"] for i in report["issues"]))
        finally:
            path.write_bytes(old)

    def test_prompt_seed_key_and_latent_pairing_fail(self):
        for field in ("prompt", "latent_seed", "key_index", "wm_latent_hash"):
            def modify(manifest):
                record = manifest["records"][str(self.ids[0])]
                if field == "prompt":
                    record[field] = "incorrect prompt"
                elif field == "wm_latent_hash":
                    record[field] = "e" * 64
                    record["latents"]["wm"]["sha256"] = record[field]
                else:
                    record[field] += 1
                manifest["records"][str(self.ids[0])] = p.seal(record)
            with self.subTest(field=field), self.changed_manifest(modify):
                report, _ = data.inspect_data(self.args())
                self.assertEqual(report["status"], "FAIL")
                self.assertTrue(any(i["code"] in {"generation_record", "sample_pairing"} for i in report["issues"]))

    def test_bad_seal_cannot_be_bypassed_by_legacy(self):
        path = self.root / "G1/coco/HSQR/generation_manifest.json"
        old = path.read_bytes()
        try:
            value = deepcopy(self.manifests["G1"])
            value["steps"] = 49
            path.write_text(json.dumps(value), encoding="utf-8")
            report, _ = data.inspect_data(self.args(legacy=True))
            self.assertEqual(report["status"], "FAIL")
        finally:
            path.write_bytes(old)

    def test_explicit_legacy_marks_unknown_seed_key(self):
        paths = [self.root / g / "coco/HSQR/generation_manifest.json" for g in ("G0", "G1")]
        originals = [path.read_bytes() for path in paths]
        try:
            for path in paths:
                path.unlink()
            formal, _ = data.inspect_data(self.args())
            self.assertEqual(formal["status"], "FAIL")
            report, images = data.inspect_data(self.args(legacy=True))
            self.assertEqual(report["status"], "WARNING")
            self.assertEqual(report["pairing_confidence"], "legacy_unverified")
            self.assertEqual(len(images), 4000)
            self.assertTrue(all(r["seed"] is None and r["key_index"] is None for r in images))
        finally:
            for path, old in zip(paths, originals):
                path.write_bytes(old)

    def test_metadata_binding_and_malformed_split_fail(self):
        path = self.metadata
        old = path.read_bytes()
        try:
            path.write_bytes(old + b" ")
            report, _ = data.inspect_data(self.args())
            self.assertEqual(report["status"], "FAIL")
        finally:
            path.write_bytes(old)
        split_path = self.root / "manifests/formal_splits.json"
        old = split_path.read_bytes()
        try:
            split_path.write_text("[]", encoding="utf-8")
            report, images = data.inspect_data(self.args())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(images, [])
        finally:
            split_path.write_bytes(old)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_root = self.root / "data"
        self.data_root.mkdir()
        self.output = self.root / "freeu_quality_outputs/smoke"
        self.ids = [17 + 11 * i for i in range(21)]
        self.images = [{"sample_id": sid, "generation_pool": pool,
            "watermark_status": status, "image_path": str(self.root / f"{sid}-{pool}-{status}.png"),
            "image_sha256": "a" * 64, "prompt": f"caption {sid}", "seed": 42 + sid, "key_index": sid}
            for sid in self.ids for pool in ("G0", "G1") for status in ("no_wm", "wm")]
        self.report = {"status": "PASS", "issues": [], "groups": {}, "sample_ids": self.ids,
            "pairing_confidence": "formal_image_verified_latent_hash_claims", "source_manifests": {}}
        self.env = {"packages": {"torch": None, "torchvision": None, "pyiqa": None},
                    "cuda_available": False, "cuda_runtime": None}
        self.calls, self.loaded, self.fail_once = [], [], False
        outer = self
        class FakeEvaluator:
            def __init__(self, metric, args):
                outer.loaded.append(metric)
                self.identity = {"model": metric, "weights": "fake", "shape": (1, 2)}
            def score(self, metric, items):
                if outer.fail_once and outer.calls:
                    outer.fail_once = False
                    raise KeyboardInterrupt()
                outer.calls.extend((metric, quality.image_key(r)) for r in items)
                base = 0.2 if metric == "clip" else 5.
                return [base + (0.1 if r["generation_pool"] == "G1" else 0.) for r in items]
            def close(self):
                pass
        self.factory = FakeEvaluator

    def tearDown(self):
        self.temp.cleanup()

    def run_main(self, extra=()):
        argv = ["smoke", "--data-root", str(self.data_root), "--output-dir", str(self.output),
                "--limit", "20", "--device", "cpu", "--batch-size", "7", "--bootstrap-samples", "100", *extra]
        with patch.object(quality, "inspect_data", return_value=(deepcopy(self.report), self.images)), \
             patch.object(quality, "environment", return_value=deepcopy(self.env)), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return quality.main(argv, evaluator_factory=self.factory)

    def test_smoke_fixed_prefix_all_metrics_and_complete_resume(self):
        self.assertEqual(self.run_main(), 0)
        with (self.output / "per_image_scores.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 80)
        self.assertEqual({int(r["sample_id"]) for r in rows}, set(self.ids[:20]))
        self.assertTrue(all(r["musiq_ava"] and r["laion_aes"] and r["clip_similarity"] for r in rows))
        self.assertEqual(len(self.calls), 240)
        self.assertEqual(len(self.loaded), 3)
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(len(self.calls), 240)
        self.assertEqual(len(self.loaded), 3)
        manifest = json.loads((self.output / "experiment_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["spec"]["metrics"], list(quality.METRICS))
        self.assertEqual(len(manifest["invocations"]), 2)
        summary = json.loads((self.output / "quality_summary.json").read_text())
        self.assertEqual(summary["formal_test_count"], 21)
        self.assertTrue(all(r["n"] == 20 and r["missing_count"] == 0 for r in summary["groups"]))
        self.assertTrue(all(abs(r["mean_difference"] - .1) < 1e-10 for r in summary["comparisons"]))
        self.assertEqual(self.run_main(["--metrics", "clip"]), 2)

    def test_interruption_preserves_batch_and_resumes_only_missing(self):
        self.fail_once = True
        self.assertEqual(self.run_main(["--metrics", "laion_aes"]), 130)
        with closing(sqlite3.connect(self.output / "scores.sqlite3")) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 7)
        manifest = json.loads((self.output / "experiment_manifest.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(self.run_main(["--metrics", "laion_aes"]), 0)
        self.assertEqual(len(self.calls), 80)
        self.assertEqual(len(set(self.calls)), 80)
        with closing(sqlite3.connect(self.output / "scores.sqlite3")) as db:
            db.execute("UPDATE scores SET sha256='corrupt' WHERE rowid=1")
            db.commit()
        self.assertEqual(self.run_main(["--metrics", "laion_aes"]), 2)

    def test_failed_preflight_does_not_instantiate_evaluator(self):
        self.report["status"] = "FAIL"
        self.report["issues"] = [{"level": "FAIL", "code": "missing", "message": "fixture"}]
        self.assertEqual(self.run_main(), 2)
        self.assertEqual(self.loaded, [])
        self.assertFalse((self.output / "scores.sqlite3").exists())

    def test_statistics_known_paired_bootstrap_and_duplicates(self):
        rows = [{"sample_id": sid, "generation_pool": pool, "watermark_status": status,
                 "laion_aes": float(sid + (2 if pool == "G1" else 0))}
                for sid in range(3) for pool in ("G0", "G1") for status in ("no_wm", "wm")]
        a = quality.summarize(rows, ["laion_aes"], 3, bootstrap_samples=300)
        b = quality.summarize(rows, ["laion_aes"], 3, bootstrap_samples=300)
        self.assertEqual(a, b)
        self.assertTrue(all(c["mean_difference"] == c["ci95_low"] == c["ci95_high"] == 2
                            and c["g1_higher_fraction"] == 1 for c in a["comparisons"]))
        self.assertTrue(all(g["std"] == 1 for g in a["groups"]))
        with self.assertRaises(ValueError):
            quality.summarize(rows + [rows[0]], ["laion_aes"], 3)

    def test_output_cannot_contain_source_metadata(self):
        args = SimpleNamespace(data_root=str(self.data_root), output_dir=str(self.output),
                               metadata=str(self.output / "quality_summary.json"))
        with self.assertRaises(ValueError):
            quality.prepare_output(args)


if __name__ == "__main__":
    unittest.main()
