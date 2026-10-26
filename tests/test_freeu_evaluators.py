"""Tiny evaluator protocol tests; no pretrained models or downloads."""
import ast
import hashlib
from pathlib import Path
import sys
import tempfile
from types import FunctionType
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image
import freeu_quality_eval as quality
try:
    import torch
except ImportError:
    torch = None


class EvaluatorTests(unittest.TestCase):
    def test_score_validation(self):
        for metric, value in (("clip", 2), ("clip", float("nan")),
                              ("musiq-ava", 100), ("laion_aes", float("inf"))):
            with self.subTest(metric=metric, value=value), self.assertRaises(ValueError):
                quality.valid_score(metric, value)
        self.assertEqual(quality.valid_score("clip", -1), -1)
        self.assertEqual(quality.valid_score("laion_aes", -2), -2)

    def test_preprocessing_identity_has_no_process_address(self):
        def rgb(image):
            return image
        clone = FunctionType(rgb.__code__, rgb.__globals__, name=rgb.__name__)
        clone.__qualname__ = rgb.__qualname__
        class Compose:
            def __init__(self, function):
                self.transforms = [function]
        a = quality.preprocessing_identity(Compose(rgb))
        b = quality.preprocessing_identity(Compose(clone))
        self.assertEqual(a, b)
        self.assertNotIn("0x", str(a))

    @unittest.skipIf(torch is None, "Torch unavailable; no packages installed by tests")
    def test_loaded_state_fingerprint_includes_parameters_and_scalar_buffers(self):
        model = torch.nn.Linear(2, 2)
        model.register_buffer("scalar", torch.tensor(3.))
        before = quality.state_fingerprint(model)
        self.assertEqual(before, quality.state_fingerprint(model))
        model.scalar.add_(1)
        after_buffer = quality.state_fingerprint(model)
        self.assertNotEqual(before, after_buffer)
        with torch.no_grad():
            model.weight[0, 0].add_(1)
        self.assertNotEqual(after_buffer, quality.state_fingerprint(model))

    @unittest.skipIf(torch is None, "Torch unavailable; no packages installed by tests")
    def test_clip_matches_original_formula_inference_mode_and_hash_guard(self):
        observations = []
        class TinyModel:
            def encode_image(self, batch):
                observations.append(torch.is_inference_mode_enabled())
                return batch.clone()
            def encode_text(self, text):
                return text.clone()
        def preprocess(image):
            return torch.tensor(image.getpixel((0, 0)), dtype=torch.float32)
        def tokenize(prompts):
            vectors = {"a": [4., 3., 0.], "b": [-2., 0., 0.]}
            return torch.tensor([vectors[prompt] for prompt in prompts])
        evaluator = quality.Evaluator.__new__(quality.Evaluator)
        evaluator.torch, evaluator.device = torch, "cpu"
        evaluator.model, evaluator.preprocess, evaluator.tokenizer = TinyModel(), preprocess, tokenize
        with tempfile.TemporaryDirectory() as temp:
            items, images = [], []
            for i, color in enumerate(((3, 4, 0), (10, 0, 0))):
                image = Image.new("RGB", (2, 2), color)
                path = Path(temp) / f"{i}.png"
                image.save(path)
                images.append(image)
                items.append({"image_path": str(path), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "prompt": "a" if i == 0 else "b"})
            actual = evaluator.score("clip", items)
            self.assertTrue(all(observations))
            # Execute the repository's actual score function without importing
            # utils and its unrelated generation/detector dependencies.
            source = Path(quality.__file__).parent / "utils.py"
            function = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                            if isinstance(n, ast.FunctionDef) and n.name == "get_clip_score")
            function.decorator_list = []
            module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
            namespace = {"torch": torch, "Image": Image}
            exec(compile(module, str(source), "exec"), namespace)
            expected = namespace["get_clip_score"](images, ["a", "b"], evaluator.model,
                                                    preprocess, tokenize, device="cpu").tolist()
            self.assertEqual(actual, expected)
            self.assertAlmostEqual(actual[0], .96, places=6)
            self.assertEqual(actual[1], -1.)
            Path(items[0]["image_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since preflight"):
                evaluator.score("clip", items)


if __name__ == "__main__":
    unittest.main()
