"""Read-only G0/G1 image-quality evaluation; never generates or inverts images.

See docs/freeu_quality_eval.md for installation, provenance and usage.
Only preflight uses no evaluator models. Each output directory binds one fixed
sample set, metric selection and evaluator configuration for automatic resume.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import tempfile

from paper_protocol import canonical, digest, file_hash
from freeu_quality_data import inspect_data

METRICS = {"musiq-ava": "musiq_ava", "laion_aes": "laion_aes",
           "clip": "clip_similarity"}
IMAGE_FIELDS = ["sample_id", "generation_pool", "watermark_status",
                "image_path", "image_sha256", *METRICS.values()]
SCHEMA = "freeu_quality_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix=".quality-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_csv(path, rows, fields):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix=".quality-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def environment():
    result = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "machine": platform.machine(),
              "processor": platform.processor(), "packages": {}}
    for name in ("torch", "torchvision", "numpy", "Pillow", "pyiqa", "timm",
                 "transformers", "diffusers", "compressai", "ftfy", "regex",
                 "huggingface-hub", "safetensors", "scipy", "openai-clip", "einops"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    result.update(cuda_available=False, cuda_runtime=None, gpu=[])
    if result["packages"]["torch"]:
        try:
            import torch
            result["cuda_runtime"] = torch.version.cuda
            result["cuda_available"] = torch.cuda.is_available()
            result["cudnn"] = torch.backends.cudnn.version()
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                result["gpu"].append({"index": index, "name": props.name,
                                      "total_memory_bytes": props.total_memory})
        except Exception as exc:
            result["torch_import_error"] = str(exc)
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True,
                           timeout=10)
        result["nvidia_driver"] = p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        result["nvidia_driver"] = None
    return result


def code_identity():
    root = Path(__file__).resolve().parent.parent
    result = {}
    for key, command in (("commit", ["rev-parse", "HEAD"]),
                         ("dirty", ["status", "--porcelain", "--untracked-files=normal"])):
        try:
            p = subprocess.run(["git", *command], cwd=root, capture_output=True,
                               text=True, timeout=10)
            result[key] = p.stdout.strip() if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result[key] = None
    result["dirty"] = bool(result["dirty"])
    paths = [root / "src" / name for name in (
        "freeu_quality_eval.py", "freeu_quality_data.py", "paper_protocol.py",
        "metric.py", "utils.py")]
    paths += sorted((root / "src" / "open_clip").rglob("*.py"))
    paths += sorted((root / "src" / "open_clip").rglob("*.json"))
    paths += sorted((root / "src" / "open_clip").rglob("*.gz"))
    result["source_sha256"] = digest({str(p.relative_to(root)): file_hash(p)
                                      for p in paths if p.is_file()})
    return result


def valid_score(metric, score):
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {metric} score: {value}")
    if metric == "clip" and not -1.00001 <= value <= 1.00001:
        raise ValueError(f"CLIP cosine outside [-1, 1]: {value}")
    if metric == "musiq-ava" and not 0.9999 <= value <= 10.0001:
        raise ValueError(f"MUSIQ-AVA AVA expectation outside [1, 10]: {value}")
    return value


def state_fingerprint(model):
    """Hash actual loaded parameters/buffers, without saving evaluator weights."""
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(canonical([name, str(tensor.dtype), list(tensor.shape)]))
        raw = tensor.detach().reshape(-1).contiguous().view(torch.uint8)
        # Transfer bounded chunks, also for the large ViT-g-14 parameters.
        for start in range(0, raw.numel(), 4 * 1024 * 1024):
            h.update(raw[start:start + 4 * 1024 * 1024].cpu().numpy().tobytes())
    return h.hexdigest()


def preprocessing_identity(transform):
    """Compose repr includes function memory addresses; retain stable semantics."""
    if hasattr(transform, "transforms"):
        return [preprocessing_identity(t) for t in transform.transforms]
    if inspect.isfunction(transform):
        return transform.__module__ + "." + transform.__qualname__
    return repr(transform)


class Evaluator:
    def __init__(self, metric, args):
        import torch
        self.torch, self.metric, self.device = torch, metric, args.device
        torch.manual_seed(42)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; inspect preflight environment")
        # Fixed float32 inference; no autocast, TF32, or silent CPU fallback.
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if metric == "clip":
            import open_clip
            from open_clip.version import __version__ as open_clip_version
            bundled = Path(__file__).resolve().parent / "open_clip"
            if Path(open_clip.__file__).resolve().parent != bundled:
                raise ValueError("CLIP must use this repository's bundled open_clip")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-g-14", pretrained="laion2b_s12b_b42k", device=self.device,
                precision="fp32")
            self.tokenizer = open_clip.get_tokenizer("ViT-g-14")
            self.identity = {"model": "ViT-g-14", "pretrained": "laion2b_s12b_b42k",
                "open_clip_version": open_clip_version,
                "pretrained_config": open_clip.get_pretrained_cfg(
                    "ViT-g-14", "laion2b_s12b_b42k"),
                "preprocessing": preprocessing_identity(self.preprocess),
                "tokenizer": "bundled open_clip.get_tokenizer(ViT-g-14), context_length=77",
                "tokenizer_vocab_sha256": file_hash(bundled / "bpe_simple_vocab_16e6.txt.gz"),
                "score_protocol": "L2-normalize each feature; matched image/text dot product; no scale, clamp or cross-sample averaging"}
        else:
            import pyiqa
            self.model = pyiqa.create_metric(metric, device=self.device, as_loss=False)
            if bool(self.model.lower_better):
                raise ValueError(f"Unexpected lower-better evaluator for {metric}")
            self.identity = {"model": metric,
                "pyiqa_version": importlib.metadata.version("pyiqa"),
                "config": repr(getattr(self.model, "metric_opts", None)),
                "network": type(self.model.net).__module__ + "." + type(self.model.net).__name__,
                "preprocessing": "PNG decoded RGB; float32 NCHW [0,1]; original resolution; internal official PyIQA transforms",
                "score_protocol": "official PyIQA " + metric + ", higher is better"}
            self.identity["pretrained_weight_identifiers"] = (
                ["musiq_ava_ckpt-e8d3f067.pth", "AVA"] if metric == "musiq-ava"
                else ["sac+logos+ava1-l14-linearMSE.pth", "OpenAI CLIP ViT-L/14"])
            # metric_opts contains the official pretrained dataset/configuration.
            from pyiqa.default_model_configs import DEFAULT_CONFIGS
            self.identity["official_config"] = json.loads(json.dumps(DEFAULT_CONFIGS[metric], default=str))
        self.model.float().eval()
        self.identity.update(dtype="float32", device=self.device,
                             inference_mode=True, autocast=False, torch_seed=42,
                             cudnn_benchmark=False,
                             cudnn_deterministic=torch.backends.cudnn.deterministic,
                             deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                             cuda_matmul_allow_tf32=False if self.device.startswith("cuda") else None,
                             loaded_state_sha256=state_fingerprint(self.model))
        if metric == "laion_aes":
            # PyIQA keeps OpenAI CLIP in a plain list, outside state_dict().
            backbone = self.model.net.clip_model[0]
            backbone.to(device=self.device, dtype=torch.float32).eval()
            self.identity["clip_backbone_state_sha256"] = state_fingerprint(backbone)
            self.identity["clip_backbone_dtype"] = str(next(backbone.parameters()).dtype)

    def score(self, metric, items):
        from PIL import Image
        import numpy as np
        torch = self.torch
        images = []
        for item in items:
            data = Path(item["image_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["image_sha256"]:
                raise ValueError(f"Image changed since preflight: {item['image_path']}")
            with Image.open(io.BytesIO(data)) as image:
                images.append(image.convert("RGB"))
        with torch.inference_mode():
            if metric == "clip":
                batch = torch.stack([self.preprocess(im) for im in images]).to(self.device)
                text = self.tokenizer([item["prompt"] for item in items]).to(self.device)
                image_features = self.model.encode_image(batch)
                text_features = self.model.encode_text(text)
                # Identical to utils.get_clip_score, without importing generation dependencies.
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                values = (image_features * text_features).sum(-1)
            else:
                batch = torch.stack([torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1)
                                     for im in images]).to(device=self.device, dtype=torch.float32) / 255.
                values = self.model(batch)
        return values.detach().reshape(-1).float().cpu().tolist()

    def close(self):
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


def summarize(rows, metrics, expected_count, bootstrap_seed=1729,
              bootstrap_samples=10000,
              pairing_confidence="formal_image_verified_latent_hash_claims"):
    import numpy as np
    groups, comparisons, differences = [], [], []
    indexed = {}
    for row in rows:
        key = (row["sample_id"], row["generation_pool"], row["watermark_status"])
        if key in indexed:
            raise ValueError(f"Duplicate scored image: {key}")
        indexed[key] = row
    for metric in metrics:
        field = METRICS[metric]
        for status in ("no_wm", "wm"):
            for pool in ("G0", "G1"):
                values = [valid_score(metric, r[field]) for r in rows
                          if r["generation_pool"] == pool and r["watermark_status"] == status
                          and r.get(field) is not None]
                n = len(values)
                groups.append({"metric": metric, "generation_pool": pool,
                    "watermark_status": status, "n": n, "expected_count": expected_count,
                    "missing_count": expected_count - n,
                    "mean": float(np.mean(values)) if n else None,
                    "std": float(np.std(values, ddof=1)) if n > 1 else None,
                    "pairing_confidence": pairing_confidence,
                    "validity": "complete" if n == expected_count else "incomplete"})
            deltas = []
            ids = sorted({r["sample_id"] for r in rows})
            for sid in ids:
                a, b = indexed.get((sid, "G0", status)), indexed.get((sid, "G1", status))
                if a is None or b is None or a.get(field) is None or b.get(field) is None:
                    continue
                a_score, b_score = valid_score(metric, a[field]), valid_score(metric, b[field])
                delta = b_score - a_score
                deltas.append(delta)
                differences.append({"sample_id": sid, "watermark_status": status,
                    "metric": metric, "g0_score": a_score, "g1_score": b_score,
                    "difference": delta, "pairing_confidence": pairing_confidence})
            n = len(deltas)
            low, high = None, None
            if n > 1:
                # Resample matched sample IDs, preserving G0/G1 dependence.
                rng = np.random.default_rng(bootstrap_seed)
                data, means = np.asarray(deltas, dtype=np.float64), []
                for start in range(0, bootstrap_samples, 256):
                    size = min(256, bootstrap_samples - start)
                    means.extend(data[rng.integers(0, n, size=(size, n))].mean(axis=1).tolist())
                low, high = map(float, np.quantile(means, [0.025, 0.975]))
            comparisons.append({"metric": metric, "watermark_status": status,
                "contrast": "G1 minus G0", "n_pairs": n,
                "missing_count": expected_count - n,
                "mean_difference": float(np.mean(deltas)) if n else None,
                "ci95_low": low, "ci95_high": high,
                "g1_higher_fraction": float(np.mean(np.asarray(deltas) > 0)) if n else None,
                "validity": "complete" if n == expected_count and n > 1 else "incomplete",
                "pairing_confidence": pairing_confidence})
    return {"groups": groups, "comparisons": comparisons,
        "per_pair_differences": differences,
        "bootstrap": {"method": "paired percentile bootstrap of mean(G1-G0)",
                      "confidence": 0.95, "seed": bootstrap_seed, "resamples": bootstrap_samples},
        "std_definition": "sample standard deviation (ddof=1); null for n<2",
        "interpretation": "A CI containing zero does not establish equivalence. Legacy associations do not establish formal pairing.",
        "enabled_metrics": list(metrics), "pairing_confidence": pairing_confidence}


def write_outputs(output_dir, rows, summary):
    output_dir = Path(output_dir)
    atomic_csv(output_dir / "per_image_scores.csv", rows, IMAGE_FIELDS)
    flat = [{"record_type": "group", **r} for r in summary["groups"]]
    flat += [{"record_type": "comparison", **r} for r in summary["comparisons"]]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    atomic_csv(output_dir / "quality_summary.csv", flat, fields)
    diff_fields = ["sample_id", "watermark_status", "metric", "g0_score", "g1_score",
                   "difference", "pairing_confidence"]
    atomic_csv(output_dir / "paired_differences.csv", summary["per_pair_differences"], diff_fields)
    atomic_json(output_dir / "quality_summary.json", summary)


@contextmanager
def output_lock(output_dir):
    with (output_dir / ".run.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("Output directory is in use by another evaluator") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError("Output directory is in use by another evaluator") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_output(args):
    root = Path(args.data_root).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "freeu_quality_outputs"
    if output == root or output in root.parents:
        raise ValueError("Output directory must be separate from source data")
    protected = [root / p for p in ("G0", "G1", "manifests")]
    protected += [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent / "tests"]
    for path in protected:
        if output == path.resolve() or path.resolve() in output.parents:
            raise ValueError(f"Output directory overlaps protected inputs: {path}")
    for name in ("metadata", "split_manifest", "generation_plan"):
        supplied = getattr(args, name, None)
        if supplied:
            path = Path(supplied).resolve()
            if path == output or output in path.parents:
                raise ValueError(f"Output directory contains an explicit input: {path}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def image_key(item):
    return f"{item['sample_id']}:{item['generation_pool']}:{item['watermark_status']}"


def run_evaluation(args, report, images, output, evaluator_factory=None):
    selected_ids = sorted({r["sample_id"] for r in images})
    if args.command == "smoke":
        selected_ids = selected_ids[:args.limit]
    selected = set(selected_ids)
    images = [r for r in images if r["sample_id"] in selected]
    if not images:
        raise ValueError("No validated samples to evaluate")
    identities = [{k: r.get(k) for k in ("sample_id", "generation_pool", "watermark_status",
                   "image_sha256", "prompt", "seed", "key_index")} for r in images]
    code = code_identity()
    # Exclude path and runtime time: a byte-identical copied dataset is resumable.
    spec = {"schema": SCHEMA, "command": args.command, "split": args.split,
        "samples_sha256": digest(identities), "sample_ids": selected_ids,
        "metrics": args.metrics, "batch_size": args.batch_size, "device": args.device,
        "bootstrap_seed": args.bootstrap_seed, "bootstrap_samples": args.bootstrap_samples,
        "source_sha256": code["source_sha256"],
        "software": report["environment"]["packages"],
        "python": platform.python_version(),
        "pairing_confidence": report["pairing_confidence"],
        "source_manifests": report.get("source_manifests", {})}
    # Source manifest refs may contain current absolute paths; bind content only.
    spec["source_manifests"] = strip_paths(spec["source_manifests"])
    spec_hash = digest(spec)
    manifest_path = output / "experiment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("spec_sha256") != spec_hash or manifest.get("spec") != spec:
            raise ValueError("Resume configuration/evidence differs; use a new output directory")
    else:
        if (output / "scores.sqlite3").exists() or (output / "per_image_scores.csv").exists():
            raise ValueError("Unbound prior results found; use a new output directory")
        manifest = {"schema": SCHEMA, "created_at": now(), "spec": spec,
            "spec_sha256": spec_hash, "git": code, "environment": report["environment"],
            "output_directory": str(output), "sample_list_sha256": spec["samples_sha256"],
            "evaluators": {}, "invocations": [], "status": "initialized"}
    manifest["invocations"].append({"started_at": now(), "command": sys.argv,
                                     "output_directory": str(output), "git": code,
                                     "environment": report["environment"]})
    manifest["status"] = "running"
    atomic_json(manifest_path, manifest)
    rows = [{**{k: r[k] for k in IMAGE_FIELDS if k in r},
             **{field: None for field in METRICS.values()}} for r in images]
    row_map = {image_key(r): r for r in rows}
    db = sqlite3.connect(str(output / "scores.sqlite3"), timeout=1)
    try:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS scores (metric TEXT, image_key TEXT, payload TEXT, sha256 TEXT, PRIMARY KEY(metric,image_key))")
        for metric, key, payload, sha in db.execute("SELECT metric,image_key,payload,sha256 FROM scores"):
            record = json.loads(payload)
            if digest(record) != sha or record["spec_sha256"] != spec_hash:
                raise ValueError("Corrupt/incompatible score checkpoint")
            if metric not in args.metrics or key not in row_map:
                raise ValueError("Unexpected metric/image in checkpoint")
            row = row_map[key]
            if record["image_sha256"] != row["image_sha256"] or record["metric"] != metric or record["image_key"] != key:
                raise ValueError("Score checkpoint image association changed")
            identity = manifest["evaluators"].get(metric)
            if identity is None or record["evaluator_sha256"] != digest(identity):
                raise ValueError("Score checkpoint evaluator identity changed")
            row[METRICS[metric]] = valid_score(metric, record["score"])
        factory = evaluator_factory or Evaluator
        for metric in args.metrics:
            pending = [im for im in images if row_map[image_key(im)][METRICS[metric]] is None]
            if not pending:
                print(f"PASS {metric}: {len(images)} cached image scores validated")
                continue
            evaluator = factory(metric, args)
            try:
                identity = json.loads(canonical(evaluator.identity).decode("utf-8"))
                previous = manifest["evaluators"].get(metric)
                if previous is not None and previous != identity:
                    raise ValueError(f"Evaluator weights/config changed for {metric}; use a new output directory")
                manifest["evaluators"][metric] = identity
                atomic_json(manifest_path, manifest)
                for start in range(0, len(pending), args.batch_size):
                    batch = pending[start:start + args.batch_size]
                    values = evaluator.score(metric, batch)
                    if len(values) != len(batch):
                        raise ValueError("Evaluator returned incorrect number of scores")
                    values = [valid_score(metric, value) for value in values]
                    with db:
                        for item, value in zip(batch, values):
                            key = image_key(item)
                            record = {"spec_sha256": spec_hash, "metric": metric,
                                "image_key": key, "image_sha256": item["image_sha256"],
                                "evaluator_sha256": digest(identity), "score": value}
                            db.execute("INSERT INTO scores VALUES (?,?,?,?)",
                                       (metric, key, canonical(record).decode("utf-8"), digest(record)))
                    for item, value in zip(batch, values):
                        row_map[image_key(item)][METRICS[metric]] = value
                    if start == 0 or start + args.batch_size >= len(pending) or (start // args.batch_size) % 25 == 0:
                        print(f"PASS {metric}: {min(start + args.batch_size, len(pending))}/{len(pending)} pending images scored", flush=True)
                atomic_csv(output / "per_image_scores.csv", rows, IMAGE_FIELDS)
            finally:
                evaluator.close()
        summary = summarize(rows, args.metrics, len(selected_ids), args.bootstrap_seed,
                            args.bootstrap_samples, report["pairing_confidence"])
        summary.update(status="complete", stage=args.command,
                       formal_test_count=len({r["sample_id"] for r in report_images(report, images)}),
                       selected_sample_count=len(selected_ids),
                       preflight_status=report["status"])
        if args.command == "smoke":
            summary["interpretation"] += " Smoke validates software only; do not tune formal configuration from these scores."
        write_outputs(output, rows, summary)
        manifest["status"] = "complete"
        manifest["completed_at"] = now()
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["last_error"] = f"{type(exc).__name__}: {exc}"
        # Durable DB batches survive even a process kill; partial exports are labelled.
        partial = summarize(rows, args.metrics, len(selected_ids), args.bootstrap_seed,
                            args.bootstrap_samples, report["pairing_confidence"])
        partial.update(status=manifest["status"], stage=args.command)
        write_outputs(output, rows, partial)
        raise
    finally:
        db.close()
        manifest["invocations"][-1]["finished_at"] = now()
        atomic_json(manifest_path, manifest)
    print(f"PASS {args.command}: {len(selected_ids)} samples x 4 groups; results: {output}")


def report_images(report, images):
    # IDs/counts are also retained in the full preflight report for smoke runs.
    return [{"sample_id": sid} for sid in report.get("sample_ids", sorted({r["sample_id"] for r in images}))]


def strip_paths(value):
    if isinstance(value, dict):
        return {k: strip_paths(v) for k, v in value.items() if k not in {"path", "image_path"}}
    if isinstance(value, list):
        return [strip_paths(v) for v in value]
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "smoke", "run"):
        p = commands.add_parser(name)
        p.add_argument("--data-root", required=True)
        p.add_argument("--split", choices=["test"], default="test")
        p.add_argument("--dataset-id", default="coco")
        p.add_argument("--split-manifest")
        p.add_argument("--generation-plan")
        p.add_argument("--metadata", help="Original metadata relocated to a new path; original SHA256 is mandatory")
        p.add_argument("--legacy", action="store_true", help="Explicit historical pilot; never equivalent to formal pairing")
        p.add_argument("--output-dir")
        p.add_argument("--metrics", nargs="+", default=list(METRICS), help="musiq-ava laion_aes clip; spaces or commas")
        p.add_argument("--device", default="cuda")
        p.add_argument("--batch-size", type=int, default=1)
        p.add_argument("--bootstrap-seed", type=int, default=1729)
        p.add_argument("--bootstrap-samples", type=int, default=10000)
        if name == "smoke":
            p.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    args.metrics = [part for item in args.metrics for part in item.split(",")]
    if not args.metrics or len(args.metrics) != len(set(args.metrics)) or set(args.metrics) - set(METRICS):
        parser.error("--metrics accepts each of musiq-ava, laion_aes, clip at most once")
    if args.batch_size < 1 or args.bootstrap_samples < 2 or getattr(args, "limit", 1) < 1:
        parser.error("batch-size/limit must be positive and bootstrap-samples >=2")
    return args


def main(argv=None, evaluator_factory=None):
    args = parse_args(argv)
    try:
        output = prepare_output(args)
        with output_lock(output):
            report, images = inspect_data(args)
            report["environment"] = environment()
            report["checked_at"] = now()
            for package in ("torch", "torchvision", "pyiqa"):
                if report["environment"]["packages"].get(package) is None:
                    report["issues"].append({"level": "WARNING", "code": "missing_evaluator_dependency",
                                              "message": f"{package} is not installed"})
            if not report["environment"]["cuda_available"]:
                report["issues"].append({"level": "WARNING", "code": "cuda_unavailable",
                                          "message": "CUDA is unavailable; real default evaluation requires CUDA"})
            if report["status"] == "PASS" and any(i["level"] == "WARNING" for i in report["issues"]):
                report["status"] = "WARNING"
            atomic_json(output / "preflight_report.json", report)
            for issue in report["issues"][:30]:
                print(f"{issue['level']} {issue['code']}: {issue['message']}")
            if len(report["issues"]) > 30:
                print(f"WARNING {len(report['issues']) - 30} further details in preflight_report.json")
            env = report["environment"]
            print(f"{'PASS' if env['cuda_available'] else 'WARNING'} environment: "
                  f"Python {platform.python_version()}, torch={env['packages']['torch']}, "
                  f"torchvision={env['packages']['torchvision']}, pyiqa={env['packages']['pyiqa']}, "
                  f"CUDA={env['cuda_runtime']}, available={env['cuda_available']}")
            for group, counts in report.get("groups", {}).items():
                print(f"{counts['status']} {group}: {counts['valid']}/{counts['expected']} valid, "
                      f"{counts['missing']} missing, {counts['invalid']} invalid")
            print(f"{report['status']} preflight: {len(images)} validated images; report: {output / 'preflight_report.json'}")
            if report["status"] == "FAIL":
                return 2
            if args.command != "preflight":
                run_evaluation(args, report, images, output, evaluator_factory)
        return 0
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("WARNING interrupted; validated completed batches retained for resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
