"""Fail-closed paper protocol manifests. No ML imports or implicit downloads.

SHA256 fields inside JSON documents hash canonical content excluding sha256.
Artifact references always hash the actual file bytes, not their filename.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

PROTOCOL = "paper_v1.0"
GENERATION = "per_sample_v2"
ATTACK_PROTOCOL = "sfw_legacy_attacks_v1"
METRIC_VERSION = "centered_v2_strict_fpr_lt_001"
GEOMETRY = {"capacity": 2048, "center": [10, 54, 10, 54],
            "region": [42, 21], "channel": 3, "feature_dim": 1764,
            "feature_order": "real_then_imag", "fft_dtype": "float32"}
FREEU = {"s1": 0.9, "s2": 0.2, "b1": 1.4, "b2": 1.6}
FORMAL_COUNTS = {"dev": 400, "fit": 1024, "calibration": 2000, "test": 1000}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seal(payload):
    payload = {k: v for k, v in payload.items() if k != "sha256"}
    return {**payload, "sha256": digest(payload)}


def load_document(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("sha256") != seal(value)["sha256"]:
        raise ValueError(f"Invalid/missing document SHA256: {path}")
    return value


def write_once(path, payload):
    """Never replace an existing immutable artifact, including legacy data."""
    path = Path(path)
    value = seal(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Immutable artifact conflict: {path}; use a new output root")
        return value
    # Exclusive creation prevents concurrent writers from silently overwriting.
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return value


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def verify_artifact(ref):
    path = Path(ref["path"])
    if not path.is_file() or file_hash(path) != ref["sha256"]:
        raise ValueError(f"Missing/changed artifact: {path}")
    return path


def git_state(root, require_clean=False):
    def git(*args):
        p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        if p.returncode:
            raise ValueError("Cannot establish git provenance: " + p.stderr.strip())
        return p.stdout.strip()
    state = {"git_commit": git("rev-parse", "HEAD"),
             "git_dirty": bool(git("status", "--porcelain", "--untracked-files=normal"))}
    if require_clean and state["git_dirty"]:
        raise ValueError("Formal execution requires a clean, committed working tree (including untracked files)")
    return state


def environment():
    result = {"python": sys.version}
    for name in ("torch", "torchvision", "diffusers", "transformers", "numpy", "scipy", "scikit-learn"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    if result["torch"]:
        import torch
        result["cuda_runtime"] = torch.version.cuda
        result["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    return result


def model_identity(model_id, revision):
    path = Path(model_id)
    if path.is_dir():
        configs = sorted(p for p in path.rglob("*.json") if p.name in {
            "model_index.json", "config.json", "scheduler_config.json", "tokenizer_config.json"})
        if not configs:
            raise ValueError("Local model has no identifiable model/config JSON files")
        # Hash weights as well: replacing weights at the same local path must invalidate caches.
        weights = sorted(p for p in path.rglob("*") if p.suffix in {".safetensors", ".bin"})
        if not weights:
            raise ValueError("Local model contains no weight files")
        contents = {str(p.relative_to(path)): file_hash(p) for p in configs + weights}
        return {"kind": "local", "path": str(path.resolve()), "files": contents,
                "fingerprint": digest(contents), "claimed_hf_revision": None}
    if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise ValueError("Formal remote models require an immutable 40-character HF commit revision; use a local model path otherwise")
    return {"kind": "huggingface", "model_id": model_id, "revision": revision}


def dataset_records(path, group_field=None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw.get("annotations") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty COCO annotations list")
    keys = sorted(set().union(*(r.keys() for r in rows)))
    if group_field is None:
        # COCO image_id and explicit group_id are semantic identifiers. Never use annotation id.
        group_field = next((k for k in ("image_id", "group_id") if all(k in r for r in rows)), None)
    if not group_field or not all(r.get(group_field) is not None for r in rows):
        raise ValueError(f"No reliable image group identifier. Available fields: {keys}. Confirm one using --group_field; caption/annotation id is not a group id.")
    if group_field in {"id", "caption", "prompt"}:
        raise ValueError("Annotation id/caption/prompt cannot be used as an image group identifier")
    records = []
    for i, row in enumerate(rows):
        prompt = row.get("caption")
        if not isinstance(prompt, str) or not isinstance(row[group_field], (str, int)):
            raise ValueError(f"Invalid caption/group in annotation {i}")
        records.append({"sample_id": i, "group_id": str(row[group_field]),
                        "prompt": prompt, "prompt_hash": digest(prompt)})
    return records, group_field


def validate_splits(manifest, exact_counts=FORMAL_COUNTS):
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Split protocol mismatch")
    seen_ids, seen_groups = set(), set()
    if set(manifest["splits"]) != set(exact_counts):
        raise ValueError("Expected dev/fit/calibration/test splits")
    for name, count in exact_counts.items():
        rows = manifest["splits"][name]
        ids = [r["sample_id"] for r in rows]
        groups = [r["group_id"] for r in rows]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"{name}: wrong count or duplicate sample IDs")
        if name != "dev" and len(set(groups)) != count:
            raise ValueError(f"{name}: repeated source image groups")
        if seen_ids.intersection(ids) or seen_groups.intersection(groups):
            raise ValueError(f"Split leakage involving {name}")
        seen_ids.update(ids)
        seen_groups.update(groups)
    return manifest


def create_splits(dataset, seed=1729, group_field=None, counts=None):
    counts = counts or FORMAL_COUNTS
    rows, field = dataset_records(dataset, group_field)
    dev = rows[:counts["dev"]]
    excluded = {r["group_id"] for r in dev}
    by_group = {}
    for row in rows[counts["dev"]:]:
        if row["group_id"] not in excluded:
            by_group.setdefault(row["group_id"], row)
    candidates = list(by_group.values())
    random.Random(seed).shuffle(candidates)
    required = sum(counts[k] for k in ("fit", "calibration", "test"))
    if len(candidates) < required:
        raise ValueError(f"Need {required} non-dev unique groups, found {len(candidates)}. Do not reduce counts automatically.")
    splits, offset = {"dev": dev}, 0
    for name in ("fit", "calibration", "test"):
        splits[name] = sorted(candidates[offset:offset + counts[name]], key=lambda r: r["sample_id"])
        offset += counts[name]
    return seal(validate_splits({"protocol": PROTOCOL, "seed": seed,
        "dataset": artifact(dataset), "dataset_hash": file_hash(dataset),
        "group_id_source": field, "counts": counts, "splits": splits}, counts))


def validate_split_source(manifest, dataset_path=None):
    """Check group/prompt claims against actual metadata, not just disjoint IDs."""
    validate_splits(manifest)
    path = dataset_path or manifest["dataset"]["path"]
    if file_hash(path) != manifest["dataset_hash"]:
        raise ValueError("Dataset hash differs from split manifest")
    expected = create_splits(path, manifest["seed"], manifest["group_id_source"])
    require_equal(manifest, {"splits": expected["splits"], "counts": expected["counts"]},
                  "Deterministic metadata-derived split")
    return manifest


def generation_plan(splits, base_seed=42):
    validate_splits(splits)
    records = []
    for name in ("fit", "calibration", "test"):
        for row in splits["splits"][name]:
            sid = row["sample_id"]
            # A local deterministic key assignment, shared by G0 and G1.
            key = int(digest(["key_v2", base_seed, sid])[:16], 16) % 2048
            records.append({**row, "split": name, "key_index": key,
                            "latent_seed": base_seed + sid})
    return seal({"protocol": PROTOCOL, "generation_protocol": GENERATION,
                 "split_manifest_hash": splits["sha256"], "dataset_hash": splits["dataset_hash"],
                 "base_seed": base_seed, "seed_scheme": "base_seed + sample_id",
                 "generation_groups": ["G0", "G1"], "records": sorted(records, key=lambda r: r["sample_id"])})


def fit_assignment(rows, kind="W_robust", seed=1729):
    if kind not in {"W_clean", "W_robust"}:
        raise ValueError("Formal fit supports only W_clean or W_robust")
    ids = sorted(r["sample_id"] for r in rows)
    random.Random(seed).shuffle(ids)
    mixture = ["clean", "jpeg", "noise", "cc", "rc", "rot5", "rot15", "rot30"]
    assignments = {}
    for i, sid in enumerate(ids):
        category = "clean" if kind == "W_clean" else mixture[i % 8]
        angle = 0
        if category.startswith("rot"):
            angle = int(category[3:]) * (-1 if (i // 8) % 2 == 0 else 1)
        assignments[str(sid)] = {"name": "rotation_bilinear" if angle else category, "angle": angle}
    return seal({"protocol": PROTOCOL, "kind": kind, "seed": seed,
                 "mixture": ["clean"] if kind == "W_clean" else mixture,
                 "assignments": assignments})


def require_equal(actual, expected, label):
    differences = [k for k, v in expected.items() if actual.get(k) != v]
    if differences:
        raise ValueError(f"{label} incompatible fields: {', '.join(differences)}")


def validate_generation_pairing(g0, g1, plan):
    common = ("generation_protocol", "generation_plan_hash", "model", "steps", "guidance",
              "model_dtype", "fft_dtype", "vae_slicing", "code", "environment", "geometry")
    require_equal(g1, {k: g0[k] for k in common}, "G0/G1 generation")
    if g0["generation_pool"] != "G0" or g1["generation_pool"] != "G1":
        raise ValueError("Incorrect generation pool routing")
    require_equal(g0, {"freeu_enabled": False, "freeu": FREEU}, "G0 FreeU")
    require_equal(g1, {"freeu_enabled": True, "freeu": FREEU}, "G1 FreeU")
    if g0["generation_plan_hash"] != plan["sha256"]:
        raise ValueError("Generation plan mismatch")
    expected = {str(r["sample_id"]): r for r in plan["records"]}
    for pool in (g0, g1):
        if set(pool["records"]) != set(expected):
            missing = sorted(set(expected) - set(pool["records"]), key=int)
            raise ValueError(f"{pool['generation_pool']} missing {len(missing)} IDs: {missing}")
    for sid, row in expected.items():
        a, b = g0["records"][sid], g1["records"][sid]
        for record in (a, b):
            require_equal(record, {k: row[k] for k in ("sample_id", "prompt_hash", "key_index", "latent_seed")}, "Generation sample")
            for kind in ("no_wm", "wm"):
                verify_artifact(record["images"][kind])
                verify_artifact(record["latents"][kind])
                if record[f"{kind}_latent_hash"] != record["latents"][kind]["sha256"]:
                    raise ValueError("Latent hash does not match the referenced artifact")
        for field in ("no_wm_latent_hash", "wm_latent_hash", "key_index", "prompt_hash"):
            if a[field] != b[field]:
                raise ValueError(f"G0/G1 pairing mismatch at {sid}: {field}")
    return True


def validate_diff_record(record, source, generation_hash, config):
    require_equal(record, {"source_image_sha256": file_hash(source),
                          "source_generation_manifest_sha256": generation_hash,
                          "config": config}, "Diff provenance; regenerate into a new run or use --overwrite in diagnostic mode")
    verify_artifact(record["output"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("splits")
    s.add_argument("--metadata", required=True)
    s.add_argument("--group_field")
    s.add_argument("--seed", type=int, default=1729)
    s.add_argument("--output", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--split_manifest", required=True)
    s.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "splits":
        value = create_splits(args.metadata, args.seed, args.group_field)
    else:
        value = generation_plan(load_document(args.split_manifest))
    write_once(args.output, value)
    print(f"Saved immutable {args.command}: {args.output}")


if __name__ == "__main__":
    main()
