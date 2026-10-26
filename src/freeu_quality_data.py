"""Read-only image/provenance audit for FreeU quality evaluation.

This deliberately does not import the generation runner, utils, or a detector.
Latent hashes are checked as recorded pairing claims; latent files are not read.
"""
import json
from pathlib import Path, PurePosixPath
import re

from paper_protocol import (PROTOCOL, GENERATION, FREEU, GEOMETRY,
    FORMAL_COUNTS, digest, file_hash, load_document, seal, generation_plan,
    validate_splits, validate_split_source)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _inside(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _inspect_data(args):
    """Return a JSON-ready report and audited image rows, never write inputs.

    A FAIL report is diagnostic only: callers must not score a partial subset.
    Explicit legacy mode still uses the sealed formal test membership. It cannot
    repair seals or suppress a mismatch in evidence that actually exists.
    """
    root = Path(args.data_root).resolve()
    split_name = getattr(args, "split", "test")
    dataset = getattr(args, "dataset_id", "coco")
    legacy = bool(getattr(args, "legacy", False))
    issues, images, sources = [], [], {}
    report = {"status": "FAIL", "data_root": str(root), "split": split_name,
        "legacy": legacy, "issues": issues, "groups": {},
        "pairing_confidence": "legacy_unverified" if legacy else "formal_image_verified_latent_hash_claims",
        "source_manifests": sources, "sample_list_sha256": None,
        "sample_ids": [], "relocated_images": 0,
        "verification_counts": {"generation_records": {"G0": 0, "G1": 0},
                                "paired_samples": 0},
        "evidence_scope": "PNG bytes and recorded pairing claims; latent files are not re-read"}

    def issue(level, code, message, **detail):
        issues.append({"level": level, "code": code, "message": str(message), **detail})

    def finish():
        report["status"] = ("FAIL" if any(x["level"] == "FAIL" for x in issues)
            else "WARNING" if any(x["level"] == "WARNING" for x in issues) else "PASS")
        report["provenance_status"] = report["status"]
        for group in report["groups"].values():
            group["image_valid"] = group["valid"]
            group["image_status"] = ("FAIL" if group["missing"] or group["invalid"] else "PASS")
            group["provenance_status"] = report["status"]
            if report["status"] == "FAIL":
                group["status"] = "FAIL"
        return report, images

    if split_name != "test":
        issue("FAIL", "unsupported_split", "This experiment requires the formal test split")
        return finish()
    if not root.is_dir():
        issue("FAIL", "missing_root", f"Data root does not exist: {root}")
        return finish()
    if not isinstance(dataset, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", dataset):
        issue("FAIL", "dataset_id", "dataset_id must be a simple directory name")
        return finish()
    try:
        from PIL import Image
    except ImportError as exc:
        issue("FAIL", "pillow_missing", exc)
        return finish()

    def document(path, label):
        value = load_document(path)
        sources[label] = {"path": str(path.resolve()), "sha256": file_hash(path),
                          "document_sha256": value["sha256"]}
        return value

    try:
        sp = Path(getattr(args, "split_manifest", None) or root / "manifests/formal_splits.json")
        splits = document(sp, "split_manifest")
        validate_splits(splits)
        if splits.get("counts") != FORMAL_COUNTS:
            raise ValueError("Split counts must match formal dev/fit/calibration/test counts")
        if (not isinstance(splits["dataset"]["path"], str)
                or not _sha(splits["dataset"]["sha256"])
                or splits["dataset"]["sha256"] != splits["dataset_hash"]):
            raise ValueError("Invalid or inconsistent source metadata artifact reference")
        for split_rows in splits["splits"].values():
            for row in split_rows:
                if (type(row["sample_id"]) is not int or row["sample_id"] < 0
                        or not isinstance(row.get("prompt"), str)
                        or row.get("prompt_hash") != digest(row["prompt"])):
                    raise ValueError("Invalid sample ID or prompt hash in split manifest")
        test_rows = sorted(splits["splits"][split_name], key=lambda x: x["sample_id"])
        ids = [r["sample_id"] for r in test_rows]
        report["sample_ids"] = ids
        report["sample_list_sha256"] = digest(ids)
        report["verification_counts"]["expected_samples"] = len(ids)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issue("FAIL", "split_manifest", exc)
        return finish()

    for pool in ("G0", "G1"):
        for kind in ("no_wm", "wm"):
            report["groups"][f"{pool}/{kind}"] = {"expected": len(ids), "present": 0,
                "missing": 0, "invalid": 0, "valid": 0, "status": "FAIL"}

    # Explicit metadata paths are user-selected. Automatic relocation never
    # follows an arbitrary absolute path embedded by another machine.
    metadata_arg = getattr(args, "metadata", None)
    metadata_candidates = [Path(metadata_arg)] if metadata_arg else [
        root / "src/text_dataset" / dataset / "meta_data.json",
        root / "text_dataset" / dataset / "meta_data.json",
        Path(__file__).resolve().parent / "text_dataset" / dataset / "meta_data.json"]
    original = Path(splits["dataset"]["path"])
    if _inside(original, root):
        metadata_candidates.append(original)
    metadata = next((p for p in metadata_candidates if p.is_file()), None)
    if metadata is None:
        issue("WARNING" if legacy else "FAIL", "missing_metadata",
              "Cannot independently verify split membership against source metadata; provide --metadata",
              searched=[str(p) for p in metadata_candidates])
    else:
        try:
            if splits["dataset"]["sha256"] != splits["dataset_hash"]:
                raise ValueError("Split dataset reference and dataset_hash disagree")
            validate_split_source(splits, metadata)
            sources["dataset_metadata"] = {"path": str(metadata.resolve()), "sha256": file_hash(metadata)}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue("FAIL", "metadata_mismatch", exc)

    plan = None
    try:
        pp = Path(getattr(args, "generation_plan", None) or root / "manifests/generation_plan.json")
        plan = document(pp, "generation_plan")
        expected_plan = generation_plan(splits, plan["base_seed"])
        if plan != expected_plan:
            raise ValueError("Generation plan is not the deterministic per_sample_v2 plan for these splits")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issue("FAIL", "generation_plan", exc)
        return finish()
    planned = {str(r["sample_id"]): r for r in plan["records"]}
    manifests, directories, configs = {}, {}, {}
    for pool in ("G0", "G1"):
        candidates = [root / pool / dataset / "HSQR", root / pool / "HSQR", root / pool]
        found = [p for p in candidates if p.is_dir() and (
            (p / "img_pil").is_dir() or (p / "img_pil_wm").is_dir()
            or (p / "generation_manifest.json").exists())]
        if len(found) > 1:
            issue("FAIL", "ambiguous_pool", f"Multiple {pool} image pool roots", paths=[str(p) for p in found])
        directory = found[0] if len(found) == 1 else candidates[0]
        directories[pool] = directory
        if not directory.is_dir() or not _inside(directory, root):
            issue("FAIL", "pool_directory", f"Missing or unsafe {pool} directory: {directory}")
        manifest_path = directory / "generation_manifest.json"
        if not manifest_path.is_file():
            issue("WARNING" if legacy else "FAIL", "missing_generation_manifest",
                  f"Missing frozen generation manifest: {manifest_path}", generation_pool=pool)
            continue
        try:
            manifest = document(manifest_path, pool)
            expected_config = {"protocol": PROTOCOL, "generation_protocol": GENERATION,
                "generation_plan_hash": plan["sha256"], "generation_pool": pool,
                "freeu_enabled": pool == "G1", "freeu": FREEU,
                "steps": 50, "guidance": 7.5, "model_dtype": "float32",
                "fft_dtype": "float32", "vae_slicing": True, "geometry": GEOMETRY}
            for key, value in expected_config.items():
                if manifest.get(key) != value:
                    raise ValueError(f"{pool} generation config mismatch: {key}")
            for key in ("model", "code", "environment", "batch_size", "records", "patterns"):
                if key not in manifest:
                    raise ValueError(f"{pool} generation manifest missing {key}")
            if not isinstance(manifest["records"], dict) or set(manifest["records"]) != set(planned):
                raise ValueError(f"{pool} frozen records do not cover the complete generation plan")
            config = {k: v for k, v in manifest.items() if k not in {"sha256", "records", "patterns"}}
            if (directory / "generation_config.json").exists():
                saved = document(directory / "generation_config.json", f"{pool}_config")
                if {k: v for k, v in saved.items() if k != "sha256"} != config:
                    raise ValueError(f"{pool} frozen manifest differs from generation_config.json")
            manifests[pool], configs[pool] = manifest, config
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue("FAIL", "generation_manifest", exc, generation_pool=pool)

    if len(manifests) == 2:
        common = ("generation_protocol", "generation_plan_hash", "model", "steps", "guidance",
                  "model_dtype", "fft_dtype", "vae_slicing", "code", "environment", "geometry")
        mismatch = [k for k in common if manifests["G0"][k] != manifests["G1"][k]]
        if mismatch:
            issue("FAIL", "pool_pairing", "G0/G1 configuration differs", fields=mismatch)
    # Recompute pairing from the frozen pools rather than trust a PASS marker.
    # If a historical marker exists, its artifact claims must also remain true
    # after relocation; absolute paths are never followed to bypass local files.
    pairing_path = root / "manifests/pairing_validation.json"
    if pairing_path.exists():
        try:
            pairing = document(pairing_path, "pairing_validation")
            if (pairing.get("protocol") != PROTOCOL or pairing.get("status") != "PASS"
                    or pairing.get("n_samples") != len(planned)):
                raise ValueError("Pairing marker protocol/status/count mismatch")
            if pairing["generation_plan"]["sha256"] != sources["generation_plan"]["sha256"]:
                raise ValueError("Pairing marker references a different generation plan file")
            for pool in ("G0", "G1"):
                if pairing["pools"][pool]["sha256"] != sources[pool]["sha256"]:
                    raise ValueError(f"Pairing marker references a different {pool} manifest file")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue("FAIL", "pairing_marker", exc)
    if legacy:
        issue("WARNING", "legacy_not_formal",
              "Explicit legacy analysis: image names cannot establish prompt/seed/key or latent pairing. Results are not equivalent to the formal protocol.")

    signature = None
    for row in test_rows:
        sid, pair = row["sample_id"], {}
        for pool in ("G0", "G1"):
            manifest = manifests.get(pool)
            record = manifest["records"].get(str(sid)) if manifest else None
            record_ok = False
            if record:
                try:
                    if record.get("sha256") != seal(record)["sha256"]:
                        raise ValueError("Invalid nested generation record seal")
                    expected_record = {**planned[str(sid)], "config_hash": digest(configs[pool])}
                    for key, value in expected_record.items():
                        if record.get(key) != value:
                            raise ValueError(f"Generation record mismatch: {key}")
                    for kind in ("no_wm", "wm"):
                        lh = record[f"{kind}_latent_hash"]
                        if not _sha(lh) or lh != record["latents"][kind]["sha256"]:
                            raise ValueError(f"Invalid {kind} recorded latent hash claim")
                    pair[pool] = record
                    record_ok = True
                    report["verification_counts"]["generation_records"][pool] += 1
                except (ValueError, KeyError, TypeError) as exc:
                    issue("FAIL", "generation_record", exc, sample_id=sid, generation_pool=pool)
            for kind, subdir in (("no_wm", "img_pil"), ("wm", "img_pil_wm")):
                group = report["groups"][f"{pool}/{kind}"]
                path = directories[pool] / subdir / f"{sid}.png"
                if not path.is_file():
                    group["missing"] += 1
                    issue("FAIL", "missing_image", f"Missing image: {path}", sample_id=sid,
                          generation_pool=pool, watermark_status=kind)
                    continue
                group["present"] += 1
                try:
                    if not _inside(path, root):
                        raise ValueError("Image path escapes data root (possibly a symlink)")
                    actual_hash = file_hash(path)
                    if record:
                        reference = record["images"][kind]
                        old = PurePosixPath(reference["path"].replace("\\", "/"))
                        expected_tail = (pool, dataset, "HSQR", subdir, f"{sid}.png")
                        if old.parts[-5:] != expected_tail:
                            raise ValueError("Recorded image path has a different pool, sample, dataset, or image kind")
                        if not _sha(reference["sha256"]) or reference["sha256"] != actual_hash:
                            raise ValueError("Image SHA256 differs from generation record")
                        if str(path.resolve()).replace("\\", "/") != str(old):
                            report["relocated_images"] += 1
                    elif not legacy:
                        raise ValueError("No verifiable formal generation record for image")
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        im.load()
                        current = (im.format, im.size, im.mode)
                        if im.format != "PNG" or im.mode != "RGB":
                            raise ValueError(f"Expected RGB PNG, found {im.format}/{im.mode}")
                        if signature is None:
                            signature = current
                        if current != signature:
                            raise ValueError(f"Image format/size/mode differs: {current} vs {signature}")
                    if manifest and not record_ok:
                        raise ValueError("Generation record failed provenance validation")
                    group["valid"] += 1
                    images.append({"sample_id": sid, "generation_pool": pool,
                        "watermark_status": kind, "image_path": str(path.resolve()),
                        "image_sha256": actual_hash, "prompt": row["prompt"],
                        "seed": record["latent_seed"] if record_ok else None,
                        "key_index": record["key_index"] if record_ok else None})
                except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
                    group["invalid"] += 1
                    issue("FAIL", "invalid_image", exc, sample_id=sid,
                          generation_pool=pool, watermark_status=kind, image_path=str(path))
        if len(pair) == 2:
            paired_ok = True
            for field in ("no_wm_latent_hash", "wm_latent_hash", "key_index", "prompt_hash", "latent_seed"):
                if pair["G0"][field] != pair["G1"][field]:
                    paired_ok = False
                    issue("FAIL", "sample_pairing", f"G0/G1 pairing mismatch: {field}", sample_id=sid)
            if paired_ok:
                report["verification_counts"]["paired_samples"] += 1
    if signature:
        report["image_spec"] = {"format": signature[0], "size": list(signature[1]), "mode": signature[2]}
    for group in report["groups"].values():
        group["status"] = "FAIL" if group["missing"] or group["invalid"] else "WARNING" if legacy else "PASS"
    report["pairing_evidence"] = {
        "prompt": "sealed split + deterministic generation plan + generation records when available",
        "seed": "latent_seed matched to generation plan for each verified generation record; null otherwise",
        "key_index": "matched to deterministic key assignment for each verified generation record; null otherwise",
        "latent_hashes": "cross-pool recorded hashes compared; latent files are not accessed",
        "legacy_warning": legacy}
    return finish()


def inspect_data(args):
    """Fail closed with a report even for structurally malformed input JSON."""
    try:
        return _inspect_data(args)
    except Exception as exc:
        # These are data validation failures, never an invitation to run a
        # reduced subset. Keep a usable report instead of a traceback-only exit.
        return {"status": "FAIL", "provenance_status": "FAIL",
            "data_root": str(getattr(args, "data_root", "")),
            "split": getattr(args, "split", "test"),
            "legacy": bool(getattr(args, "legacy", False)),
            "pairing_confidence": "unverified",
            "issues": [{"level": "FAIL", "code": "malformed_evidence",
                        "message": f"{type(exc).__name__}: {exc}"}],
            "groups": {}, "sample_ids": [], "sample_list_sha256": None,
            "source_manifests": {}}, []
