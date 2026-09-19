"""Formal-only execution path. Legacy ranges/results are never inferred here."""
from dataclasses import asdict
from pathlib import Path
import json
import time

from paper_protocol import (PROTOCOL, FREEU, GEOMETRY, ATTACK_PROTOCOL, METRIC_VERSION,
    artifact, digest, file_hash, load_document, write_once, verify_artifact,
    validate_splits, require_equal, git_state, environment, model_identity,
    fit_assignment, validate_generation_pairing, validate_diff_record, validate_split_source)


def formal_guard(args):
    from research_config import EXPERIMENTS
    if args.experiment not in {f"Q{i}" for i in range(6)}:
        raise ValueError("D0/D1 are diagnostic-only")
    if args.torch_dtype != "float32":
        raise ValueError("Formal HSQR model and FFT dtype are locked to float32")
    for k, value in FREEU.items():
        if getattr(args, f"freeu_{k}") != value:
            raise ValueError("Formal FreeU parameters are locked: " + str(FREEU))
    if args.oracle_attack_whitening or args.overwrite:
        raise ValueError("Formal execution prohibits oracle whitening and overwrite; use a new run root")
    if args.exact_update != "forward":
        raise ValueError("Formal Exact uses the approved forward-step refinement")
    if args.generation_batch_size < 1:
        raise ValueError("generation_batch_size must be positive")
    if args.stage == "all":
        raise ValueError("Formal stages must be explicit: generate, fit, calibrate, evaluate")
    if not args.split_manifest or not args.generation_plan:
        raise ValueError("Formal execution requires --split_manifest and --generation_plan")
    if args.split_name not in {"fit", "calibration", "test"}:
        raise ValueError("Select --split_name fit/calibration/test")
    for field in ("sample_start", "sample_count", "fit_start", "fit_count", "eval_start", "eval_count"):
        if field in getattr(args, "explicit_arguments", set()):
            raise ValueError(f"Formal execution prohibits --{field}; use split manifests")
    root = Path(args.output_dir).resolve()
    if any(part.lower() in {"dev400", "formal4024", "research_outputs", "outputs"} for part in root.parts):
        raise ValueError("Formal output root must be isolated (recommended paper_v1)")
    repo = Path(__file__).resolve().parent.parent
    code = git_state(repo, require_clean=True)
    if not args.acceptance_report:
        raise ValueError("Formal execution requires --acceptance_report from paper_gate.py --gpu_smoke")
    acceptance = load_document(args.acceptance_report)
    require_equal(acceptance, {"protocol": PROTOCOL, "code": code,
                              "implementation_acceptance": "PASS"}, "Implementation acceptance gate")
    splits = load_document(args.split_manifest)
    validate_splits(splits)
    # Dataset may be mounted elsewhere, but the exact contents must agree.
    dataset_path = args.dataset_metadata or splits["dataset"]["path"]
    if file_hash(dataset_path) != splits["dataset_hash"]:
        raise ValueError("Dataset contents differ from split manifest")
    validate_split_source(splits, dataset_path)
    if args.dataset_id != "coco":
        raise ValueError("Current formal split protocol is COCO-only")
    plan = load_document(args.generation_plan)
    require_equal(plan, {"split_manifest_hash": splits["sha256"],
                        "dataset_hash": splits["dataset_hash"]}, "Generation plan")
    from paper_protocol import generation_plan
    if plan.get("generation_protocol") == "per_sample_v2":
        require_equal(plan, {"records": generation_plan(splits, plan["base_seed"])["records"]}, "Generation sample plan")
    elif plan.get("generation_protocol") == "legacy_batch_v1_frozen":
        if args.stage == "generate":
            raise ValueError("Legacy frozen pools prohibit generation")
        expected = {r["sample_id"]: r for name in ("fit", "calibration", "test") for r in splits["splits"][name]}
        if {r["sample_id"] for r in plan["records"]} != set(expected):
            raise ValueError("Legacy plan does not cover the formal splits")
        for row in plan["records"]:
            require_equal(row, expected[row["sample_id"]], "Legacy plan sample")
    else:
        raise ValueError("Unsupported generation protocol")
    model = model_identity(args.model_id, args.model_revision)
    env = environment()
    require_equal(acceptance, {"environment": env}, "Acceptance environment")
    require_equal(acceptance["gpu_smoke"], {"status": "PASS", "model": model}, "Real-model smoke")
    if any(env.get(k) is None for k in ("torch", "torchvision", "diffusers", "transformers", "numpy", "scipy", "scikit-learn")):
        raise ValueError("Formal ML dependencies unavailable; installation is never automatic")
    args.local_files_only = True
    args.output_dir = str(root)
    write_once(root / "manifests" / "paper_root.json", {"protocol": PROTOCOL,
        "split_manifest_hash": splits["sha256"], "generation_plan_hash": plan["sha256"]})
    return splits, plan, model, code, env


def frozen_pools(args, plan):
    root = Path(args.output_dir)
    pairing = load_document(root / "manifests" / "pairing_validation.json")
    require_equal(pairing, {"status": "PASS", "protocol": PROTOCOL}, "Generation pairing gate")
    if load_document(verify_artifact(pairing["generation_plan"]))["sha256"] != plan["sha256"]:
        raise ValueError("Pairing refers to a different generation plan")
    pools = {g: load_document(verify_artifact(pairing["pools"][g])) for g in ("G0", "G1")}
    if plan.get("generation_protocol") == "legacy_batch_v1_frozen":
        from paper_generation import audit_legacy
        report = audit_legacy(verify_artifact(pairing["legacy_evidence"]))
        if report["status"] != "legacy_batch_v1_frozen":
            raise ValueError("Legacy frozen evidence no longer valid")
    validate_generation_pairing(pools["G0"], pools["G1"], plan)
    for pool in pools.values():
        verify_artifact(pool["patterns"])
    return pools


def base_protocol(args, splits, plan, pool, config, model, code, env):
    from research_config import EXPERIMENTS
    q = EXPERIMENTS[args.experiment]
    require_equal(pool, {"model": model, "freeu": FREEU, "model_dtype": "float32",
                        "fft_dtype": "float32", "steps": 50, "guidance": 7.5,
                        "vae_slicing": True, "geometry": GEOMETRY}, "Generation protocol")
    return {"protocol": PROTOCOL, "experiment": args.experiment,
        "display_name": q.display_name, "generation_pool": q.generation_group,
        "generation_manifest_hash": pool["sha256"], "generation_plan_hash": plan["sha256"],
        "split_manifest_hash": splits["sha256"], "dataset_id": args.dataset_id,
        "dataset_hash": splits["dataset_hash"], "model": model,
        "model_id": args.model_id, "model_revision": args.model_revision,
        "generation_freeu": asdict(q.generation_freeu), "inversion_freeu": asdict(q.inversion_freeu),
        "inversion": config.cache_dict(), "model_dtype": "float32", "fft_dtype": "float32",
        "geometry": GEOMETRY, "attack_protocol": ATTACK_PROTOCOL,
        "metric_version": METRIC_VERSION, "code": code, "environment": env}


def validate_whitening(path, base, splits):
    path = Path(path)
    meta = load_document(path.with_suffix(".meta.json"))
    require_equal(meta, {"base_protocol": base,
        "fit_sample_ids_hash": digest([r["sample_id"] for r in splits["splits"]["fit"]]),
        "fitting_residual_count": len(splits["splits"]["fit"])}, "Whitening model")
    if meta["whitening_model_sha256"] != file_hash(path):
        raise ValueError("Whitening file changed after fitting")
    assignment = load_document(verify_artifact(meta["assignment"]))
    require_equal(assignment, {"sha256": meta["fit_attack_assignment_hash"]}, "Fit assignments")
    if set(assignment["assignments"]) != {str(r["sample_id"]) for r in splits["splits"]["fit"]}:
        raise ValueError("Fit assignment sample set mismatch")
    require_equal(assignment, fit_assignment(splits["splits"]["fit"], assignment["kind"], assignment["seed"]),
                  "Predeclared fit mixture")
    return meta


class Extractor:
    """One-image execution; feature and provenance are cached together."""
    def __init__(self, args, pipe, config, base, pool, profiler):
        self.args, self.pipe, self.config = args, pipe, config
        self.base, self.pool, self.profiler = base, pool, profiler
        self.root = Path(args.output_dir) / args.experiment

    def get(self, row, attack, kind):
        import numpy as np
        import torch
        from PIL import Image
        from research_attacks import apply_attack_pair
        from inversion import invert_image
        from hsqr_metrics import extract_hsqr_feature
        from research_hsqr import _inversion_kwargs
        sid = row["sample_id"]
        original = self.pool["records"][str(sid)]
        refs = original["images"]
        if attack.name == "diff":
            pool_root = Path(self.args.output_dir) / self.base["generation_pool"] / self.args.dataset_id / "HSQR"
            refs = {}
            for image_kind in ("no_wm", "wm"):
                record = load_document(pool_root / "diff_records" / f"{sid}-{image_kind}.json")
                config_doc = load_document(pool_root / "diff_config.json")
                attack_config = {k: v for k, v in config_doc.items() if k != "sha256"}
                validate_diff_record(record, original["images"][image_kind]["path"], self.pool["sha256"], attack_config)
                refs[image_kind] = record["output"]
        paths = {k: verify_artifact(v) for k, v in refs.items()}
        with Image.open(paths["no_wm"]) as im:
            no = im.convert("RGB")
        with Image.open(paths["wm"]) as im:
            wm = im.convert("RGB")
        # Reproduce baseline random-attack pair ordering even when only one
        # member needs inversion, so cache state cannot change attack pixels.
        no, wm = apply_attack_pair(no, wm, attack.name, angle=attack.angle,
                                  seed=42 + sid, device=self.args.device)
        image = no if kind == "no_wm" else wm
        import hashlib
        pixels_hash = hashlib.sha256(image.tobytes()).hexdigest()
        key = {"base": self.base, "sample_id": sid, "image_kind": kind,
               "attack": attack.slug, "attack_parameters": attack.parameters,
               "source_hash": refs[kind]["sha256"], "attacked_pixels_hash": pixels_hash,
               "execution": "single_image_v2"}
        cache_path = self.root / "features" / f"{digest(key)}.json"
        self.profiler.record_requested_images(1)
        if self.args.cache_policy == "reuse" and cache_path.exists():
            stored = load_document(cache_path)
            require_equal(stored, {"key": key}, "Feature cache")
            self.profiler.record_cache_hits(1)
            return np.array(stored["feature"], dtype=np.float32), {**stored["provenance"], "cache_status": "hit"}
        info = {"sample_id": sid, "image_kind": kind, "attack": attack.slug,
                "experiment": self.args.experiment, "cache_status": "miss",
                "source_image_sha256": refs[kind]["sha256"], "feature_hash": digest(key)}
        provenance = []
        calls = self.profiler.unet_forward_calls
        started = self.profiler.start_inversion()
        try:
            z = invert_image(self.pipe, image, method=self.config.method,
                             provenance=provenance, **_inversion_kwargs(self.config))
            info.update(provenance[0] if provenance else {"inversion_success": True,
                "converged": None, "failure_reason": None, "timesteps": self.config.num_inference_steps})
            feature = extract_hsqr_feature(z)[0].detach().cpu().numpy()
            if not np.isfinite(feature).all():
                raise ValueError("non_finite_feature")
        except Exception as exc:
            info.update(inversion_success=False, failure_reason=f"{type(exc).__name__}: {exc}")
            feature = None
        finally:
            self.profiler.finish_inversion(started)
            self.profiler.record_inversion(self.config.method, 1, self.config.num_inference_steps, calls)
        if not info["inversion_success"]:
            write_once(self.root / "failures" / f"{digest(key)}.json", {"key": key, "provenance": info})
        elif self.args.cache_policy == "reuse":
            write_once(cache_path, {"key": key, "feature": feature.tolist(), "provenance": info})
        return feature, info


def diff_attack(args, splits, pool, code, env):
    from paper_diff_attack import run
    return run(args, splits, pool, code, env)


def run(args):
    splits, plan, model_id, code, env = formal_guard(args)
    if args.stage == "generate":
        from paper_generation import generate
        return generate(args, splits, plan, model_id, code, env)
    pools = frozen_pools(args, plan)
    from research_config import EXPERIMENTS
    pool = pools[EXPERIMENTS[args.experiment].generation_group]
    if args.stage == "diff_attack":
        return diff_attack(args, splits, pool, code, env)
    if args.stage not in {"fit", "calibrate", "evaluate"}:
        raise ValueError("Use paper_export.py with --run_manifest for formal summary/export")
    required_split = {"fit": "fit", "calibrate": "calibration", "evaluate": "test"}[args.stage]
    if args.split_name != required_split:
        raise ValueError(f"{args.stage} only accepts split {required_split}")
    from research_hsqr import _inversion_config, _load_pipe, _attack_specs, AttackSpec
    from hsqr_metrics import HSQRDistanceModel, hsqr_reference_feature
    from runtime_profiling import RuntimeProfiler
    import numpy as np
    import torch
    config = _inversion_config(args)
    base = base_protocol(args, splits, plan, pool, config, model_id, code, env)
    qdir = Path(args.output_dir) / args.experiment
    rows = splits["splits"][args.split_name]
    args.expected_runtime_images = len(rows) * (2 * len(_attack_specs(args)) if args.stage == "evaluate" else 1)
    path = Path(args.whitening_model) if args.whitening_model else qdir / "models" / f"{args.whitening_name}-{digest(base)[:16]}.npz"
    if path.suffix != ".npz":
        raise ValueError("Formal whitening model paths must end in .npz")
    thresholds = None
    if args.stage != "fit":
        validate_whitening(path, base, splits)
    if args.stage == "evaluate":
        if not args.threshold_model:
            raise ValueError("Formal test requires --threshold_model from independent calibration")
        thresholds = load_document(args.threshold_model)
        require_equal(thresholds, {"base_protocol": base, "whitening_model_sha256": file_hash(path),
            "calibration_sample_ids_hash": digest([r["sample_id"] for r in splits["splits"]["calibration"]]),
            "calibration_attack": "clean"}, "Frozen thresholds")
        verify_artifact(thresholds["raw_calibration"])
        # Required attacks are checked before loading the diffusion model.
        selected_attacks = _attack_specs(args)
        args.validated_diff_records = []
        args.validated_diff_config = None
        if any(a.name == "diff" for a in selected_attacks):
            pool_dir = Path(args.output_dir) / base["generation_pool"] / args.dataset_id / "HSQR"
            config_doc = load_document(pool_dir / "diff_config.json")
            args.validated_diff_config = {k: v for k, v in config_doc.items() if k != "sha256"}
            for row in rows:
                for kind in ("no_wm", "wm"):
                    rp = pool_dir / "diff_records" / f"{row['sample_id']}-{kind}.json"
                    record = load_document(rp)
                    validate_diff_record(record, pool["records"][str(row["sample_id"])]["images"][kind]["path"],
                                         pool["sha256"], args.validated_diff_config)
                    args.validated_diff_records.append(artifact(rp))
    profiler = RuntimeProfiler(args.device)
    profiler.start_model_load()
    pipe = _load_pipe(args)
    profiler.finish_model_load()
    # Explicit fixed warmup before measurement. Never use test/calibration images.
    if args.runtime_benchmark:
        if args.cache_policy != "no_cache":
            raise ValueError("Formal runtime benchmark requires --cache_policy no_cache")
        from PIL import Image
        from inversion import invert_image
        from research_hsqr import _inversion_kwargs
        invert_image(pipe, Image.new("RGB", (512, 512), (127, 127, 127)),
                     method=config.method, **_inversion_kwargs(config))
    profiler.attach_unet(pipe.unet)
    profiler.start_processing()
    extractor = Extractor(args, pipe, config, base, pool, profiler)
    patterns = torch.load(verify_artifact(pool["patterns"]), map_location="cpu", weights_only=True)
    references = hsqr_reference_feature(patterns).numpy()
    if references.shape != (2048, 1764):
        raise ValueError("Expected all 2048 HSQR reference keys")
    try:
        if args.stage == "fit":
            if path.exists() or path.with_suffix(".meta.json").exists():
                raise ValueError("Refuse to overwrite whitening model")
            assignment = fit_assignment(rows, args.whitening_name)
            apath = qdir / "models" / f"assignment-{assignment['sha256']}.json"
            write_once(apath, assignment)
            features, targets = [], []
            for row in rows:
                spec = assignment["assignments"][str(row["sample_id"]) ]
                f, info = extractor.get(row, AttackSpec(spec["name"], spec["angle"]), "wm")
                if not info["inversion_success"]:
                    raise ValueError(f"Fit inversion failed: {info}")
                features.append(f)
                targets.append(references[pool["records"][str(row["sample_id"])]["key_index"]])
            fitted = HSQRDistanceModel().fit(features, targets)
            path.parent.mkdir(parents=True, exist_ok=True)
            fitted.save(path)
            write_once(path.with_suffix(".meta.json"), {"base_protocol": base,
                "fit_sample_ids_hash": digest([r["sample_id"] for r in rows]),
                "fit_attack_assignment_hash": assignment["sha256"], "assignment": artifact(apath),
                "fitting_residual_count": len(rows), "residual_mean": fitted.residual_mean.tolist(),
                "shrinkage": fitted.shrinkage, "jitter": fitted.jitter,
                "whitening_model_sha256": file_hash(path)})
            print(f"WHITENING_MODEL={path}")
        elif args.stage == "calibrate":
            from paper_statistics import METRICS, calibrate_clean_negatives
            fitted = HSQRDistanceModel.load(path)
            records, scores = [], {m: [] for m in METRICS}
            for row in rows:
                f, info = extractor.get(row, AttackSpec("clean"), "no_wm")
                if not info["inversion_success"]:
                    raise ValueError(f"Calibration inversion failed: {info}")
                key = pool["records"][str(row["sample_id"])]["key_index"]
                distances = {m: float(v[0, 0]) for m, v in fitted.distances(f, references[key:key+1]).items()}
                records.append({**row, "claimed_key": key, "distances": distances, "provenance": info})
                for m in scores:
                    scores[m].append(distances[m])
            target = qdir / "calibration" / f"thresholds-{digest([base, file_hash(path)])[:16]}.json"
            raw = target.with_name(target.stem + "-raw.json")
            write_once(raw, {"base_protocol": base, "split": "calibration", "records": records})
            write_once(target, {"base_protocol": base, "whitening_model_sha256": file_hash(path),
                "calibration_sample_ids_hash": digest([r["sample_id"] for r in rows]),
                "calibration_attack": "clean", "raw_calibration": artifact(raw),
                "thresholds": {m: {**calibrate_clean_negatives(scores[m]), "metric": m,
                    "experiment": args.experiment, "protocol_hash": digest(base)} for m in scores}})
            print(f"THRESHOLD_MODEL={target}")
        else:
            evaluate(args, rows, pool, base, path, thresholds, extractor, references, _attack_specs(args))
    finally:
        profiler.finish_processing()
        profiler.close()
        runtime_path = profiler.save(qdir / "runtime", {"base_protocol": base, "stage": args.stage,
            "cache_policy": args.cache_policy, "warmup_policy": "one_synthetic_gray_image" if args.runtime_benchmark else "none_not_benchmark",
            "paper_runtime_eligible": bool(args.runtime_benchmark and profiler.num_images > 0
                                           and profiler.cache_hit_images == 0
                                           and profiler.requested_images == profiler.num_images
                                           and profiler.num_images == args.expected_runtime_images),
            "batch_size": 1, "gpu": env.get("gpu"), "torch": env["torch"],
            "cuda_runtime": env.get("cuda_runtime"), "dtype": "float32"})
        if hasattr(args, "completed_run"):
            manifest_path, manifest = args.completed_run
            if manifest_path.exists():
                previous = load_document(manifest_path)
                require_equal(previous, manifest, "Existing immutable run")
                for ref in previous["runtime"]:
                    verify_artifact(ref)
            else:
                write_once(manifest_path, {**manifest, "runtime": [artifact(runtime_path)]})
            print(f"RUN_MANIFEST={manifest_path}")


def evaluate(args, rows, pool, base, whitening_path, thresholds, extractor, references, attacks):
    import numpy as np
    from hsqr_metrics import HSQRDistanceModel
    from paper_statistics import METRICS, coverage
    fitted = HSQRDistanceModel.load(whitening_path)
    prepared = fitted.prepare_references(references)
    signature_payload = {**base, "split": "test", "test_sample_ids_hash": digest([r["sample_id"] for r in rows]),
        "whitening_model_sha256": file_hash(whitening_path), "threshold_model_sha256": file_hash(args.threshold_model),
        "diff_provenance_hash": digest(args.validated_diff_records)}
    signature = digest(signature_payload)
    run_id = digest([signature, [a.slug for a in attacks]])[:24]
    root = Path(args.output_dir) / args.experiment / "runs" / run_id
    request = {"protocol": PROTOCOL, "protocol_signature": signature, "signature_payload": signature_payload,
        "run_id": run_id, "requested_attacks": [a.slug for a in attacks],
        "sample_ids": [r["sample_id"] for r in rows], "sample_groups": {str(r["sample_id"]): r["group_id"] for r in rows},
        "thresholds": artifact(args.threshold_model), "whitening": artifact(whitening_path),
        "whitening_metadata": artifact(Path(whitening_path).with_suffix(".meta.json")),
        "split_manifest": artifact(args.split_manifest), "generation_plan": artifact(args.generation_plan),
        "pairing_validation": artifact(Path(args.output_dir) / "manifests" / "pairing_validation.json")}
    request["acceptance_report"] = artifact(args.acceptance_report)
    request["diff_records"] = args.validated_diff_records
    request["diff_config"] = args.validated_diff_config
    write_once(root / "request.json", request)
    raw_refs = {}
    for attack in attacks:
        out = root / f"raw-{attack.slug}.json"
        if out.exists():
            old = load_document(out)
            require_equal(old, {"protocol_signature": signature, "sample_ids": request["sample_ids"]}, "Existing raw result")
            verify_artifact(old["npz"])
            if args.runtime_benchmark:
                # A resumed benchmark must measure the full cohort, not just
                # the uncompleted attacks. Keep locked raw files untouched.
                by_id = {r["sample_id"]: r for r in old["records"]}
                for row in rows:
                    key = pool["records"][str(row["sample_id"])]["key_index"]
                    for kind in ("no_wm", "wm"):
                        feature, info = extractor.get(row, attack, kind)
                        previous = by_id[row["sample_id"]][kind]
                        if not info["inversion_success"] or not previous["inversion_success"]:
                            raise ValueError("Cannot certify a resumed benchmark with failed inversion")
                        scores = fitted.distances(feature, references[key:key+1])
                        candidates = fitted.distances_to_prepared(feature, prepared)
                        for metric in METRICS:
                            if not np.isfinite(candidates[metric]).all():
                                raise ValueError("Non-finite resumed candidate scores")
                            if not np.isclose(float(scores[metric][0, 0]), previous["distances"][metric], rtol=1e-5, atol=1e-6):
                                raise ValueError("Resumed inversion scores differ from locked raw data")
                            if int(np.argmin(candidates[metric][0])) != previous["predicted"][metric]:
                                raise ValueError("Resumed identification differs from locked raw data")
            raw_refs[attack.slug] = artifact(out)
            continue
        records = []
        for row in rows:
            key = pool["records"][str(row["sample_id"])]["key_index"]
            result = {"sample_id": row["sample_id"], "group_id": row["group_id"],
                      "experiment": args.experiment, "attack": attack.slug, "gt_key": key,
                      "protocol_signature": signature}
            for kind in ("no_wm", "wm"):
                info = {"sample_id": row["sample_id"], "image_kind": kind,
                        "attack": attack.slug, "experiment": args.experiment,
                        "cache_status": "not_reached"}
                try:
                    f, info = extractor.get(row, attack, kind)
                    distances, predicted = {}, {}
                    if info["inversion_success"]:
                        distances = {m: float(v[0, 0]) for m, v in fitted.distances(f, references[key:key+1]).items()}
                        # This candidate search has no ground-truth argument.
                        candidates = fitted.distances_to_prepared(f, prepared)
                        predicted = {m: int(np.argmin(v[0])) for m, v in candidates.items()}
                        if not all(np.isfinite(v).all() for v in candidates.values()):
                            raise ValueError("Non-finite candidate distance")
                        if not all(np.isfinite(v) for v in distances.values()):
                            raise ValueError("Non-finite claimed-key distance")
                    result[kind] = {**info, "distances": distances, "predicted": predicted}
                except Exception as exc:
                    result[kind] = {**info, "inversion_success": False, "failure_reason": f"{type(exc).__name__}: {exc}",
                                    "distances": {}, "predicted": {}}
            records.append(result)
        arrays = {"sample_ids": np.array(request["sample_ids"]),
                  "protocol_signature": np.array(signature), "attack": np.array(attack.slug)}
        for m in METRICS:
            for kind in ("no_wm", "wm"):
                arrays[f"{kind}_{m}"] = np.array([r[kind]["distances"].get(m, np.nan) for r in records])
            arrays[f"id_correct_{m}"] = np.array([r["wm"]["predicted"].get(m) == r["gt_key"] for r in records])
        npz = out.with_suffix(".npz")
        if npz.exists():
            raise ValueError(f"Unmanifested raw NPZ exists: {npz}; preserve it and use a new run root")
        with npz.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        write_once(out, {"protocol_signature": signature, "sample_ids": request["sample_ids"],
            "attack": {"name": attack.name, "slug": attack.slug, "angle": attack.angle, "parameters": attack.parameters},
            "records": records, "coverage": coverage(records)[0], "npz": artifact(npz)})
        raw_refs[attack.slug] = artifact(out)
    manifest = {**request, "raw_results": raw_refs}
    args.completed_run = (root / "run_manifest.json", manifest)
    return manifest
