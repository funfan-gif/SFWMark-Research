"""Original Diff attack, with immutable input/output provenance per image.

The original attacker draws Gaussian noise from the global torch RNG. A scoped
fork_rng plus per-call seed makes its original batch procedure reproducible
without leaking state. Formal resume recomputes a complete original 8-image
batch into a temporary directory, then verifies any already committed outputs.
It never changes the attack's noise step, model dtype, strength or procedure.
"""
from pathlib import Path
import tempfile
import shutil

from paper_protocol import (PROTOCOL, artifact, digest, file_hash, load_document,
    write_once, require_equal, verify_artifact, model_identity, validate_diff_record)


def run(args, splits, pool, code, env):
    import torch
    from diff_attack.attdiffusion import ReSDPipeline, DiffWMAttacker
    from runtime_profiling import RuntimeProfiler
    rows = splits["splits"][args.split_name]
    pool_dir = Path(args.output_dir) / pool["generation_pool"] / args.dataset_id / "HSQR"
    identity = model_identity(args.diff_attack_model_id, args.diff_attack_model_revision)
    config = {"protocol": "original_diff_fp16_provenance_v2", "model": identity,
        "noise_step": 60, "batch_size": 8, "dtype": "float16", "steps": 50,
        "guidance": 7.5, "captions": {}, "seed_policy": "scoped_seed_1024_plus_first_sample_id",
        "code": code, "environment": env}
    write_once(pool_dir / "diff_config.json", config)
    profiler = RuntimeProfiler(args.device)
    pipe = None
    # Fixed manifest ordering, NOT ordering of missing files. Resume never shifts a batch.
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        for kind in ("no_wm", "wm"):
            records = []
            for row in batch:
                sid = row["sample_id"]
                source = verify_artifact(pool["records"][str(sid)]["images"][kind])
                rp = pool_dir / "diff_records" / f"{sid}-{kind}.json"
                output = pool_dir / ("img_pil-diffatt_fp16" if kind == "no_wm" else "img_pil_wm-diffatt_fp16") / f"{sid}.png"
                if rp.exists():
                    validate_diff_record(load_document(rp), source, pool["sha256"], config)
                elif output.exists():
                    raise ValueError(f"Unmanifested old Diff image cannot be used: {output}")
                records.append((sid, source, rp, output))
            profiler.record_requested_images(len(records))
            if all(rp.exists() for _, _, rp, _ in records):
                profiler.record_cache_hits(len(records))
                continue
            if pipe is None:
                profiler.start_model_load()
                kwargs = {"torch_dtype": torch.float16, "local_files_only": True}
                if args.diff_attack_model_revision and identity["kind"] != "local":
                    kwargs["revision"] = args.diff_attack_model_revision
                pipe = ReSDPipeline.from_pretrained(args.diff_attack_model_id, **kwargs).to(args.device)
                pipe.set_progress_bar_config(disable=True)
                attacker = DiffWMAttacker(pipe, batch_size=8, noise_step=60, captions={})
                profiler.finish_model_load()
                profiler.attach_unet(pipe.unet)
                profiler.start_processing()
            with tempfile.TemporaryDirectory(prefix="sfw-diff-") as temporary:
                targets = [Path(temporary) / f"{sid}.png" for sid, _, _, _ in records]
                seed = 1024 + batch[0]["sample_id"]
                with torch.random.fork_rng():
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    attacker.attack([str(source) for _, source, _, _ in records], [str(p) for p in targets])
                for (sid, source, rp, output), generated in zip(records, targets):
                    if rp.exists():
                        if file_hash(output) != file_hash(generated):
                            raise ValueError("Diff resumed batch differs from frozen output; retain old outputs and investigate determinism")
                        continue
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with generated.open("rb") as src, output.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                    write_once(rp, {"sample_id": sid, "image_kind": kind,
                        "source_image_sha256": file_hash(source),
                        "source_generation_manifest_sha256": pool["sha256"], "config": config,
                        "batch_sample_ids": [r["sample_id"] for r in batch], "seed": seed,
                        "output": artifact(output)})
                profiler.record_processed_images(len(records))
    if pipe is None:
        profiler.model_load_seconds = 0.
        profiler.start_processing()
    profiler.finish_processing()
    profiler.close()
    profiler.save(pool_dir / "runtime", {"stage": "diff_attack", "config": config,
        "generation_pool": pool["generation_pool"], "cache_policy": "validated_whole_batch_resume",
        "warmup_policy": "none_not_benchmark", "paper_runtime_eligible": False})
