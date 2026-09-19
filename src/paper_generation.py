"""Per-sample generation, immutable image/latent records, and pool finalization."""
from pathlib import Path
import argparse
import json

from paper_protocol import (PROTOCOL, GENERATION, GEOMETRY, FREEU, artifact,
    digest, file_hash, load_document, write_once, require_equal,
    validate_generation_pairing, verify_artifact)


def sample_latent(pipe, seed, resolution=512):
    import torch
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe.prepare_latents(1, pipe.unet.config.in_channels, resolution,
                               resolution, pipe.unet.dtype, pipe.device, generator)


def generation_config(args, plan, model, code, env):
    from research_config import EXPERIMENTS
    group = EXPERIMENTS[args.experiment].generation_group
    return {"protocol": PROTOCOL, "generation_protocol": GENERATION,
            "generation_plan_hash": plan["sha256"], "generation_pool": group,
            "freeu_enabled": group == "G1", "freeu": FREEU, "model": model,
            "steps": 50, "guidance": 7.5, "model_dtype": "float32",
            "fft_dtype": "float32", "vae_slicing": True, "geometry": GEOMETRY,
            "batch_size": args.generation_batch_size, "code": code, "environment": env}


def generate(args, splits, plan, model, code, env):
    import numpy as np
    import torch
    from diffusers import DiffusionPipeline, DDIMScheduler
    from freeu import configure_freeu
    from research_config import FreeUConfig
    from runtime_profiling import RuntimeProfiler
    from utils import inject_hsqr, make_hsqr_pattern

    config = generation_config(args, plan, model, code, env)
    pool = Path(args.output_dir) / config["generation_pool"] / args.dataset_id / "HSQR"
    if (pool / "generation_manifest.json").exists():
        raise ValueError("Generation pool is frozen; refuse generation/overwrite")
    write_once(pool / "generation_config.json", config)
    requested = {r["sample_id"] for r in splits["splits"][args.split_name]}
    rows = [r for r in plan["records"] if r["sample_id"] in requested]
    pending = []
    for row in rows:
        record_path = pool / "generation_records" / f"{row['sample_id']}.json"
        if record_path.exists():
            record = load_document(record_path)
            require_equal(record, {"config_hash": digest(config), **row}, "Generation resume")
            for ref in list(record["images"].values()) + list(record["latents"].values()):
                verify_artifact(ref)
        else:
            pending.append(row)
    if not pending:
        print("All requested generation records validated; no generation needed")
        return
    profiler = RuntimeProfiler(args.device)
    profiler.start_model_load()
    kwargs = {"torch_dtype": torch.float32, "local_files_only": True}
    if args.model_revision:
        kwargs["revision"] = args.model_revision
    pipe = DiffusionPipeline.from_pretrained(args.model_id, **kwargs).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    configure_freeu(pipe, FreeUConfig(enabled=config["freeu_enabled"], **FREEU))
    profiler.finish_model_load()
    profiler.attach_unet(pipe.unet)
    patterns = torch.stack([make_hsqr_pattern(7433 + i) for i in range(2048)]).cpu()
    pattern_path = pool / "pattern_list-2048.pt"
    if pattern_path.exists():
        old = torch.load(pattern_path, map_location="cpu", weights_only=True)
        if not torch.equal(old, patterns):
            raise ValueError("Existing reference bank differs")
    else:
        with pattern_path.open("xb") as handle:
            torch.save(patterns, handle)
    profiler.record_requested_images(2 * len(rows))
    profiler.record_cache_hits(2 * (len(rows) - len(pending)))
    profiler.start_processing()
    for start in range(0, len(pending), args.generation_batch_size):
        batch = pending[start:start + args.generation_batch_size]
        with torch.no_grad():
            no = torch.cat([sample_latent(pipe, r["latent_seed"]) for r in batch])
            wm = inject_hsqr(no, patterns[[r["key_index"] for r in batch]], center=True, device=args.device)
            images = pipe([r["prompt"] for r in batch] * 2,
                          latents=torch.cat([no, wm]), guidance_scale=7.5,
                          num_inference_steps=50).images
        for i, row in enumerate(batch):
            sid = row["sample_id"]
            record = {**row, "config_hash": digest(config), "images": {}, "latents": {}}
            for kind, tensor, image in (("no_wm", no[i], images[i]),
                                         ("wm", wm[i], images[len(batch) + i])):
                image_path = pool / ("img_pil" if kind == "no_wm" else "img_pil_wm") / f"{sid}.png"
                latent_path = pool / f"gt_latents_{kind}" / f"{sid}.npy"
                for path in (image_path, latent_path):
                    if path.exists():
                        raise ValueError(f"Unmanifested output exists; cannot prove origin: {path}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                with image_path.open("xb") as handle:
                    image.save(handle, format="PNG")
                with latent_path.open("xb") as handle:
                    np.save(handle, tensor.detach().float().cpu().numpy(), allow_pickle=False)
                record["images"][kind] = artifact(image_path)
                record["latents"][kind] = artifact(latent_path)
                record[f"{kind}_latent_hash"] = file_hash(latent_path)
            write_once(pool / "generation_records" / f"{sid}.json", record)
        profiler.record_processed_images(2 * len(batch))
    profiler.finish_processing()
    profiler.close()
    profiler.save(pool / "runtime", {**config, "stage": "generation",
        "cache_policy": "resume_validated", "warmup_policy": "none_not_benchmark",
        "paper_runtime_eligible": False})


def assemble_pool(root, group, dataset, plan):
    pool = Path(root) / group / dataset / "HSQR"
    config = load_document(pool / "generation_config.json")
    config_hash = digest({k: v for k, v in config.items() if k != "sha256"})
    records, missing = {}, []
    for row in plan["records"]:
        sid = row["sample_id"]
        path = pool / "generation_records" / f"{sid}.json"
        if not path.exists():
            missing.append(sid)
            continue
        record = load_document(path)
        require_equal(record, {"config_hash": config_hash, **row}, "Generation record")
        for kind in ("no_wm", "wm"):
            if record[f"{kind}_latent_hash"] != record["latents"][kind]["sha256"]:
                raise ValueError("Latent content hash does not match artifact hash")
        records[str(sid)] = record
    if missing:
        raise ValueError(f"{group} needs {len(missing)} more samples. Missing sample IDs: {missing}")
    return {**{k: v for k, v in config.items() if k != "sha256"}, "records": records,
            "patterns": artifact(pool / "pattern_list-2048.pt")}


def freeze_pair(root, dataset, plan_path):
    plan = load_document(plan_path)
    g0, g1 = [assemble_pool(root, g, dataset, plan) for g in ("G0", "G1")]
    validate_generation_pairing(g0, g1, plan)
    refs = {}
    for group, manifest in (("G0", g0), ("G1", g1)):
        path = Path(root) / group / dataset / "HSQR" / "generation_manifest.json"
        write_once(path, manifest)
        refs[group] = artifact(path)
    return write_once(Path(root) / "manifests" / "pairing_validation.json",
                      {"protocol": PROTOCOL, "status": "PASS", "generation_plan": artifact(plan_path),
                       "pools": refs, "n_samples": len(plan["records"])})


def audit_legacy(evidence_path=None):
    """No inference from filenames. Promotion needs contemporaneous hashed evidence.

    Evidence is a sealed document containing pools in the same schema as the
    frozen manifest, the original plan, and a historical generation-ledger file
    for each pool. The ledger binds each output hash to its GT latent hash and
    the exact invocation (range, batch, seeds, model/settings). Reconstructed
    hashes alone cannot establish a historical image-to-latent association.
    """
    if not evidence_path:
        return {"status": "legacy_not_formal", "reason": "Missing contemporaneous generation ledger and GT latent evidence; keep as pilot"}
    try:
        evidence = load_document(evidence_path)
        plan = load_document(verify_artifact(evidence["plan"]))
        pools = [load_document(verify_artifact(evidence[g])) for g in ("G0", "G1")]
        for group, pool in zip(("G0", "G1"), pools):
            ledger = load_document(verify_artifact(evidence["historical_ledgers"][group]))
            if ledger["records"] != pool["records"]:
                raise ValueError("Historical output-to-latent ledger differs")
            require_equal(pool, {"generation_protocol": "legacy_batch_v1_frozen",
                "sample_start": ledger["sample_start"], "batch_size": ledger["batch_size"],
                "seed_scheme": ledger["seed_scheme"], "partial_regeneration": False}, "Legacy ledger")
        require_equal(pools[1], {k: pools[0][k] for k in
                      ("sample_start", "batch_size", "seed_scheme")}, "Legacy pairing")
        validate_generation_pairing(*pools, plan)
        return {"status": "legacy_batch_v1_frozen", "evidence": artifact(evidence_path),
                "pools": {g: evidence[g] for g in ("G0", "G1")}, "plan": evidence["plan"]}
    except (ValueError, KeyError, OSError) as exc:
        return {"status": "legacy_not_formal", "reason": str(exc)}


def promote_legacy(evidence_path, target_root, dataset):
    """Explicit promotion only; retains original images and protects source pools.

    This requires an already group-isolated formal plan and historical ledgers.
    It cannot manufacture missing GT latents or establish the origin of old PNGs.
    """
    result = audit_legacy(evidence_path)
    if result["status"] != "legacy_batch_v1_frozen":
        raise ValueError(result["reason"])
    plan = load_document(verify_artifact(result["plan"]))
    if plan.get("generation_protocol") != "legacy_batch_v1_frozen":
        raise ValueError("Legacy promotion requires a legacy plan, not a fabricated per_sample_v2 plan")
    splits = load_document(verify_artifact(plan["split_manifest"]))
    from paper_protocol import validate_split_source
    validate_split_source(splits)
    expected = {r["sample_id"]: r for name in ("fit", "calibration", "test") for r in splits["splits"][name]}
    if {r["sample_id"] for r in plan["records"]} != set(expected):
        raise ValueError("Old pool does not cover the full group-isolated formal split")
    for row in plan["records"]:
        require_equal(row, expected[row["sample_id"]], "Legacy sample-to-dataset binding")
    refs = {}
    for group in ("G0", "G1"):
        source = verify_artifact(result["pools"][group])
        manifest = load_document(source)
        # Old generation/attack entry points explicitly refuse frozen markers.
        write_once(source.parent / "_formal_frozen.json", {"evidence": artifact(evidence_path),
                   "generation_manifest": artifact(source), "status": "legacy_batch_v1_frozen"})
        target = Path(target_root) / group / dataset / "HSQR" / "generation_manifest.json"
        write_once(target, manifest)
        refs[group] = artifact(target)
    return write_once(Path(target_root) / "manifests" / "pairing_validation.json",
        {"protocol": PROTOCOL, "status": "PASS", "legacy_status": "legacy_batch_v1_frozen",
         "legacy_evidence": artifact(evidence_path), "generation_plan": result["plan"], "pools": refs,
         "n_samples": len(plan["records"])})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default="paper_v1")
    p.add_argument("--dataset_id", default="coco")
    p.add_argument("--generation_plan")
    p.add_argument("--audit_legacy", action="store_true")
    p.add_argument("--evidence")
    p.add_argument("--promote_legacy", action="store_true", help="Explicitly freeze a fully evidenced old pool into output_dir; does not copy/modify its images")
    args = p.parse_args()
    if args.promote_legacy:
        if not args.evidence:
            p.error("--promote_legacy requires --evidence")
        promote_legacy(args.evidence, args.output_dir, args.dataset_id)
    elif args.audit_legacy:
        print(json.dumps(audit_legacy(args.evidence), indent=2))
    elif args.generation_plan:
        freeze_pair(args.output_dir, args.dataset_id, args.generation_plan)
        print("GENERATION PAIRING: PASS (not the full formal experiment gate)")
    else:
        p.error("--generation_plan or --audit_legacy is required")


if __name__ == "__main__":
    main()
