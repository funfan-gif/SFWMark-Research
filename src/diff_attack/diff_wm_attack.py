"""Original SFWMark diffusion-regeneration attack with range-safe CLI support."""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from .attdiffusion import DiffWMAttacker, ReSDPipeline
except ImportError:  # Direct ``python diff_wm_attack.py`` compatibility.
    from attdiffusion import DiffWMAttacker, ReSDPipeline

try:
    from runtime_profiling import RuntimeProfiler
except ImportError:  # Direct execution adds only src/diff_attack to sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from runtime_profiling import RuntimeProfiler


BASELINE_MODEL_ID = "stabilityai/stable-diffusion-2-1"
BASELINE_MODEL_REVISION = "fp16"
BASELINE_BATCH_SIZE = 8
BASELINE_NOISE_STEP = 60
BASELINE_DETECT_TRIALS = 1000


def _revision(value):
    return None if value is None or str(value).lower() in {"", "none", "null"} else value


def _range(args):
    start = int(args.sample_start)
    stop = (
        BASELINE_DETECT_TRIALS
        if args.sample_count is None
        else start + int(args.sample_count)
    )
    if start < 0 or start >= BASELINE_DETECT_TRIALS:
        raise ValueError(
            f"sample_start must be in [0,{BASELINE_DETECT_TRIALS}), got {start}"
        )
    if args.sample_count is not None and int(args.sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    return range(start, min(stop, BASELINE_DETECT_TRIALS))


def _paths(pool_dir, indices, source_name, output_name):
    inputs = [pool_dir / source_name / f"{index}.png" for index in indices]
    outputs = [pool_dir / output_name / f"{index}.png" for index in indices]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"Missing {len(missing)} diffusion-attack source image(s); first: {preview}"
        )
    return inputs, outputs


def run_diffusion_attack(args):
    """Run the unchanged baseline attack in one explicit G0/G1 pool."""

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        output_root = project_root / output_root
    pool_dir = output_root / args.dataset_id / args.wm_type
    if any((pool_dir / name).exists() for name in ("generation_manifest.json", "generation_config.json", "_formal_frozen.json")):
        raise ValueError("Frozen/formal generation pool: use research_hsqr.py --formal --stage diff_attack")
    no_output_dir = pool_dir / "img_pil-diffatt_fp16"
    wm_output_dir = pool_dir / "img_pil_wm-diffatt_fp16"
    no_output_dir.mkdir(parents=True, exist_ok=True)
    wm_output_dir.mkdir(parents=True, exist_ok=True)

    indices = list(_range(args))
    no_inputs, no_outputs = _paths(
        pool_dir, indices, "img_pil", "img_pil-diffatt_fp16"
    )
    wm_inputs, wm_outputs = _paths(
        pool_dir, indices, "img_pil_wm", "img_pil_wm-diffatt_fp16"
    )
    overwrite = bool(getattr(args, "overwrite", False))
    from paper_protocol import file_hash, load_document, require_equal, write_once, artifact
    attack_config = {"model_id": args.diff_attack_model_id,
                     "model_revision": _revision(args.diff_attack_model_revision),
                     "noise_step": BASELINE_NOISE_STEP, "batch_size": BASELINE_BATCH_SIZE,
                     "dtype": "float16", "protocol": "legacy_diff_provenance_v2"}
    for source, output in zip(no_inputs + wm_inputs, no_outputs + wm_outputs):
        if output.exists() and not overwrite:
            record_path = output.with_suffix(".provenance.json")
            record = load_document(record_path)
            require_equal(record, {"source_image_sha256": file_hash(source),
                "config": attack_config, "output_sha256": file_hash(output)},
                "Existing Diff result; use --overwrite to regenerate legacy outputs")
    pending_count = sum(overwrite or not path.is_file() for path in no_outputs)
    pending_count += sum(overwrite or not path.is_file() for path in wm_outputs)
    skipped_count = 2 * len(indices) - pending_count

    profiler = RuntimeProfiler(getattr(args, "device", "cuda"))
    profiler.record_requested_images(2 * len(indices))
    profiler.record_cache_hits(skipped_count)
    revision = _revision(args.diff_attack_model_revision)

    if pending_count:
        load_kwargs = {
            "torch_dtype": torch.float16,
            "local_files_only": bool(args.diff_attack_local_files_only),
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        profiler.start_model_load()
        att_pipe = ReSDPipeline.from_pretrained(
            args.diff_attack_model_id, **load_kwargs
        )
        att_pipe.set_progress_bar_config(disable=True)
        att_pipe.to(args.device)
        profiler.finish_model_load()
        profiler.attach_unet(att_pipe.unet)
        attacker = DiffWMAttacker(
            att_pipe,
            batch_size=BASELINE_BATCH_SIZE,
            noise_step=BASELINE_NOISE_STEP,
            captions={},
        )

        profiler.start_processing()
        for batch_start in tqdm(
            range(0, len(indices), BASELINE_BATCH_SIZE), desc="diff attack"
        ):
            batch_stop = batch_start + BASELINE_BATCH_SIZE
            # Preserve the original protocol: reset the attacker's generator
            # for each 8-image no-wm call and each corresponding wm call.
            attacker.attack(
                [str(path) for path in no_inputs[batch_start:batch_stop]],
                [str(path) for path in no_outputs[batch_start:batch_stop]],
                multi=overwrite,
            )
            attacker.attack(
                [str(path) for path in wm_inputs[batch_start:batch_stop]],
                [str(path) for path in wm_outputs[batch_start:batch_stop]],
                multi=overwrite,
            )
        profiler.record_processed_images(pending_count)
        profiler.finish_processing()
        profiler.close()
        for source, output in zip(no_inputs + wm_inputs, no_outputs + wm_outputs):
            record_path = output.with_suffix(".provenance.json")
            record = {"source_image_sha256": file_hash(source), "config": attack_config,
                      "source_generation_manifest_sha256": "legacy_unverified",
                      "output_sha256": file_hash(output)}
            if overwrite and record_path.exists():
                # Explicit legacy overwrite retains old provenance as a hash-named audit record.
                old = load_document(record_path)
                write_once(record_path.with_name(record_path.stem + '-' + old['sha256'] + '.json'), old)
                record_path.unlink()
            write_once(record_path, record)
    else:
        profiler.model_load_seconds = 0.0
        profiler.start_processing()
        profiler.finish_processing()
        print("All requested diffusion-regeneration outputs already exist; skipped safely.")

    generation_pool = getattr(args, "generation_pool", None)
    manifest = {
        "generation_pool": generation_pool,
        "dataset_id": args.dataset_id,
        "wm_type": args.wm_type,
        "sample_start": indices[0],
        "sample_stop": indices[-1] + 1,
        "sample_count": len(indices),
        "overwrite": overwrite,
        "skipped_existing_images": skipped_count,
        "generation_model_id": getattr(args, "model_id", None),
        "generation_model_revision": getattr(args, "model_revision", None),
        "diff_attack_model_id": args.diff_attack_model_id,
        "diff_attack_model_revision": revision,
        "diff_attack_local_files_only": bool(args.diff_attack_local_files_only),
        "torch_dtype": "float16",
        "resolution": 512,
        "batch_size": BASELINE_BATCH_SIZE,
        "noise_step": BASELINE_NOISE_STEP,
    }
    manifest_path = pool_dir / (
        f"diff_attack_manifest-{indices[0]}-{indices[-1] + 1}.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    runtime_path = profiler.save(
        pool_dir / "research_results",
        {
            **manifest,
            "experiment": getattr(args, "experiment", None),
            "stage": "diff_attack",
            "inversion": None,
            "freeu_generation": getattr(args, "freeu_generation", None),
            "freeu_inversion": None,
            "git_commit": getattr(args, "git_commit", "unknown"),
        },
    )
    return {
        "pool_dir": pool_dir,
        "manifest_path": manifest_path,
        "runtime_path": runtime_path,
        "processed_images": pending_count,
        "skipped_images": skipped_count,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Original SFWMark diffusion-regeneration attack"
    )
    parser.add_argument("--wm_type", required=True)
    parser.add_argument("--dataset_id", choices=("coco", "Gustavo", "DB1k"), required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diff_attack_model_id", default=BASELINE_MODEL_ID)
    parser.add_argument("--diff_attack_model_revision", default=BASELINE_MODEL_REVISION)
    parser.add_argument("--diff_attack_local_files_only", action="store_true")
    return parser


if __name__ == "__main__":
    run_diffusion_attack(build_parser().parse_args())
