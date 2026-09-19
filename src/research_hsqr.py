"""Q0-Q5 HSQR research runner with reusable inversion-feature caches.

Generation pools are shared by construction: Q0-Q2 use G0 (FreeU off) and
Q3-Q5 use G1 (FreeU on).  D0/D1 are diagnostic-only mismatched generation /
inversion configurations.  A whitening model is fitted once and must then be
named explicitly for evaluation, unless the user deliberately requests the
separate attack-specific oracle diagnostic.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import diffusers
import numpy as np
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from PIL import Image
from tqdm import tqdm

from feature_cache import FeatureCache, FeatureCacheKey
from hsqr_metrics import HSQRDistanceModel, extract_hsqr_feature, hsqr_reference_feature
from inversion import InversionConfig, invert_image
from research_attacks import (
    ORIGINAL_ATTACKS,
    apply_attack_pair,
    attack_display_name,
    attack_parameters,
)
from research_config import (
    EXPERIMENTS,
    FREEU_DEFAULTS,
    ROTATION_BILINEAR_ANGLES,
    FreeUConfig,
)
from research_summary import summarize_results, summarize_runtime
from runtime_profiling import RuntimeProfiler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WM_CAPACITY = 2048
DISTANCES = ("complex_l1", "l2", "diag", "mahalanobis")
DEFAULT_INVERSION = InversionConfig()


@dataclass(frozen=True)
class AttackSpec:
    name: str
    angle: float = 0.0

    @property
    def parameters(self):
        return attack_parameters(self.name, self.angle)

    @property
    def display_name(self):
        return attack_display_name(self.name, self.angle)

    @property
    def slug(self):
        if self.name not in {"rotation_bilinear", "orthogonal"}:
            return self.name
        angle = f"{self.angle:g}".replace("-", "m").replace(".", "p")
        return f"{self.name}-{angle}"

    @property
    def is_original12(self):
        return self.name in ORIGINAL_ATTACKS


def _attack_specs(args) -> List[AttackSpec]:
    if args.attack == "paper_all":
        from copy import copy
        rotation_args = copy(args)
        rotation_args.attack = "all_rotations"
        return [AttackSpec(name) for name in ORIGINAL_ATTACKS] + _attack_specs(rotation_args)
    if args.attack == "original12":
        return [AttackSpec(name) for name in ORIGINAL_ATTACKS]
    if args.attack in ORIGINAL_ATTACKS:
        return [AttackSpec(args.attack)]
    if args.attack == "rot75_nn":
        return [AttackSpec("rot75_nn", 75.0)]
    if args.attack == "rotation_bilinear":
        angles = args.angles if args.angles is not None else ROTATION_BILINEAR_ANGLES
        return [AttackSpec("rotation_bilinear", float(angle)) for angle in angles]
    if args.attack == "orthogonal":
        angles = args.angles if args.angles is not None else (90, 180, 270)
        return [AttackSpec("orthogonal", float(angle)) for angle in angles]
    if args.attack == "all_rotations":
        result = [AttackSpec("rot75_nn", 75.0)]
        result.extend(
            AttackSpec("rotation_bilinear", float(angle))
            for angle in ROTATION_BILINEAR_ANGLES
        )
        result.extend(AttackSpec("orthogonal", float(angle)) for angle in (90, 180, 270))
        return result
    raise ValueError(f"Unknown attack selection: {args.attack}")


def _output_root(args) -> Path:
    root = Path(args.output_dir)
    return root if root.is_absolute() else PROJECT_ROOT / root


def _experiment_dir(args) -> Path:
    return _output_root(args) / args.experiment / args.dataset_id / "HSQR"


def _generation_dir(args) -> Path:
    group = EXPERIMENTS[args.experiment].generation_group
    return _output_root(args) / group / args.dataset_id / "HSQR"


def _resolve_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_model_path(value) -> Path:
    path = _resolve_path(value)
    return path if path.suffix.lower() == ".npz" else Path(str(path) + ".npz")


def _effective_freeu(args, enabled: bool) -> FreeUConfig:
    return FreeUConfig(
        enabled=enabled,
        s1=args.freeu_s1,
        s2=args.freeu_s2,
        b1=args.freeu_b1,
        b2=args.freeu_b2,
    )


def _torch_dtype(args):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.torch_dtype]


def _run_generation(args):
    from generate import main as generate_main

    experiment = EXPERIMENTS[args.experiment]
    generation_freeu = _effective_freeu(args, experiment.generation_freeu.enabled)
    # generate.py appends dataset/watermark; group is the shared G0/G1 root.
    output_root = _output_root(args) / experiment.generation_group
    generate_main(SimpleNamespace(
        output_dir=str(output_root),
        dataset_id=args.dataset_id,
        wm_type="HSQR",
        generation_freeu=generation_freeu.enabled,
        freeu_s1=generation_freeu.s1,
        freeu_s2=generation_freeu.s2,
        freeu_b1=generation_freeu.b1,
        freeu_b2=generation_freeu.b2,
        sample_start=args.sample_start,
        sample_count=args.sample_count,
        model_id=args.model_id,
        model_revision=args.model_revision,
        local_files_only=args.local_files_only,
        torch_dtype=args.torch_dtype,
        device=args.device,
        save_gt_latents=args.save_gt_latents,
        experiment=args.experiment,
        generation_pool=experiment.generation_group,
        git_commit=args.git_commit,
    ))


def _run_diff_attack(args):
    from diff_attack.diff_wm_attack import run_diffusion_attack

    experiment = EXPERIMENTS[args.experiment]
    generation_root = _output_root(args) / experiment.generation_group
    return run_diffusion_attack(SimpleNamespace(
        output_dir=str(generation_root),
        dataset_id=args.dataset_id,
        wm_type="HSQR",
        sample_start=args.sample_start,
        sample_count=args.sample_count,
        overwrite=args.overwrite,
        device=args.device,
        diff_attack_model_id=args.diff_attack_model_id,
        diff_attack_model_revision=args.diff_attack_model_revision,
        diff_attack_local_files_only=args.diff_attack_local_files_only,
        experiment=args.experiment,
        generation_pool=experiment.generation_group,
        freeu_generation=experiment.generation_freeu.enabled,
        model_id=args.model_id,
        model_revision=args.model_revision,
        git_commit=args.git_commit,
    ))


def _load_pipe(args):
    load_kwargs = {
        "torch_dtype": _torch_dtype(args),
        "local_files_only": args.local_files_only,
    }
    if args.model_revision is not None:
        load_kwargs["revision"] = args.model_revision
    pipe = DiffusionPipeline.from_pretrained(args.model_id, **load_kwargs)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _inversion_config(args) -> InversionConfig:
    experiment = EXPERIMENTS[args.experiment]
    inversion_freeu = _effective_freeu(args, experiment.inversion_freeu.enabled)
    return InversionConfig(
        method=experiment.inversion_method,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.invert_guidance,
        prompt=args.invert_prompt,
        exact_max_iterations=args.exact_max_iterations,
        exact_tolerance=args.exact_tolerance,
        exact_step_size=args.exact_step_size,
        exact_update=args.exact_update,
        gnri_max_iterations=args.gnri_max_iterations,
        gnri_tolerance=args.gnri_tolerance,
        gnri_lambda=args.gnri_lambda,
        gnri_eta=args.gnri_eta,
        gnri_step_scale=args.gnri_step_scale,
        gnri_max_update_norm=args.gnri_max_update_norm,
        diagnostics=args.diagnostics,
        inversion_freeu=inversion_freeu,
    )


def _inversion_kwargs(config: InversionConfig):
    return {
        field.name: getattr(config, field.name)
        for field in fields(InversionConfig)
        if field.name != "method"
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _image_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(f"{image.width}x{image.height}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _cache_key(args, config, attack, sample_id, image_kind, source_sha256):
    experiment = EXPERIMENTS[args.experiment]
    generation_freeu = _effective_freeu(args, experiment.generation_freeu.enabled)
    return FeatureCacheKey(
        inversion_method=config.method,
        inversion_hyperparameters=config.cache_dict(),
        generation_freeu=asdict(generation_freeu),
        inversion_freeu=asdict(config.inversion_freeu),
        generation_group=experiment.generation_group,
        model_id=args.model_id,
        model_revision=args.model_revision,
        diffusers_version=str(diffusers.__version__),
        torch_version=str(torch.__version__),
        torch_dtype=args.torch_dtype,
        num_inference_steps=config.num_inference_steps,
        invert_guidance=config.guidance_scale,
        invert_prompt=config.prompt,
        git_commit=args.git_commit,
        source_image_sha256=source_sha256,
        attack=attack.name,
        attack_parameters=attack.parameters,
        dataset_id=args.dataset_id,
        sample_id=str(sample_id),
        wm_type="HSQR",
        image_kind=image_kind,
    )


def _load_attacked_pair(args, attack: AttackSpec, index: int):
    generation_dir = _generation_dir(args)
    if attack.name == "diff":
        no_dir = generation_dir / "img_pil-diffatt_fp16"
        wm_dir = generation_dir / "img_pil_wm-diffatt_fp16"
    else:
        no_dir = generation_dir / "img_pil"
        wm_dir = generation_dir / "img_pil_wm"
    no_path = no_dir / f"{index}.png"
    wm_path = wm_dir / f"{index}.png"
    if not no_path.is_file() or not wm_path.is_file():
        if attack.name == "diff":
            group = EXPERIMENTS[args.experiment].generation_group
            raise FileNotFoundError(
                f"Diff regeneration results missing for {group}. "
                f"Run --stage diff_attack first. Missing: {no_path} / {wm_path}"
            )
        raise FileNotFoundError(
            f"Missing attack input pair for sample {index}: {no_path} / {wm_path}"
        )
    with Image.open(no_path) as image:
        no_wm = image.convert("RGB")
    with Image.open(wm_path) as image:
        wm = image.convert("RGB")
    return apply_attack_pair(
        no_wm, wm, attack.name, angle=attack.angle,
        seed=42 + index, device=args.device,
    )


def _features_for_pair(pipe, cache, args, config, attack, index, profiler=None):
    # Load/attack first so the cache key fingerprints the exact pixels consumed
    # by inversion, including regenerated Diff inputs.
    no_wm, wm = _load_attacked_pair(args, attack, index)
    images = {"no_wm": no_wm, "wm": wm}
    keys = {
        kind: _cache_key(
            args, config, attack, index, kind, _image_sha256(images[kind])
        )
        for kind in images
    }
    result = {}
    missing_kinds = []
    if profiler is not None:
        profiler.record_requested_images(len(images))
    for kind, key in keys.items():
        if cache.contains(key):
            result[kind] = cache.load(key).numpy()
            if profiler is not None:
                profiler.record_cache_hits(1)
        else:
            missing_kinds.append(kind)

    if missing_kinds:
        calls_before = profiler.unet_forward_calls if profiler is not None else 0
        inversion_started = profiler.start_inversion() if profiler is not None else None
        recovered = invert_image(
            pipe,
            [images[kind] for kind in missing_kinds],
            method=config.method,
            **_inversion_kwargs(config),
        )
        if profiler is not None:
            profiler.finish_inversion(inversion_started)
            profiler.record_inversion(
                config.method,
                len(missing_kinds),
                config.num_inference_steps,
                calls_before,
            )
        features = extract_hsqr_feature(recovered, center=True).detach().cpu().numpy()
        for row, kind in enumerate(missing_kinds):
            cache.store(keys[kind], features[row])
            result[kind] = features[row]
    return result["no_wm"], result["wm"]


def _require_diff_results(args, attacks, indices):
    if not any(attack.name == "diff" for attack in attacks):
        return
    generation_dir = _generation_dir(args)
    missing = []
    for index in sorted(set(indices)):
        for directory in ("img_pil-diffatt_fp16", "img_pil_wm-diffatt_fp16"):
            path = generation_dir / directory / f"{index}.png"
            if not path.is_file():
                missing.append(path)
    if missing:
        group = EXPERIMENTS[args.experiment].generation_group
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"Diff regeneration results missing for {group}. "
            f"Run --stage diff_attack first. Missing {len(missing)} file(s); first: {preview}"
        )


def _load_references_and_labels(generation_dir: Path):
    patterns = torch.load(
        generation_dir / f"pattern_list-{WM_CAPACITY}.pt", map_location="cpu"
    )
    references = hsqr_reference_feature(patterns).numpy()
    label_files = sorted(generation_dir.glob("identify_gt_indices_*.npy"))
    if len(label_files) != 1:
        raise RuntimeError(f"Expected one identify_gt_indices file in {generation_dir}")
    labels = np.load(label_files[0])
    return references, labels


def _configuration_payload(args, config, attacks: Sequence[AttackSpec]):
    experiment = EXPERIMENTS[args.experiment]
    return {
        "experiment": args.experiment,
        "experiment_display_name": experiment.display_name,
        "generation_group": experiment.generation_group,
        "generation_freeu": asdict(
            _effective_freeu(args, experiment.generation_freeu.enabled)
        ),
        "inversion": config.cache_dict(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "local_files_only": args.local_files_only,
        "torch_dtype": args.torch_dtype,
        "torch_version": str(torch.__version__),
        "diffusers_version": str(diffusers.__version__),
        "git_commit": args.git_commit,
        "dataset_id": args.dataset_id,
        "fit_start": args.fit_start,
        "fit_count": args.fit_count,
        "attacks": [
            {"name": attack.name, "angle": attack.angle, "parameters": attack.parameters}
            for attack in attacks
        ],
    }


def _configuration_digest(args, config, attacks: Sequence[AttackSpec]) -> str:
    payload = json.dumps(
        _configuration_payload(args, config, attacks),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _default_model_path(args, config, attacks: Sequence[AttackSpec], stem: str) -> Path:
    digest = _configuration_digest(args, config, attacks)
    return _experiment_dir(args) / "research_models" / f"{stem}-{digest}.npz"


def _write_model_metadata(path: Path, payload: dict):
    metadata_path = path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _fit(pipe, cache, args, config, attacks, references, labels, model_path,
         profiler=None):
    stop = min(args.fit_start + args.fit_count, len(labels))
    if args.fit_start < 0 or stop <= args.fit_start:
        raise ValueError("The fitting split is empty")
    query_features = []
    correct_references = []
    for attack in attacks:
        for index in tqdm(range(args.fit_start, stop), desc=f"fit {attack.slug}"):
            _, wm_feature = _features_for_pair(
                pipe, cache, args, config, attack, index, profiler=profiler
            )
            query_features.append(wm_feature)
            correct_references.append(references[int(labels[index])])
    model = HSQRDistanceModel().fit(query_features, correct_references)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    _write_model_metadata(model_path, {
        **_configuration_payload(args, config, attacks),
        "whitening_model": str(model_path.resolve()),
        "fitting_residual_count": len(query_features),
        "oracle_attack_specific": bool(args.oracle_attack_whitening),
    })
    print(f"Saved whitening model: {model_path}")
    return model_path


def _result_tag(args, config, attack, model_path):
    payload = {
        "configuration": _configuration_payload(args, config, [attack]),
        "whitening_model": str(model_path.resolve()),
        "eval_start": args.eval_start,
        "eval_count": args.eval_count,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _evaluate(pipe, cache, args, config, attack, references, labels, model_path,
              profiler=None):
    if not model_path.is_file():
        raise FileNotFoundError(f"Whitening model not found: {model_path}")
    model = HSQRDistanceModel.load(model_path)
    stop = min(args.eval_start + args.eval_count, len(labels))
    if args.eval_start < 0 or stop <= args.eval_start:
        raise ValueError("The evaluation split is empty")
    if references.shape[0] != WM_CAPACITY:
        raise ValueError(
            f"Identification requires {WM_CAPACITY} candidate keys, got {references.shape[0]}"
        )
    prepared_candidates = model.prepare_references(references)
    output = {
        f"{kind}_{metric}": []
        for metric in DISTANCES
        for kind in ("no_wm", "wm", "predicted", "id_correct")
    }

    for index in tqdm(range(args.eval_start, stop), desc=f"eval {attack.slug}"):
        no_feature, wm_feature = _features_for_pair(
            pipe, cache, args, config, attack, index, profiler=profiler
        )
        key_index = int(labels[index])
        claimed_reference = references[key_index:key_index + 1]
        no_scores = model.distances(no_feature, claimed_reference, DISTANCES)
        wm_scores = model.distances(wm_feature, claimed_reference, DISTANCES)
        # Identification sees all 2048 candidates and receives no GT hint.
        candidate_scores = model.distances_to_prepared(
            wm_feature, prepared_candidates, DISTANCES
        )
        for metric in DISTANCES:
            predicted = int(np.argmin(candidate_scores[metric][0]))
            output[f"no_wm_{metric}"].append(float(no_scores[metric][0, 0]))
            output[f"wm_{metric}"].append(float(wm_scores[metric][0, 0]))
            output[f"predicted_{metric}"].append(predicted)
            output[f"id_correct_{metric}"].append(predicted == key_index)

    metadata = {
        **_configuration_payload(args, config, [attack]),
        "attack_name": attack.name,
        "attack_display_name": attack.display_name,
        "attack_slug": attack.slug,
        "attack_parameters": attack.parameters,
        "is_original12": attack.is_original12,
        "whitening_model": str(model_path.resolve()),
        "oracle_attack_specific": bool(args.oracle_attack_whitening),
        "eval_start": args.eval_start,
        "eval_stop": stop,
    }
    result_dir = _experiment_dir(args) / "research_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / (
        f"distances-{attack.slug}-{_result_tag(args, config, attack, model_path)}.npz"
    )
    np.savez_compressed(
        result_path,
        **{key: np.asarray(value) for key, value in output.items()},
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Saved raw distances: {result_path}")
    return result_path


def _oracle_model_path(args, config, attack):
    return _default_model_path(args, config, [attack], f"oracle-{attack.slug}")


def _runtime_metadata(args, config, attacks, stage, model_reused=False):
    experiment = EXPERIMENTS[args.experiment]
    generation_freeu = _effective_freeu(args, experiment.generation_freeu.enabled)
    return {
        "experiment": args.experiment,
        "generation_pool": experiment.generation_group,
        "stage": stage,
        "inversion": config.method,
        "freeu_generation": generation_freeu.enabled,
        "freeu_inversion": config.inversion_freeu.enabled,
        "freeu": asdict(config.inversion_freeu),
        "attacks": [attack.slug for attack in attacks],
        "configured_inversion_steps": config.num_inference_steps,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "torch_dtype": args.torch_dtype,
        "model_reused_from_same_process": model_reused,
        "git_commit": args.git_commit,
    }


def _start_stage_profiler(pipe, args, model_load_seconds):
    profiler = RuntimeProfiler(args.device)
    profiler.model_load_seconds = model_load_seconds
    profiler.attach_unet(pipe.unet)
    profiler.start_processing()
    return profiler


def _finish_stage_profiler(profiler, args, config, attacks, stage,
                           model_reused=False):
    profiler.finish_processing()
    profiler.close()
    return profiler.save(
        _experiment_dir(args) / "research_results",
        _runtime_metadata(args, config, attacks, stage, model_reused=model_reused),
    )


def main(args):
    if getattr(args, "formal", False):
        from paper_runner import run
        return run(args)
    args.git_commit = _git_commit()
    if args.stage in ("generate", "all"):
        _run_generation(args)
        if args.stage == "generate":
            return

    if args.stage == "diff_attack":
        _run_diff_attack(args)
        return

    if args.stage in ("summarize", "summarize_runtime"):
        selected_model = (
            str(_resolve_model_path(args.whitening_model).resolve())
            if args.whitening_model else None
        )
        experiment_results = _experiment_dir(args) / "research_results"
        generation_results = _generation_dir(args) / "research_results"
        if args.stage == "summarize":
            summarize_results(
                experiment_results,
                output_prefix=args.summary_prefix,
                whitening_model=selected_model,
                allow_mixed_models=args.oracle_attack_whitening,
            )
        summarize_runtime(
            (experiment_results, generation_results), experiment_results
        )
        return

    if args.oracle_attack_whitening and args.whitening_model:
        raise ValueError(
            "--oracle_attack_whitening and --whitening_model are mutually exclusive"
        )

    if (args.stage == "evaluate" and not args.whitening_model
            and not args.oracle_attack_whitening):
        raise ValueError(
            "Evaluation requires --whitening_model PATH. The runner will not "
            "infer a detector from the attack name."
        )

    attacks = _attack_specs(args)
    generation_dir = _generation_dir(args)
    references, labels = _load_references_and_labels(generation_dir)
    required_diff_indices = []
    if args.stage in ("fit", "all"):
        required_diff_indices.extend(
            range(args.fit_start, min(args.fit_start + args.fit_count, len(labels)))
        )
    if args.stage in ("evaluate", "all"):
        required_diff_indices.extend(
            range(args.eval_start, min(args.eval_start + args.eval_count, len(labels)))
        )
    _require_diff_results(args, attacks, required_diff_indices)
    config = _inversion_config(args)
    load_profiler = RuntimeProfiler(args.device)
    load_profiler.start_model_load()
    pipe = _load_pipe(args)
    model_load_seconds = load_profiler.finish_model_load()
    cache = FeatureCache(_experiment_dir(args) / "feature_cache")
    fitted_paths: Dict[str, Path] = {}

    if args.stage in ("fit", "all"):
        profiler = _start_stage_profiler(pipe, args, model_load_seconds)
        try:
            if args.oracle_attack_whitening:
                for attack in attacks:
                    path = _oracle_model_path(args, config, attack)
                    fitted_paths[attack.slug] = _fit(
                        pipe, cache, args, config, [attack], references, labels, path,
                        profiler=profiler,
                    )
            else:
                model_path = (
                    _resolve_model_path(args.whitening_model)
                    if args.whitening_model
                    else _default_model_path(args, config, attacks, args.whitening_name)
                )
                shared_path = _fit(
                    pipe, cache, args, config, attacks, references, labels, model_path,
                    profiler=profiler,
                )
                fitted_paths = {attack.slug: shared_path for attack in attacks}
        except Exception:
            profiler.close()
            raise
        _finish_stage_profiler(
            profiler, args, config, attacks, "fit", model_reused=False
        )

    if args.stage in ("evaluate", "all"):
        if args.whitening_model:
            shared_path = _resolve_model_path(args.whitening_model)
            evaluation_paths = {attack.slug: shared_path for attack in attacks}
        elif fitted_paths:
            # stage=all may immediately reuse the single model just fitted.
            evaluation_paths = fitted_paths
        elif args.oracle_attack_whitening:
            evaluation_paths = {
                attack.slug: _oracle_model_path(args, config, attack)
                for attack in attacks
            }
        else:
            raise ValueError(
                "Evaluation requires --whitening_model PATH. Fit W_clean/W_robust "
                "once, then pass that same file to every attack. Use "
                "--oracle_attack_whitening only for the explicitly labelled oracle diagnostic."
            )
        reused = args.stage == "all"
        profiler = _start_stage_profiler(
            pipe, args, 0.0 if reused else model_load_seconds
        )
        try:
            for attack in attacks:
                _evaluate(
                    pipe, cache, args, config, attack, references, labels,
                    evaluation_paths[attack.slug], profiler=profiler,
                )
        except Exception:
            profiler.close()
            raise
        _finish_stage_profiler(
            profiler, args, config, attacks, "evaluate", model_reused=reused
        )


def build_parser():
    parser = argparse.ArgumentParser(description="SFWMark HSQR research runner")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument(
        "--stage", choices=(
            "generate", "diff_attack", "fit", "evaluate", "all",
            "summarize", "summarize_runtime", "calibrate",
        ),
        required=True,
    )
    parser.add_argument("--dataset_id", choices=("coco", "Gustavo", "DB1k"), required=True)
    parser.add_argument("--output_dir", default="research_outputs")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_count", type=int)
    parser.add_argument("--save_gt_latents", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly overwrite existing diffusion-regeneration outputs",
    )

    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--model_revision")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--diff_attack_model_id", default="stabilityai/stable-diffusion-2-1"
    )
    parser.add_argument("--diff_attack_model_revision", default="fp16")
    parser.add_argument("--diff_attack_local_files_only", action="store_true")
    parser.add_argument(
        "--torch_dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeu_s1", type=float, default=FREEU_DEFAULTS.s1)
    parser.add_argument("--freeu_s2", type=float, default=FREEU_DEFAULTS.s2)
    parser.add_argument("--freeu_b1", type=float, default=FREEU_DEFAULTS.b1)
    parser.add_argument("--freeu_b2", type=float, default=FREEU_DEFAULTS.b2)

    parser.add_argument(
        "--attack",
        choices=tuple(ORIGINAL_ATTACKS) + (
            "original12", "rot75_nn", "rotation_bilinear", "orthogonal",
            "all_rotations", "paper_all",
        ),
        default="clean",
    )
    parser.add_argument("--angles", nargs="+", type=float)
    parser.add_argument("--fit_start", type=int, default=0)
    parser.add_argument("--fit_count", type=int, default=200)
    parser.add_argument("--eval_start", type=int, default=200)
    parser.add_argument("--eval_count", type=int, default=800)
    parser.add_argument(
        "--whitening_model",
        help="Explicit shared Ledoit-Wolf model path used by every evaluated attack",
    )
    parser.add_argument(
        "--whitening_name", default="W_clean",
        help="Filename stem when fitting without an explicit --whitening_model",
    )
    parser.add_argument(
        "--oracle_attack_whitening", action="store_true",
        help="Diagnostic only: fit/use a separate whitening model for each attack",
    )
    parser.add_argument("--summary_prefix", default="summary")

    parser.add_argument(
        "--num_inference_steps", type=int, default=DEFAULT_INVERSION.num_inference_steps
    )
    parser.add_argument("--invert_prompt", default=DEFAULT_INVERSION.prompt)
    parser.add_argument(
        "--invert_guidance", type=float, default=DEFAULT_INVERSION.guidance_scale
    )
    parser.add_argument(
        "--exact_max_iterations", type=int, default=DEFAULT_INVERSION.exact_max_iterations
    )
    parser.add_argument(
        "--exact_tolerance", type=float, default=DEFAULT_INVERSION.exact_tolerance
    )
    parser.add_argument(
        "--exact_step_size", type=float, default=DEFAULT_INVERSION.exact_step_size
    )
    parser.add_argument(
        "--exact_update", choices=("forward", "gradient"),
        default=DEFAULT_INVERSION.exact_update,
    )
    parser.add_argument(
        "--gnri_max_iterations", type=int, default=DEFAULT_INVERSION.gnri_max_iterations
    )
    parser.add_argument(
        "--gnri_tolerance", type=float, default=DEFAULT_INVERSION.gnri_tolerance
    )
    parser.add_argument("--gnri_lambda", type=float, default=DEFAULT_INVERSION.gnri_lambda)
    parser.add_argument("--gnri_eta", type=float, default=DEFAULT_INVERSION.gnri_eta)
    parser.add_argument(
        "--gnri_step_scale", type=float, default=DEFAULT_INVERSION.gnri_step_scale
    )
    parser.add_argument(
        "--gnri_max_update_norm", type=float,
        default=DEFAULT_INVERSION.gnri_max_update_norm,
    )
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--split_manifest")
    parser.add_argument("--split_name", choices=("fit", "calibration", "test"))
    parser.add_argument("--generation_plan")
    parser.add_argument("--dataset_metadata")
    parser.add_argument("--threshold_model")
    parser.add_argument("--generation_batch_size", type=int, default=1)
    parser.add_argument("--cache_policy", choices=("reuse", "no_cache"), default="reuse")
    parser.add_argument("--runtime_benchmark", action="store_true")
    parser.add_argument("--acceptance_report", help="Clean-commit unit tests and real-model GPU smoke report required by formal mode")
    return parser


if __name__ == "__main__":
    import sys
    arguments = build_parser().parse_args()
    arguments.explicit_arguments = {v.split("=", 1)[0][2:] for v in sys.argv[1:] if v.startswith("--")}
    main(arguments)
