"""Q0-Q5 HSQR research runner with reusable recovered-feature caches.

This script is intentionally separate from the baseline detect.py.  It can
generate the requested FreeU variant, fit covariance models on a fitting split,
and evaluate all four distances from each single cached inversion.
"""

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, DiffusionPipeline
from tqdm import tqdm

from feature_cache import FeatureCache, FeatureCacheKey
from hsqr_metrics import HSQRDistanceModel, extract_hsqr_feature, hsqr_reference_feature
from inversion import InversionConfig, invert_image
from research_attacks import apply_rotation_attack
from research_config import EXPERIMENTS, FREEU_DEFAULTS, ROTATION_BILINEAR_ANGLES, FreeUConfig


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
        if self.name == "clean":
            return {"canvas": [512, 512]}
        if self.name == "rot75_nn":
            return {
                "angle": 75.0, "interpolation": "nearest", "expand": False,
                "canvas": [512, 512], "fill": 0,
            }
        if self.name == "orthogonal":
            return {
                "angle": self.angle, "interpolation": "none", "expand": False,
                "canvas": [512, 512], "fill": 0,
            }
        return {
            "angle": self.angle, "interpolation": "bilinear", "expand": False,
            "canvas": [512, 512], "fill": 0,
        }

    @property
    def slug(self):
        if self.name == "clean":
            return "clean"
        angle = f"{self.angle:g}".replace("-", "m").replace(".", "p")
        return f"{self.name}-{angle}"


def _attack_specs(args) -> List[AttackSpec]:
    if args.attack == "clean":
        return [AttackSpec("clean")]
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
        result.extend(AttackSpec("rotation_bilinear", float(a)) for a in ROTATION_BILINEAR_ANGLES)
        result.extend(AttackSpec("orthogonal", float(a)) for a in (90, 180, 270))
        return result
    raise ValueError(args.attack)


def _experiment_dir(args) -> Path:
    root = Path(args.output_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / args.experiment / args.dataset_id / "HSQR"


def _effective_freeu(args, enabled: bool) -> FreeUConfig:
    return FreeUConfig(
        enabled=enabled,
        s1=args.freeu_s1,
        s2=args.freeu_s2,
        b1=args.freeu_b1,
        b2=args.freeu_b2,
    )


def _run_generation(args):
    from generate import main as generate_main

    experiment = EXPERIMENTS[args.experiment]
    generation_freeu = _effective_freeu(args, experiment.generation_freeu.enabled)
    output_root = _experiment_dir(args).parents[1]
    generate_main(SimpleNamespace(
        output_dir=str(output_root),
        dataset_id=args.dataset_id,
        wm_type="HSQR",
        generation_freeu=generation_freeu.enabled,
        freeu_s1=generation_freeu.s1,
        freeu_s2=generation_freeu.s2,
        freeu_b1=generation_freeu.b1,
        freeu_b2=generation_freeu.b2,
    ))


def _load_pipe(args):
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32)
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
        inversion_freeu=inversion_freeu,
    )


def _inversion_kwargs(config: InversionConfig):
    return {
        field.name: getattr(config, field.name)
        for field in fields(InversionConfig)
        if field.name != "method"
    }


def _cache_key(args, config, attack, sample_id, image_kind):
    experiment = EXPERIMENTS[args.experiment]
    generation_freeu = _effective_freeu(args, experiment.generation_freeu.enabled)
    return FeatureCacheKey(
        inversion_method=config.method,
        inversion_hyperparameters=config.cache_dict(),
        generation_freeu=asdict(generation_freeu),
        inversion_freeu=asdict(config.inversion_freeu),
        attack=attack.name,
        attack_parameters=attack.parameters,
        dataset_id=args.dataset_id,
        sample_id=str(sample_id),
        wm_type="HSQR",
        image_kind=image_kind,
    )


def _features_for_pair(pipe, cache, args, config, attack, index):
    keys = {
        kind: _cache_key(args, config, attack, index, kind)
        for kind in ("no_wm", "wm")
    }
    result = {}
    missing_kinds = []
    for kind, key in keys.items():
        if cache.contains(key):
            result[kind] = cache.load(key).numpy()
        else:
            missing_kinds.append(kind)

    if missing_kinds:
        experiment_dir = _experiment_dir(args)
        with Image.open(experiment_dir / "img_pil" / f"{index}.png") as image:
            no_wm = image.convert("RGB")
        with Image.open(experiment_dir / "img_pil_wm" / f"{index}.png") as image:
            wm = image.convert("RGB")
        no_wm, wm = apply_rotation_attack(no_wm, wm, attack.name, attack.angle)
        attacked = {"no_wm": no_wm, "wm": wm}
        missing = [attacked[kind] for kind in missing_kinds]

        recovered = invert_image(
            pipe, missing, method=config.method, **_inversion_kwargs(config)
        )
        features = extract_hsqr_feature(recovered, center=True).detach().cpu().numpy()
        for row, kind in enumerate(missing_kinds):
            cache.store(keys[kind], features[row])
            result[kind] = features[row]
    return result["no_wm"], result["wm"]


def _load_references_and_labels(experiment_dir: Path):
    patterns = torch.load(
        experiment_dir / f"pattern_list-{WM_CAPACITY}.pt", map_location="cpu"
    )
    references = hsqr_reference_feature(patterns).numpy()
    label_files = sorted(experiment_dir.glob("identify_gt_indices_*.npy"))
    if len(label_files) != 1:
        raise RuntimeError(f"Expected one identify_gt_indices file in {experiment_dir}")
    labels = np.load(label_files[0])
    return references, labels


def _configuration_tag(args, config, attack: AttackSpec) -> str:
    digest = _cache_key(
        args, config, attack, "__configuration__", "wm"
    ).digest()[:12]
    return f"{digest}-fit{args.fit_start}-{args.fit_count}"


def _model_path(experiment_dir: Path, args, config, attack: AttackSpec) -> Path:
    tag = _configuration_tag(args, config, attack)
    return experiment_dir / "research_models" / f"whitening-{attack.slug}-{tag}.npz"


def _fit(pipe, cache, args, config, attack, references, labels):
    stop = min(args.fit_start + args.fit_count, len(labels))
    query_features = []
    correct_references = []
    for index in tqdm(range(args.fit_start, stop), desc=f"fit {attack.slug}"):
        _, wm_feature = _features_for_pair(pipe, cache, args, config, attack, index)
        query_features.append(wm_feature)
        correct_references.append(references[int(labels[index])])
    model = HSQRDistanceModel().fit(query_features, correct_references)
    model.save(_model_path(_experiment_dir(args), args, config, attack))


def _evaluate(pipe, cache, args, config, attack, references, labels):
    model = HSQRDistanceModel.load(
        _model_path(_experiment_dir(args), args, config, attack)
    )
    stop = min(args.eval_start + args.eval_count, len(labels))
    output = {
        f"{kind}_{metric}": []
        for metric in DISTANCES
        for kind in ("no_wm", "wm", "predicted", "id_correct")
    }

    for index in tqdm(range(args.eval_start, stop), desc=f"eval {attack.slug}"):
        no_feature, wm_feature = _features_for_pair(
            pipe, cache, args, config, attack, index
        )
        key_index = int(labels[index])
        claimed_reference = references[key_index:key_index + 1]

        no_scores = model.distances(no_feature, claimed_reference, DISTANCES)
        wm_scores = model.distances(wm_feature, claimed_reference, DISTANCES)
        # Identification gets every candidate and no ground-truth hint.
        candidate_scores = model.distances(wm_feature, references, DISTANCES)
        for metric in DISTANCES:
            predicted = int(np.argmin(candidate_scores[metric][0]))
            output[f"no_wm_{metric}"].append(float(no_scores[metric][0, 0]))
            output[f"wm_{metric}"].append(float(wm_scores[metric][0, 0]))
            output[f"predicted_{metric}"].append(predicted)
            output[f"id_correct_{metric}"].append(predicted == key_index)

    result_dir = _experiment_dir(args) / "research_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        result_dir / f"distances-{attack.slug}-{_configuration_tag(args, config, attack)}.npz",
        **{key: np.asarray(value) for key, value in output.items()},
    )


def main(args):
    if args.stage in ("generate", "all"):
        _run_generation(args)
        if args.stage == "generate":
            return

    experiment_dir = _experiment_dir(args)
    references, labels = _load_references_and_labels(experiment_dir)
    config = _inversion_config(args)
    pipe = _load_pipe(args)
    cache = FeatureCache(experiment_dir / "feature_cache")

    for attack in _attack_specs(args):
        if args.stage in ("fit", "all"):
            _fit(pipe, cache, args, config, attack, references, labels)
        if args.stage in ("evaluate", "all"):
            _evaluate(pipe, cache, args, config, attack, references, labels)


def build_parser():
    parser = argparse.ArgumentParser(description="SFWMark HSQR Q0-Q5 research runner")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--stage", choices=("generate", "fit", "evaluate", "all"), required=True)
    parser.add_argument("--dataset_id", choices=("coco", "Gustavo", "DB1k"), required=True)
    parser.add_argument("--output_dir", default="research_outputs")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeu_s1", type=float, default=FREEU_DEFAULTS.s1)
    parser.add_argument("--freeu_s2", type=float, default=FREEU_DEFAULTS.s2)
    parser.add_argument("--freeu_b1", type=float, default=FREEU_DEFAULTS.b1)
    parser.add_argument("--freeu_b2", type=float, default=FREEU_DEFAULTS.b2)

    parser.add_argument(
        "--attack",
        choices=("clean", "rot75_nn", "rotation_bilinear", "orthogonal", "all_rotations"),
        default="clean",
    )
    parser.add_argument("--angles", nargs="+", type=float)
    parser.add_argument("--fit_start", type=int, default=0)
    parser.add_argument("--fit_count", type=int, default=200)
    parser.add_argument("--eval_start", type=int, default=200)
    parser.add_argument("--eval_count", type=int, default=800)

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
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
