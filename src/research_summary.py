"""Summarize raw HSQR research distances without changing saved raw results."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve


DISTANCES = ("complex_l1", "l2", "diag", "mahalanobis")
ORIGINAL_12 = (
    "Clean", "Brightness", "Contrast", "JPEG", "Blur", "Noise", "BM3D",
    "VAE-B", "VAE-C", "Diff", "CC", "RC",
)


def _verification(no_wm_distance, wm_distance):
    no_wm_distance = np.asarray(no_wm_distance, dtype=np.float64)
    wm_distance = np.asarray(wm_distance, dtype=np.float64)
    labels = np.concatenate([
        np.zeros(no_wm_distance.size, dtype=np.int64),
        np.ones(wm_distance.size, dtype=np.int64),
    ])
    # Lower distance means a better watermark match, matching detect.py's
    # historical use of negative L1 as the positive-class score.
    scores = -np.concatenate([no_wm_distance, wm_distance])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return {
        "AUC": float(auc(fpr, tpr)),
        "MaxAcc": float(1.0 - np.min((fpr + (1.0 - tpr)) / 2.0)),
        "TPR@1%FPR": float(np.max(tpr[fpr <= 0.01])) if np.any(fpr <= 0.01) else 0.0,
    }


def _metadata(data, path: Path):
    if "metadata_json" in data.files:
        return json.loads(str(data["metadata_json"].item()))
    return {
        "attack_display_name": path.stem,
        "attack_name": "unknown",
        "attack_slug": path.stem,
        "is_original12": False,
    }


def _latest_results(result_dir: Path, whitening_model=None, allow_mixed_models=False):
    selected = {}
    observed_models = set()
    for path in result_dir.glob("distances-*.npz"):
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata(data, path)
        model = metadata.get("whitening_model")
        if model is not None:
            observed_models.add(model)
        if (whitening_model is not None
                and model != whitening_model):
            continue
        key = metadata.get("attack_slug", path.stem)
        if key not in selected or path.stat().st_mtime > selected[key].stat().st_mtime:
            selected[key] = path
    if whitening_model is None and len(observed_models) > 1 and not allow_mixed_models:
        raise ValueError(
            "Results contain multiple whitening models; pass --whitening_model PATH "
            "to avoid mixing detector configurations. Oracle summaries must also "
            "pass --oracle_attack_whitening explicitly."
        )
    return sorted(selected.values())


def summarize_results(result_dir, output_prefix="summary", whitening_model=None,
                      allow_mixed_models=False):
    result_dir = Path(result_dir)
    paths = _latest_results(
        result_dir, whitening_model=whitening_model,
        allow_mixed_models=allow_mixed_models,
    )
    if not paths:
        detail = f" for whitening model {whitening_model}" if whitening_model else ""
        raise FileNotFoundError(f"No distances-*.npz files in {result_dir}{detail}")

    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata(data, path)
            display = metadata.get("attack_display_name", metadata.get("attack_slug", path.stem))
            group = "Original-12" if metadata.get("is_original12", False) else (
                "Rotation" if metadata.get("attack_name") in {
                    "rot75_nn", "rotation_bilinear", "orthogonal"
                } else "Other"
            )
            for metric in DISTANCES:
                required = {
                    f"no_wm_{metric}", f"wm_{metric}", f"id_correct_{metric}"
                }
                if not required.issubset(data.files):
                    continue
                values = _verification(data[f"no_wm_{metric}"], data[f"wm_{metric}"])
                rows.append({
                    "group": group,
                    "attack": display,
                    "metric": metric,
                    **values,
                    "Id-Acc": float(np.asarray(data[f"id_correct_{metric}"], dtype=float).mean()),
                    "identification_candidates": 2048,
                    "raw_file": str(path),
                })

    original_names = {row["attack"] for row in rows if row["group"] == "Original-12"}
    original_complete = set(ORIGINAL_12).issubset(original_names)
    if original_complete:
        for metric in DISTANCES:
            metric_rows = [
                row for row in rows
                if row["group"] == "Original-12" and row["metric"] == metric
                and row["attack"] in ORIGINAL_12
            ]
            rows.append({
                "group": "Original-12",
                "attack": "Original-12 Avg",
                "metric": metric,
                **{
                    key: float(np.mean([row[key] for row in metric_rows]))
                    for key in ("AUC", "MaxAcc", "TPR@1%FPR", "Id-Acc")
                },
                "identification_candidates": 2048,
                "raw_file": "",
            })

    payload = {
        "distance_to_score": "score = -distance",
        "identification": "argmin distance over all 2048 candidate keys",
        "whitening_model_filter": whitening_model,
        "original12_complete": original_complete,
        "missing_original12": sorted(set(ORIGINAL_12) - original_names),
        "rows": rows,
    }
    json_path = result_dir / f"{output_prefix}.json"
    csv_path = result_dir / f"{output_prefix}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    fieldnames = [
        "group", "attack", "metric", "AUC", "MaxAcc", "TPR@1%FPR",
        "Id-Acc", "identification_candidates", "raw_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['group']:11s} | {row['attack']:28s} | {row['metric']:12s} | "
            f"AUC {row['AUC']:.4f} | MaxAcc {row['MaxAcc']:.4f} | "
            f"TPR@1%FPR {row['TPR@1%FPR']:.4f} | Id-Acc {row['Id-Acc']:.4f}"
        )
    if not original_complete:
        print("Original-12 Avg omitted; missing: " + ", ".join(payload["missing_original12"]))
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    return payload


RUNTIME_FIELDS = (
    "experiment", "generation_pool", "stage", "inversion",
    "freeu_generation", "freeu_inversion", "attacks", "num_images", "requested_images",
    "cache_hit_images", "model_load_seconds", "processing_seconds",
    "seconds_per_image", "inversion_processing_seconds",
    "inversion_seconds_per_image", "peak_cuda_allocated_gb",
    "peak_cuda_reserved_gb",
    "unet_forward_calls", "unet_forward_calls_per_image", "inversion_steps",
    "configured_inversion_steps", "total_refinement_iterations",
    "total_newton_iterations", "model_id", "model_revision", "torch_dtype",
    "diff_attack_model_id", "diff_attack_model_revision", "git_commit",
    "model_reused_from_same_process", "runtime_recorded_at_utc", "source_file",
)


def summarize_runtime(runtime_dirs, output_dir):
    """Write runtime profiles separately from verification/ID metrics."""

    paths = set()
    for directory in runtime_dirs:
        directory = Path(directory)
        paths.update(directory.glob("runtime-*.json"))
    rows = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            field: (str(path) if field == "source_file" else payload.get(field))
            for field in RUNTIME_FIELDS
        })
    if not rows:
        searched = ", ".join(str(Path(path)) for path in runtime_dirs)
        raise FileNotFoundError(f"No runtime-*.json files in: {searched}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runtime-summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")
    return rows
