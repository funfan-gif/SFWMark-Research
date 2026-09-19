"""Audit locked raw runs, paired bootstrap, and deterministic paper exports.

No directory search, latest-file selection, or test-time threshold optimization.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import numpy as np

from paper_protocol import (PROTOCOL, METRIC_VERSION, artifact, digest, load_document,
    require_equal, verify_artifact, write_once, validate_splits, validate_generation_pairing,
    validate_split_source)
from paper_statistics import METRICS, bootstrap_metrics, coverage

ORIGINAL12 = ("clean", "brightness", "contrast", "jpeg", "blur", "noise", "bm3d",
              "vae_b", "vae_c", "diff", "cc", "rc")


def validate_raw(run, raw, allow_failed=False):
    require_equal(raw, {"protocol_signature": run["protocol_signature"],
                        "sample_ids": run["sample_ids"]}, "Raw result")
    records = raw["records"]
    if [r["sample_id"] for r in records] != run["sample_ids"]:
        raise ValueError("Missing, duplicated, or reordered raw samples")
    if len(set(run["sample_ids"])) != len(run["sample_ids"]):
        raise ValueError("Duplicate requested sample IDs")
    for row in records:
        require_equal(row, {"protocol_signature": run["protocol_signature"],
            "experiment": run["signature_payload"]["experiment"], "attack": raw["attack"]["slug"],
            "group_id": run["sample_groups"][str(row["sample_id"])]}, "Raw sample")
        if row["gt_key"] not in range(2048):
            raise ValueError("Invalid GT key")
        for kind in ("no_wm", "wm"):
            for value in row[kind].get("predicted", {}).values():
                if value not in range(2048):
                    raise ValueError("Invalid 2048-key prediction")
    counts, mask = coverage(records)
    if counts["n_failed"] and not allow_failed:
        raise ValueError(f"Formal summary refused: {counts['n_failed']} failed/missing/NaN samples")
    # Raw JSON and retained NPZ must agree, not just each have a valid hash.
    if "npz" in raw:
        with np.load(verify_artifact(raw["npz"]), allow_pickle=False) as data:
            if data["sample_ids"].tolist() != run["sample_ids"] or str(data["protocol_signature"].item()) != run["protocol_signature"]:
                raise ValueError("NPZ sample/protocol mismatch")
            for m in METRICS:
                for kind in ("no_wm", "wm"):
                    expected = np.array([r[kind]["distances"].get(m, np.nan) for r in records])
                    if not np.array_equal(data[f"{kind}_{m}"], expected, equal_nan=True):
                        raise ValueError("NPZ and raw JSON scores disagree")
                expected = [r["wm"]["predicted"].get(m) == r["gt_key"] for r in records]
                if not np.array_equal(data[f"id_correct_{m}"], expected):
                    raise ValueError("NPZ and raw JSON identification disagree")
    return counts, mask


def validate_run(path, allow_failed=False):
    run = load_document(path)
    require_equal(run, {"protocol": PROTOCOL}, "Run manifest")
    signature = run["signature_payload"]
    acceptance = load_document(verify_artifact(run["acceptance_report"]))
    require_equal(acceptance, {"protocol": PROTOCOL, "implementation_acceptance": "PASS",
                              "code": signature["code"], "environment": signature["environment"]}, "Acceptance provenance")
    if digest(signature) != run["protocol_signature"]:
        raise ValueError("Protocol signature does not match payload")
    if signature["experiment"] not in {f"Q{i}" for i in range(6)}:
        raise ValueError("Diagnostic configuration cannot enter a paper table")
    if signature["metric_version"] != METRIC_VERSION or signature["code"]["git_dirty"]:
        raise ValueError("Wrong metric version or dirty code provenance")
    splits = load_document(verify_artifact(run["split_manifest"]))
    validate_splits(splits)
    validate_split_source(splits)
    if signature["split_manifest_hash"] != splits["sha256"]:
        raise ValueError("Wrong split manifest")
    ids = [r["sample_id"] for r in splits["splits"]["test"]]
    if run["sample_ids"] != ids or signature["test_sample_ids_hash"] != digest(ids):
        raise ValueError("Run is not the full locked test split")
    plan = load_document(verify_artifact(run["generation_plan"]))
    pairing = load_document(verify_artifact(run["pairing_validation"]))
    require_equal(pairing, {"status": "PASS", "protocol": PROTOCOL}, "Pairing gate")
    require_equal(plan, {"sha256": signature["generation_plan_hash"],
                        "split_manifest_hash": splits["sha256"],
                        "dataset_hash": signature["dataset_hash"]}, "Generation plan binding")
    if load_document(verify_artifact(pairing["generation_plan"]))["sha256"] != plan["sha256"]:
        raise ValueError("Pairing gate refers to another generation plan")
    pools = {g: load_document(verify_artifact(pairing["pools"][g])) for g in ("G0", "G1")}
    validate_generation_pairing(pools["G0"], pools["G1"], plan)
    pool = pools[signature["generation_pool"]]
    for p in pools.values():
        verify_artifact(p["patterns"])
    if pool["sha256"] != signature["generation_manifest_hash"]:
        raise ValueError("Wrong frozen generation pool")
    whitening = verify_artifact(run["whitening"])
    if signature["whitening_model_sha256"] != run["whitening"]["sha256"]:
        raise ValueError("Whitening signature differs from the referenced model")
    if verify_artifact(run["whitening_metadata"]) != whitening.with_suffix(".meta.json"):
        raise ValueError("Whitening metadata does not belong to the referenced model")
    from paper_runner import validate_whitening
    excluded = {"split", "test_sample_ids_hash", "whitening_model_sha256", "threshold_model_sha256", "diff_provenance_hash"}
    base = {k: v for k, v in signature.items() if k not in excluded}
    validate_whitening(whitening, base, splits)
    threshold_path = verify_artifact(run["thresholds"])
    thresholds = load_document(threshold_path)
    require_equal(thresholds, {"base_protocol": base, "calibration_attack": "clean",
        "whitening_model_sha256": run["whitening"]["sha256"],
        "calibration_sample_ids_hash": digest([r["sample_id"] for r in splits["splits"]["calibration"]])}, "Threshold provenance")
    if signature["threshold_model_sha256"] != run["thresholds"]["sha256"]:
        raise ValueError("Wrong frozen threshold artifact")
    calibration = load_document(verify_artifact(thresholds["raw_calibration"]))
    require_equal(calibration, {"base_protocol": base, "split": "calibration"}, "Calibration raw protocol")
    if any(not r.get("provenance", {}).get("inversion_success") or r["provenance"].get("image_kind") != "no_wm" for r in calibration["records"]):
        raise ValueError("Calibration raw records must be successful no-watermark inversions")
    from paper_statistics import calibrate_clean_negatives
    if [r["sample_id"] for r in calibration["records"]] != [r["sample_id"] for r in splits["splits"]["calibration"]]:
        raise ValueError("Wrong calibration sample set")
    for m in METRICS:
        calculated = calibrate_clean_negatives([r["distances"][m] for r in calibration["records"]])
        require_equal(thresholds["thresholds"][m], calculated, "Threshold recomputation")
    if set(run["requested_attacks"]) != set(run["raw_results"]):
        raise ValueError("Missing requested attack result")
    missing = set(ORIGINAL12) - set(run["raw_results"])
    if missing:
        raise ValueError(f"Original-12 incomplete; no Avg permitted. Missing: {sorted(missing)}")
    diff_pairs = []
    for ref in run.get("diff_records", []):
        from paper_protocol import validate_diff_record
        record = load_document(verify_artifact(ref))
        diff_pairs.append((record["sample_id"], record["image_kind"]))
        source = pool["records"][str(record["sample_id"])]["images"][record["image_kind"]]["path"]
        validate_diff_record(record, source, pool["sha256"], run["diff_config"])
    if len(run.get("diff_records", [])) != 2 * len(ids):
        raise ValueError("Complete test Diff provenance is required")
    if set(diff_pairs) != {(sid, kind) for sid in ids for kind in ("no_wm", "wm")}:
        raise ValueError("Diff provenance sample set differs from test")
    if signature.get("diff_provenance_hash") != digest(run["diff_records"]):
        raise ValueError("Diff provenance signature mismatch")
    raw = {}
    for name, ref in run["raw_results"].items():
        raw[name] = load_document(verify_artifact(ref))
        if "npz" not in raw[name]:
            raise ValueError("Formal raw result must retain its original NPZ")
        if name != raw[name]["attack"]["slug"]:
            raise ValueError("Attack manifest/filename mismatch")
        validate_raw(run, raw[name], allow_failed)
        for row in raw[name]["records"]:
            if row["gt_key"] != pool["records"][str(row["sample_id"])]["key_index"]:
                raise ValueError("GT label differs from generation plan")
    return run, raw, thresholds


def arrays(records, metric, mask=None):
    selected = records if mask is None else [r for r, ok in zip(records, mask) if ok]
    return (np.array([r["no_wm"]["distances"][metric] for r in selected]),
            np.array([r["wm"]["distances"][metric] for r in selected]),
            np.array([r["wm"]["predicted"][metric] == r["gt_key"] for r in selected], dtype=float))


def summarize(run, raw, thresholds, resamples=10000, seed=1729, allow_failed=False):
    if set(ORIGINAL12) - set(raw):
        raise ValueError("Original-12 incomplete; no Avg permitted")
    if set(run["requested_attacks"]) != set(raw):
        raise ValueError("Missing or extra raw attacks")
    rows = []
    for attack in run["requested_attacks"]:
        value = raw[attack]
        counts, mask = validate_raw(run, value, allow_failed)
        for m in METRICS:
            statistics = {}
            if counts["n_success"]:
                no, wm, correct = arrays(value["records"], m, mask)
                statistics = bootstrap_metrics(no, wm, correct, thresholds["thresholds"][m]["threshold"], resamples, seed)
            rows.append({"experiment": run["signature_payload"]["experiment"],
                "display_name": run["signature_payload"]["display_name"], "attack": attack,
                "group": "Original-12" if attack in ORIGINAL12 else "Rotation",
                "metric": m, **counts, "statistics": statistics,
                "diagnostic": bool(counts["n_failed"] or allow_failed),
                "score_population": "successful_pairs_only" if counts["n_failed"] else "all_requested_pairs",
                "conservative_id_accuracy": sum(r["wm"].get("predicted", {}).get(m) == r["gt_key"] and r["wm"].get("inversion_success", False) for r in value["records"]) / len(value["records"]),
                "n_nonconverged": sum(r[k].get("converged") is False for r in value["records"] for k in ("wm", "no_wm"))})
    # Point estimates of the original Avg are arithmetic means of exactly 12
    # attacks. CIs per attack above use paired sample bootstraps; Avg CI is
    # deliberately not fabricated from averaging per-attack CI endpoints.
    if all(coverage(raw[a]["records"])[0]["n_failed"] == 0 for a in ORIGINAL12):
        for m in METRICS:
            selected = [r for r in rows if r["attack"] in ORIGINAL12 and r["metric"] == m]
            rows.append({"experiment": run["signature_payload"]["experiment"],
                "display_name": run["signature_payload"]["display_name"],
                "attack": "Original-12 Avg", "group": "Original-12", "metric": m,
                "n_requested": len(run["sample_ids"]), "n_success": len(run["sample_ids"]),
                "n_failed": 0, "coverage": 1., "diagnostic": allow_failed,
                "statistics": {k: {"estimate": float(np.mean([r["statistics"][k]["estimate"] for r in selected])),
                    "ci_low": None, "ci_high": None, "n": len(run["sample_ids"]),
                    "ci_status": "not_requested_for_original12_average"} for k in selected[0]["statistics"]}})
    return rows


def csv_bytes(rows):
    flat = []
    for row in rows:
        for measure, stats in row["statistics"].items():
            flat.append({k: row.get(k) for k in ("experiment", "display_name", "group", "attack", "metric", "n_requested", "n_success", "n_failed", "coverage", "diagnostic", "score_population", "conservative_id_accuracy", "n_nonconverged")} | {"measure": measure, **stats})
    fields = sorted(set().union(*(r.keys() for r in flat))) if flat else ["experiment"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat)
    return buffer.getvalue().encode("utf-8")


def write_bytes_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Export conflict: {path}")
    else:
        with path.open("xb") as f:
            f.write(data)


def paired_comparisons(validated, resamples, seed):
    output = []
    if len(validated) < 2:
        return output
    anchor, others = validated[0], validated[1:]
    for other in others:
        a, b = anchor[0], other[0]
        for k in ("sample_ids", "sample_groups"):
            if a[k] != b[k]:
                raise ValueError("Paired comparisons require identical ordered test samples/groups")
        for k in ("split_manifest_hash", "generation_plan_hash", "dataset_hash", "metric_version", "code"):
            if a["signature_payload"][k] != b["signature_payload"][k]:
                raise ValueError(f"Incompatible method comparison: {k}")
        for attack in sorted(set(anchor[1]) & set(other[1])):
            for m in METRICS:
                no, wm, correct = arrays(other[1][attack]["records"], m)
                bno, bwm, bc = arrays(anchor[1][attack]["records"], m)
                stats = bootstrap_metrics(no, wm, correct, other[2]["thresholds"][m]["threshold"],
                    resamples, seed, comparison=(bno, bwm, bc, anchor[2]["thresholds"][m]["threshold"]))
                output.append({"difference": f"{b['signature_payload']['experiment']} minus {a['signature_payload']['experiment']}",
                               "attack": attack, "metric": m, "statistics": stats})
    return output


def export(paths, output, resamples=10000, seed=1729, allow_failed=False, figures=True):
    validated = [validate_run(path, allow_failed) for path in paths]
    if len({v[0]["signature_payload"]["experiment"] for v in validated}) != len(validated):
        raise ValueError("Choose one locked run per Q; do not mix multiple configurations of the same Q")
    source = {"runs": [artifact(p) for p in paths], "bootstrap_resamples": resamples,
              "bootstrap_seed": seed, "diagnostic": allow_failed, "figures": figures}
    root = Path(output)
    if allow_failed:
        root = root / "diagnostic"
    manifest_path = root / "export_manifest.json"
    if manifest_path.exists():
        existing = load_document(manifest_path)
        require_equal(existing, {"source": source}, "Existing export")
        for ref in existing["outputs"]:
            verify_artifact(ref)
        return existing
    all_rows = []
    for run, raw, thresholds in validated:
        all_rows.extend(summarize(run, raw, thresholds, resamples, seed, allow_failed))
    comparisons = [] if allow_failed else paired_comparisons(validated, resamples, seed)
    files = []
    for name, selected in (("main_results", [r for r in all_rows if r["group"] == "Original-12"]),
                           ("rotation_results", [r for r in all_rows if r["group"] == "Rotation"])):
        target = root / "tables" / f"{name}.csv"
        write_bytes_once(target, csv_bytes(selected))
        files.append(artifact(target))
        target = target.with_suffix(".json")
        write_once(target, {"source": source, "rows": selected})
        files.append(artifact(target))
    target = root / "tables" / "method_differences.json"
    write_once(target, {"source": source, "comparisons": comparisons})
    files.append(artifact(target))
    runtime_rows = []
    for run, _, _ in validated:
        for ref in run.get("runtime", []):
            payload = json.loads(verify_artifact(ref).read_text(encoding="utf-8"))
            runtime_rows.append({"run_id": run["run_id"], **{k: payload.get(k) for k in (
                "stage", "paper_runtime_eligible", "cache_policy", "warmup_policy", "batch_size", "gpu", "torch",
                "cuda_runtime", "dtype", "model_load_seconds", "inversion_total_seconds",
                "inversion_seconds_per_processed_image", "num_requested", "num_cache_hits", "num_actual_inverted",
                "unet_forward_calls", "unet_calls_per_actually_inverted_image", "peak_cuda_allocated_gb", "peak_cuda_reserved_gb")}})
    buffer = io.StringIO(newline="")
    fields = sorted(set().union(*(r.keys() for r in runtime_rows))) if runtime_rows else ["run_id", "paper_runtime_eligible"]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(runtime_rows)
    target = root / "tables" / "runtime_results.csv"
    write_bytes_once(target, buffer.getvalue().encode())
    files.append(artifact(target))
    if figures:
        files.extend(plot_figures(root, all_rows))
    return write_once(manifest_path, {"source": source, "run_ids": [v[0]["run_id"] for v in validated],
        "protocol_signatures": [v[0]["protocol_signature"] for v in validated],
        "raw_result_hashes": {v[0]["run_id"]: {k: ref["sha256"] for k, ref in v[0]["raw_results"].items()} for v in validated},
        "git_commits": sorted({v[0]["signature_payload"]["code"]["git_commit"] for v in validated}),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "outputs": files,
        "status": "DIAGNOSTIC" if allow_failed else "VALIDATED_RAW_EXPORT"})


def plot_figures(root, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.hashsalt"] = "sfw-paper-v1"
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    refs = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for q in sorted({r["experiment"] for r in rows}):
        for metric in ("complex_l1", "mahalanobis"):
            selected = [r for r in rows if r["experiment"] == q and r["metric"] == metric and r["attack"].startswith("rotation_bilinear-") and r["statistics"]]
            selected.sort(key=lambda r: float(r["attack"].split("rotation_bilinear-")[1].replace("m", "-").replace("p", ".")))
            if selected:
                x = [float(r["attack"].split("rotation_bilinear-")[1].replace("m", "-").replace("p", ".")) for r in selected]
                ax.plot(x, [r["statistics"]["Id-Acc"]["estimate"] for r in selected], marker="o", label=f"{q} / {metric}")
    ax.set(xlabel="Rotation angle (degrees, bilinear)", ylabel="2048-key identification accuracy", ylim=(0, 1))
    if ax.lines:
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "rotation_curve.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    refs.append(artifact(path))
    fig, ax = plt.subplots(figsize=(10, 5))
    selected = [r for r in rows if r["attack"] == "Original-12 Avg"]
    ax.bar([f"{r['experiment']}\n{r['metric']}" for r in selected], [r["statistics"]["Id-Acc"]["estimate"] for r in selected])
    ax.set(ylabel="Original-12 mean identification accuracy", ylim=(0, 1))
    ax.tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    path = out / "method_comparison.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    refs.append(artifact(path))
    return refs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_manifest", nargs="+", required=True)
    p.add_argument("--output_dir", default="paper_v1")
    p.add_argument("--bootstrap_resamples", type=int, default=10000)
    p.add_argument("--bootstrap_seed", type=int, default=1729)
    p.add_argument("--allow_failed_samples", action="store_true")
    p.add_argument("--no_figures", action="store_true")
    args = p.parse_args()
    export(args.run_manifest, args.output_dir, args.bootstrap_resamples,
           args.bootstrap_seed, args.allow_failed_samples, not args.no_figures)
    print("DIAGNOSTIC EXPORT" if args.allow_failed_samples else "LOCKED RAW EXPORT VALIDATION: PASS")


if __name__ == "__main__":
    main()
