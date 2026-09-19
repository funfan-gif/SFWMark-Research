"""Deterministic strict-ROC, calibration-only thresholds and paired bootstrap.

No runner imports: calibration functions cannot access a test split or labels.
All resamples select positive/negative pairs together, by their ordered sample ID.
"""
import math
import numpy as np

METRICS = ("complex_l1", "l2", "diag", "mahalanobis")


def finite_vector(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Missing/non-finite scores or empty sample set")
    return a


def strict_roc(no_distances, wm_distances):
    """All distinct thresholds; ties move together. Equivalent to sklearn ROC."""
    no, wm = finite_vector(no_distances), finite_vector(wm_distances)
    scores = -np.concatenate([no, wm])
    labels = np.concatenate([np.zeros(len(no)), np.ones(len(wm))])
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.r_[0., np.cumsum(labels)[ends]]
    fp = np.r_[0., (1 + ends) - np.cumsum(labels)[ends]]
    tpr, fpr = tp / len(wm), fp / len(no)
    # np.trapz is present on NumPy 1.x; trapezoid on NumPy 2.x.
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return {"AUC": float(integrate(tpr, fpr)),
            "MaxAcc": float(np.max((tpr + 1 - fpr) / 2)),
            "TPR@1%FPR": float(np.max(tpr[fpr < .01]))}


def calibrate_clean_negatives(distances):
    values = np.sort(finite_vector(distances))
    # Decision is distance < threshold. The order statistic at k permits at
    # most k negatives; strict comparison excludes all ties at the boundary.
    # Integer arithmetic avoids float rounding at exactly 1% for n=2000.
    allowed = (len(values) - 1) // 100
    threshold = float(values[allowed])
    fpr = float(np.mean(values < threshold))
    if not fpr < .01:
        raise AssertionError("Calibration violated strict FPR bound")
    return {"threshold": threshold, "calibration_n": len(values),
            "calibration_empirical_fpr": fpr, "decision": "distance < threshold",
            "selection_rule": "sorted_clean_negative[floor((n-1)/100)], strict comparison, ties excluded"}


def point_metrics(no, wm, id_correct, threshold):
    no, wm, correct = finite_vector(no), finite_vector(wm), finite_vector(id_correct)
    if not len(no) == len(wm) == len(correct):
        raise ValueError("Unpaired score arrays")
    tpr, fpr = float(np.mean(wm < threshold)), float(np.mean(no < threshold))
    return {**strict_roc(no, wm), "Id-Acc": float(correct.mean()),
            "Frozen TPR": tpr, "Frozen FPR": fpr,
            "Frozen Balanced Accuracy": (tpr + 1 - fpr) / 2}


def bootstrap_metrics(no, wm, correct, threshold, resamples=10000, seed=1729,
                      comparison=None):
    no, wm, correct = map(finite_vector, (no, wm, correct))
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    n = len(no)
    estimate = point_metrics(no, wm, correct, threshold)
    if comparison is not None:
        bno, bwm, bc, bt = comparison
        if not len(bno) == len(bwm) == len(bc) == n:
            raise ValueError("Paired comparisons require identical ordered sample IDs")
        other = point_metrics(bno, bwm, bc, bt)
        estimate = {k: estimate[k] - other[k] for k in estimate}
    values = {k: [] for k in estimate}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        scores = point_metrics(no[idx], wm[idx], correct[idx], threshold)
        if comparison is not None:
            other = point_metrics(np.asarray(bno)[idx], np.asarray(bwm)[idx], np.asarray(bc)[idx], bt)
            scores = {k: scores[k] - other[k] for k in scores}
        for k, value in scores.items():
            values[k].append(value)
    return {k: {"estimate": v, "ci_low": float(np.quantile(values[k], .025)),
                "ci_high": float(np.quantile(values[k], .975)), "n": n,
                "bootstrap_seed": seed, "bootstrap_resamples": resamples}
            for k, v in estimate.items()}


def coverage(records):
    def successful(row):
        for kind in ("no_wm", "wm"):
            item = row.get(kind, {})
            if not item.get("inversion_success"):
                return False
            for metric in METRICS:
                value = item.get("distances", {}).get(metric)
                if value is None or not math.isfinite(value):
                    return False
                if metric not in item.get("predicted", {}):
                    return False
        return True
    mask = [successful(row) for row in records]
    return {"n_requested": len(records), "n_success": sum(mask),
            "n_failed": len(records) - sum(mask),
            "coverage": sum(mask) / len(records) if records else 0.}, mask
