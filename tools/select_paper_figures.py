#!/usr/bin/env python3
"""Read-only paper candidate selection. Python 3.10+, Pillow and NumPy only.

Copy this file alone to another computer; no repository imports or models are used.
See --help. CSV schemas match src/freeu_quality_eval.py at commit
3559e5fc908525120fb5e28be3abcbf23384c04c. All three metrics are required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps

VERSION = "1.0"
BASE_COMMIT = "3559e5fc908525120fb5e28be3abcbf23384c04c"
METRICS = {"musiq-ava": "musiq_ava", "laion_aes": "laion_aes",
           "clip": "clip_similarity"}
OUTPUTS = ("overview_contact_sheet.png", "freeu_contact_sheet.png",
           "selected_candidates.csv", "selection_protocol.md")
FEATURES = ("brightness", "contrast", "clipped_fraction", "entropy",
            "edge_strength", "spatial_balance", "rotation_edge_retention")
CANVAS = (512, 512)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode()).hexdigest()


def read_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"{path.name}: required unique columns: {sorted(required)}")
        rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path.name}: empty CSV or malformed row")
    return rows


def sample_id(value: str) -> str:
    # The formal generator writes canonical nonnegative integer filenames.
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"Invalid canonical sample_id: {value!r}")
    return value


def finite(value: str, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}")
    return result


def within(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)


def load_inputs(root: Path, scores: Path, differences: Path, expected: int):
    """Validate the entire wm population, never silently use an intersection."""
    if not root.is_dir():
        raise ValueError(f"Missing data root: {root}")
    rows = read_csv(scores, {"sample_id", "generation_pool", "watermark_status",
                             "image_path", "image_sha256", *METRICS.values()})
    indexed = {}
    for row in rows:
        if row["watermark_status"] not in {"wm", "no_wm"}:
            raise ValueError("Unknown watermark_status in scores")
        if row["watermark_status"] != "wm":
            continue
        sid, pool = sample_id(row["sample_id"]), row["generation_pool"]
        if pool not in {"G0", "G1"} or (sid, pool) in indexed:
            raise ValueError(f"Invalid pool or duplicate wm score: {sid}/{pool}")
        relative = Path(pool) / "coco/HSQR/img_pil_wm" / f"{sid}.png"
        # Old absolute Windows/Linux paths can be relocated, but their suffix
        # must identify exactly the same pool, image kind, and sample.
        recorded = PurePosixPath(row["image_path"].replace("\\", "/"))
        if recorded.parts[-5:] != relative.parts or ".." in recorded.parts:
            raise ValueError(f"Score image_path does not match {relative.as_posix()}")
        path = (root / relative).resolve()
        if not within(path, root) or not path.is_file():
            raise ValueError(f"Missing or unsafe image: {relative.as_posix()}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["image_sha256"]):
            raise ValueError(f"Missing/invalid image_sha256: {sid}/{pool}")
        if sha256(path) != row["image_sha256"]:
            raise ValueError(f"Image SHA256 mismatch: {sid}/{pool}")
        for field in METRICS.values():
            row[field] = finite(row[field], f"{sid}/{pool}/{field}")
        if not 1 <= row["musiq_ava"] <= 10 or not -1 <= row["clip_similarity"] <= 1:
            raise ValueError(f"MUSIQ-AVA/CLIP score outside evaluator range: {sid}/{pool}")
        row["local_path"], row["relative_path"] = path, relative.as_posix()
        indexed[sid, pool] = row
    ids = sorted({sid for sid, _ in indexed}, key=int)
    if len(ids) != expected or len(indexed) != 2 * expected:
        raise ValueError(f"Expected {expected} complete wm pairs; found {len(ids)} IDs / {len(indexed)} images")
    for sid in ids:
        if any((sid, pool) not in indexed for pool in ("G0", "G1")):
            raise ValueError(f"Unpaired wm sample: {sid}")
    # Formal generation pools also contain fit/calibration PNGs (4,024 total).
    # The checked 1,000-row score population, optionally matched to test manifest,
    # defines candidates; other PNGs are inventoried, never scored or selected.
    inventory = {}
    for pool in ("G0", "G1"):
        directory = root / pool / "coco/HSQR/img_pil_wm"
        disk_ids = {p.stem for p in directory.iterdir() if p.suffix.lower() == ".png"}
        if not set(ids).issubset(disk_ids):
            raise ValueError(f"{pool} PNG membership is missing scored wm IDs")
        extra = sorted(disk_ids - set(ids))
        inventory[pool] = {"scored_pngs": len(ids), "unscored_pngs_excluded": len(extra),
                           "excluded_ids_sha256": object_hash(extra)}
    manifest = root / "manifests/formal_splits.json"
    manifest_info = {"status": "not supplied; CSV/PNG membership only, formal provenance not independently established"}
    if manifest.exists():
        document = json.loads(manifest.read_text(encoding="utf-8-sig"))
        members = document["splits"]["test"]
        manifest_ids = [sample_id(str(row["sample_id"])) for row in members]
        if len(set(manifest_ids)) != len(manifest_ids) or set(manifest_ids) != set(ids):
            raise ValueError("formal_splits.json test membership differs from wm scores")
        manifest_info = {"path": str(manifest), "sha256": sha256(manifest),
                         "status": "test membership checked; manifest seals/provenance not independently audited"}
    diff_rows = read_csv(differences, {"sample_id", "watermark_status", "metric",
                                       "g0_score", "g1_score", "difference"})
    seen, confidence = set(), set()
    for row in diff_rows:
        if row["watermark_status"] not in {"wm", "no_wm"}:
            raise ValueError("Unknown watermark_status in differences")
        if row["watermark_status"] != "wm":
            continue
        sid, metric = sample_id(row["sample_id"]), row["metric"]
        key = (sid, metric)
        if sid not in ids or metric not in METRICS or key in seen:
            raise ValueError(f"Unknown/duplicate wm difference: {sid}/{metric}")
        seen.add(key)
        field = METRICS[metric]
        a, b = indexed[sid, "G0"][field], indexed[sid, "G1"][field]
        for label, value in (("g0_score", a), ("g1_score", b), ("difference", b - a)):
            actual = finite(row[label], f"{sid}/{metric}/{label}")
            if not math.isclose(actual, value, rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError(f"Scores/differences disagree: {sid}/{metric}/{label}")
        confidence.add(row.get("pairing_confidence", "unspecified") or "unspecified")
    if seen != {(sid, metric) for sid in ids for metric in METRICS}:
        raise ValueError("Missing wm paired differences for one or more metrics/IDs")
    pairs, signature = [], None
    for sid in ids:
        pair = {"sample_id": sid}
        for pool in ("G0", "G1"):
            row = indexed[sid, pool]
            with Image.open(row["local_path"]) as image:
                image.load()
                spec = (image.format, image.mode, image.size)
                if spec != ("PNG", "RGB", CANVAS):
                    raise ValueError(f"Expected 512x512 RGB PNG: {sid}/{pool}, got {spec}")
                signature = spec
            pair[pool] = row
        pairs.append(pair)
    return pairs, {"score_rows": len(rows), "difference_rows": len(diff_rows),
                   "wm_pairs": len(pairs), "image_spec": signature,
                   "directory_inventory": inventory,
                   "split_manifest": manifest_info,
                   "recorded_pairing_confidence": sorted(confidence),
                   "population_ids_sha256": object_hash(ids),
                   "population_png_sha256": object_hash([
                       [sid, pool, indexed[sid, pool]["image_sha256"]]
                       for sid in ids for pool in ("G0", "G1")])}


def fixed_canvas(image: Image.Image) -> Image.Image:
    """Pixel-for-pixel copy of research_attacks._fixed_black_canvas semantics."""
    if image.size == CANVAS:
        return image.copy()
    canvas = Image.new(image.mode, CANVAS, color=0)
    left, top = max((image.width - 512) // 2, 0), max((image.height - 512) // 2, 0)
    cropped = image.crop((left, top, min(left + 512, image.width), min(top + 512, image.height)))
    canvas.paste(cropped, (max((512 - cropped.width) // 2, 0),
                          max((512 - cropped.height) // 2, 0)))
    return canvas


def rotate_preview(image: Image.Image, angle: float) -> Image.Image:
    # Audited against src/research_attacks.py:100-142 at BASE_COMMIT.
    # Pillow default center=(width/2,height/2); positive angles are CCW.
    return fixed_canvas(image).rotate(float(angle), resample=Image.Resampling.BILINEAR,
                                     expand=False, fillcolor=0)


def image_features(image: Image.Image, angle: float):
    rgb = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR), dtype=float) / 255
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(gray)
    edges = np.hypot(gx, gy)
    cells = edges.reshape(4, 32, 4, 32).mean(axis=(1, 3)).ravel()
    mass = cells / max(float(cells.sum()), 1e-12)
    spatial = float(-(mass * np.log2(np.maximum(mass, 1e-12))).sum() / 4)
    hist = np.histogram(gray, bins=32, range=(0, 1))[0] / gray.size
    entropy = float(-(hist * np.log2(np.maximum(hist, 1e-12))).sum() / 5)
    y, x = np.mgrid[:128, :128] + 0.5 - 64
    theta = math.radians(angle)
    retained = ((np.abs(x * math.cos(theta) - y * math.sin(theta)) <= 64)
                & (np.abs(x * math.sin(theta) + y * math.cos(theta)) <= 64))
    retention = float(edges[retained].sum() / max(float(edges.sum()), 1e-12))
    stats = np.array([gray.mean(), gray.std(), ((gray < .02) | (gray > .98)).mean(),
                      entropy, edges.mean(), spatial, retention])
    # Coarse colour/spatial/edge descriptors are diversity proxies, not semantics.
    colour = np.concatenate([np.histogram(rgb[:, :, c], bins=8, range=(0, 1))[0] / gray.size
                             for c in range(3)])
    layout = gray.reshape(4, 32, 4, 32).mean(axis=(1, 3)).ravel()
    descriptor = np.concatenate([colour, layout, cells])
    return stats, descriptor


def rank01(values):
    """Midranks make ties equal, independent of source row order."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2 / max(len(values) - 1, 1)
        start = end
    return result


def prepare_features(pairs, angle):
    stats, descriptors = [], []
    for pair in pairs:
        current = []
        for pool in ("G0", "G1"):
            with Image.open(pair[pool]["local_path"]) as image:
                current.append(image_features(image, angle))
        stats.append(np.mean([x[0] for x in current], axis=0))
        descriptors.append(np.mean([x[1] for x in current], axis=0))
    stats, descriptors = np.array(stats), np.array(descriptors)
    means = np.array([[np.mean([p[pool][field] for pool in ("G0", "G1")])
                       for field in METRICS.values()] for p in pairs])
    quality = np.mean(np.column_stack([rank01(means[:, c]) for c in range(3)]), axis=1)
    quality_rank = rank01(quality)
    spread = descriptors.std(axis=0)
    descriptors = np.clip((descriptors - descriptors.mean(axis=0)) / np.maximum(spread, 1e-6), -3, 3)
    # Rank normalized proxies balance clarity, tonal variation, spatial structure
    # and survival of structure under rotation; none establishes subject semantics.
    readability = np.mean(np.column_stack([
        rank01(-np.abs(stats[:, 0] - .5)), rank01(stats[:, 1]), rank01(stats[:, 3]),
        rank01(stats[:, 4]), rank01(stats[:, 5]), rank01(stats[:, 6])]), axis=1)
    return stats, descriptors, quality_rank, readability


def select_overview(stats, descriptors, quality, readability, number, rng):
    eligible = ((stats[:, 0] >= .12) & (stats[:, 0] <= .90) & (stats[:, 1] >= .08)
                & (stats[:, 2] <= .45) & (stats[:, 4] >= .008)
                & (stats[:, 5] >= .55) & (stats[:, 6] >= .65)
                & (quality >= .10) & (quality <= .90))
    if int(eligible.sum()) < number:
        raise ValueError(f"Only {int(eligible.sum())} Overview candidates pass fixed thresholds; requested {number}. "
                         "Reduce --num-overview; thresholds are not silently relaxed.")
    # Round-robin over population quality quartiles. Each candidate is evaluated
    # against all previous selections; seeded jitter breaks otherwise exact ties.
    strata = np.minimum((quality * 4).astype(int), 3)
    tie = rng.random(len(quality)) * 1e-9
    selected, remaining = [], set(np.flatnonzero(eligible).tolist())
    while len(selected) < number:
        for stratum in range(4):
            choices = sorted(i for i in remaining if strata[i] == stratum)
            if not choices:
                continue
            diversity = np.ones(len(choices)) if not selected else np.array([
                min(np.linalg.norm(descriptors[i] - descriptors[j]) / math.sqrt(descriptors.shape[1])
                    for j in selected) for i in choices])
            scores = .65 * readability[choices] + .35 * np.minimum(diversity / 2, 1) + tie[choices]
            chosen = choices[int(np.argmax(scores))]
            selected.append(chosen)
            remaining.remove(chosen)
            if len(selected) == number:
                break
    return [(i, "overview", None, None) for i in selected], {
        "eligible": int(eligible.sum()), "eligible_by_quality_quartile": [
            int((eligible & (strata == s)).sum()) for s in range(4)],
        "selected_by_quality_quartile": [sum(strata[i] == s for i in selected) for s in range(4)]}


def select_freeu(pairs, number, rng):
    deltas = np.array([[p["G1"][field] - p["G0"][field] for field in METRICS.values()]
                       for p in pairs])
    # No mean subtraction: the sign remains G1-G0 rather than above/below average.
    scales = deltas.std(axis=0)
    scales = np.where(scales > 1e-10, scales, np.maximum(np.max(np.abs(deltas), axis=0), 1.0))
    normalized = deltas / scales
    effect = normalized.mean(axis=1)
    rms = np.sqrt(np.mean(normalized ** 2, axis=1))
    near = (np.abs(effect) <= .10) & (np.max(np.abs(normalized), axis=1) <= .25)
    masks = {"positive": (effect > 0) & ((deltas > 0).sum(axis=1) >= 2) & ~near,
             "near_zero": near,
             "negative": (effect < 0) & ((deltas < 0).sum(axis=1) >= 2) & ~near}
    selected, info, warnings = [], {}, []
    tie = rng.random(len(pairs))
    for category, mask in masks.items():
        choices = np.flatnonzero(mask).tolist()
        values = rms if category == "near_zero" else effect
        count = min(number, len(choices))
        quantiles = np.linspace(0, .5, count) if category == "near_zero" else np.linspace(.25, .75, count)
        if count == 1:
            quantiles = np.array([0 if category == "near_zero" else .5])
        targets = np.quantile(values[choices], quantiles).tolist() if choices else []
        info[category] = {"eligible": len(choices), "target_quantiles": quantiles.tolist(),
                          "target_values": targets}
        if len(choices) < number:
            warnings.append(f"{category}: only {len(choices)} eligible pairs; requested {number}. No samples invented or relabelled.")
        if category != "near_zero" and 0 < len(choices) < 5:
            warnings.append(f"{category}: sparse sign distribution; conditional quartiles can coincide with population extremes. Inspect manually.")
        available = set(choices)
        for quantile, target in zip(quantiles, targets):
            i = min(available, key=lambda j: (abs(values[j] - target), tie[j], j))
            selected.append((i, category, float(quantile), float(target)))
            available.remove(i)
    return selected, {"delta_scales": dict(zip(METRICS.values(), scales.tolist())),
                       "categories": info, "warnings": warnings}, effect, rms, deltas


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        # Pillow bundles a scalable default in recent versions; old versions
        # have a smaller bitmap default. Both avoid OS-specific font paths.
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def label(draw, xy, text, size=18, fill=(30, 37, 45)):
    draw.text(xy, text, font=font(size), fill=fill)


def paste_thumbnail(sheet, path, box, angle=None):
    with Image.open(path) as original:
        image = rotate_preview(original, angle) if angle is not None else original.copy()
    image = ImageOps.contain(image, (box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    sheet.paste(image, (box[0] + (box[2] - box[0] - image.width) // 2,
                        box[1] + (box[3] - box[1] - image.height) // 2))


def render_overview(pairs, selected, angle, destination):
    width, row_height = 1140, 316
    sheet = Image.new("RGB", (width, 96 + len(selected) * row_height), "#eef1f4")
    draw = ImageDraw.Draw(sheet)
    label(draw, (20, 12), "TAIR-WM | Overview candidates", 25)
    label(draw, (20, 46), f"Watermarked originals + {angle:+g} deg bilinear PREVIEWS | manual content review required", 18)
    for row, (i, _, _, _) in enumerate(selected):
        pair, top = pairs[i], 96 + row * row_height
        label(draw, (20, top), f"#{row + 1:02d} sample_id={pair['sample_id']}", 19)
        columns = [("G0", None), ("G1", None), ("G0", angle), ("G1", angle)]
        for col, (pool, rotation) in enumerate(columns):
            left = 20 + col * 280
            title = f"{pool} wm" if rotation is None else f"{pool} {angle:+g} deg preview"
            label(draw, (left, top + 26), title, 17)
            paste_thumbnail(sheet, pair[pool]["local_path"], (left, top + 52, left + 256, top + 308), rotation)
    sheet.save(destination)


def render_freeu(pairs, selected, effect, destination):
    width, row_height = 1180, 300
    sheet = Image.new("RGB", (width, 100 + max(1, len(selected)) * row_height), "#eef1f4")
    draw = ImageDraw.Draw(sheet)
    label(draw, (20, 12), "TAIR-WM | FreeU paired quality candidates", 25)
    label(draw, (20, 48), "G0=FreeU OFF / G1=FreeU ON | wm originals | delta=G1-G0 | examples, not population estimates", 17)
    if not selected:
        label(draw, (20, 130), "No eligible FreeU categories. See protocol warnings.", 21)
    names = {"musiq_ava": "MUSIQ-AVA", "laion_aes": "LAION-Aes", "clip_similarity": "CLIP"}
    for row, (i, category, quantile, target) in enumerate(selected):
        pair, top = pairs[i], 100 + row * row_height
        for col, pool in enumerate(("G0", "G1")):
            left = 20 + 280 * col
            label(draw, (left, top), f"{pool} wm", 18)
            paste_thumbnail(sheet, pair[pool]["local_path"], (left, top + 28, left + 256, top + 284))
        label(draw, (585, top + 2), f"{category} | sample_id={pair['sample_id']}", 21)
        label(draw, (585, top + 40), f"standardized mean delta={effect[i]:+.4f}", 18)
        label(draw, (585, top + 72), f"conditional q={quantile:.2f}, target={target:+.4f}", 17)
        for m, field in enumerate(METRICS.values()):
            a, b = pair["G0"][field], pair["G1"][field]
            label(draw, (585, top + 116 + m * 46), names[field], 18)
            label(draw, (710, top + 116 + m * 46), f"G0 {a:.5f}  G1 {b:.5f}  d {b-a:+.5f}", 17)
    sheet.save(destination)


def write_candidates(destination, pairs, selections, stats, quality, readability, effect, rms, deltas):
    records = []
    for rank, (i, category, quantile, target) in enumerate(selections, 1):
        pair = pairs[i]
        row = {"selection_rank": rank, "selection": category, "sample_id": pair["sample_id"],
               "target_conditional_quantile": quantile, "target_value": target,
               "quality_percentile": quality[i], "readability_proxy": readability[i],
               "standardized_mean_delta": effect[i], "standardized_delta_rms": rms[i],
               "effect_population_percentile": rank01(effect)[i]}
        for pool in ("G0", "G1"):
            row[f"{pool}_image_path"] = pair[pool]["relative_path"]
            row[f"{pool}_image_sha256"] = pair[pool]["image_sha256"]
        row.update(zip(FEATURES, stats[i].tolist()))
        for m, field in enumerate(METRICS.values()):
            for pool in ("G0", "G1"):
                row[f"{pool}_{field}"] = pair[pool][field]
            row[f"delta_{field}"] = float(deltas[i, m])
            row[f"delta_{field}_population_percentile"] = rank01(deltas[:, m])[i]
        records.append(row)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_protocol(destination, config, audit, overview, freeu, pairs, selections):
    text = f"""# Paper figure candidate selection protocol

Selector version {VERSION}. This is a candidate shortlist for manual review, not a population quality estimate.
{'NON-FORMAL SIZE OVERRIDE: this run is not the requested 1,000-pair selection.' if config['expected_pairs'] != 1000 else 'Expected population: 1,000 matched wm samples; see provenance scope below.'}

## Inputs and validation

Only G0/G1 `coco/HSQR/img_pil_wm/{{sample_id}}.png` are read. All wm score IDs,
directory presence, paired metric records, finite scores, G0/G1 values,
G1-G0 differences (rtol=1e-7, atol=1e-8), 512x512 RGB PNGs and recorded SHA256
are checked before any output. Extra unscored PNGs (e.g. fit/calibration images)
are counted and excluded, never used as candidates. Historical absolute CSV paths are relocated only
after the exact five-component suffix matches. Non-wm rows are excluded.
If available, formal_splits.json test IDs must match exactly. No generation,
latent, watermark detection, Q0-Q5 data, or quality evaluator is executed or changed.
Membership/hash checks do not establish missing historical prompt/seed/key provenance;
recorded pairing_confidence is reported, not upgraded. No GPU or torch imports.

## Overview algorithm (fixed in advance)

Compute 128x128 image proxies for BOTH pools, average symmetrically. Brightness
is luma mean; contrast is luma population std; clipping is fraction <.02 or >.98;
entropy uses 32 gray bins / 5; edge strength is mean gradient magnitude;
spatial balance is normalized entropy of edge mass in a 4x4 grid. Rotation edge
retention estimates source gradient mass remaining inside the rotated frame.
These proxies do not identify objects or prove semantic/spatial intelligibility.

Eligibility: brightness [.12,.90], contrast >=.08, clipped_fraction <=.45,
edge_strength >=.008, spatial_balance >=.55, rotation_edge_retention >=.65.
Average G0/G1 scores for each of MUSIQ-AVA, LAION-Aes and CLIP; average their
population midrank percentiles, then midrank again to obtain quality_percentile.
Keep quality percentiles [.10,.90]; no FreeU gain/detector criterion is used.
Cycle population quality quartiles 0,1,2,3, skipping exhausted strata.
Within each stratum greedily maximize .65*readability + .35*diversity, with seeded
1e-9 jitter for ties. Readability is the mean midrank of brightness proximity
to .5, contrast, gray entropy, edge strength, spatial balance and rotation retention.
Diversity is minimum RMS distance to selected colour histograms (8 bins/channel),
4x4 luma and edge grids, column standardized (std floor 1e-6), clipped [-3,3],
divided by 2 and capped at 1 (first selection diversity=0.5).
Thresholds never relax automatically. Colour/layout diversity is only a proxy
for content diversity; manual inspection must check subjects, duplicates,
representativeness, and whether rotation remains understandable.

## Rotation preview audit

Pixel operations checked against `src/research_attacks.py`, `_fixed_black_canvas`
and `rotate_image(..., interpolation='bilinear')`, base commit `{BASE_COMMIT}`.
Fixed 512x512 black canvas; center crop/pad WITHOUT resizing if needed;
Pillow Image.rotate(angle, BILINEAR, expand=False, fillcolor=0), default
center (256,256), positive counterclockwise, no translation. Formal inputs must
already be 512x512 RGB. Rotation occurs at full resolution BEFORE thumbnailing.
Both G0 and G1 previews are displayed. Thumbnailing uses LANCZOS for display only.
This reproduces the audited rotation pixel transform; it is a PREVIEW, not a
rerun of the formal attack/detection evaluation or evidence of watermark survival.

## FreeU algorithm (fixed in advance)

For each metric use existing paired G1-G0 difference, cross-checked with scores.
Divide by its full-population delta population std WITHOUT subtracting a mean.
For std <=1e-10, use max(max absolute delta,1). Composite effect is mean of the
three standardized deltas; all raw metrics/deltas remain visible, including dissent.
Near-zero requires abs(composite)<=.10 AND EVERY abs(standardized delta)<=.25,
preventing large opposing metrics from cancelling into a false near-zero example.
Positive/negative: composite has that sign AND at least 2/3 raw metrics have that
sign; exclude near-zero. These labels describe metric trends, not human judgments.
Choose nearest unique observed pairs to conditional effect quantiles linearly
spaced [.25,.75] for positive and negative (single candidate uses median).
Near-zero uses conditional RMS standardized delta quantiles [0,.50], single uses
minimum RMS. Targets are computed before selecting; seeded tie breaking,
no extrapolated images or manufactured categories. Sparse/absent categories
are reported. No candidate is reused between FreeU categories; Overview can overlap.
Quantiles are conditional on category, not claims about overall improvement.

## Reproduction metadata

```json
{json.dumps({'configuration': config, 'validation': audit, 'overview': overview, 'freeu': freeu}, indent=2, default=int)}
```

## Selected sample IDs (ordered; overlap between figures is allowed)

"""
    for category in ("overview", "positive", "near_zero", "negative"):
        ids = [pairs[i]["sample_id"] for i, cat, _, _ in selections if cat == category]
        text += f"- {category}: {', '.join(ids) or '(none)'}\n"
    text += "\nAll four outputs are derived artifacts. Source images and score CSVs are never opened for writing.\n"
    destination.write_text(text, encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--data-root", required=True, type=Path, help="Root containing G0/G1/coco/HSQR/img_pil_wm")
    result.add_argument("--scores", required=True, type=Path, help="per_image_scores.csv (all 3 metrics + image SHA256)")
    result.add_argument("--differences", required=True, type=Path, help="Long-format paired_differences.csv")
    result.add_argument("--output-dir", required=True, type=Path, help="New or empty output directory")
    result.add_argument("--rotation", type=float, default=30, help="CCW bilinear preview angle; default +30 degrees")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--num-overview", type=int, default=20)
    result.add_argument("--num-freeu-per-category", type=int, default=3)
    result.add_argument("--expected-pairs", type=int, default=1000,
                        help="Default 1000; explicit overrides mark output NON-FORMAL (e.g. synthetic tests)")
    return result


def run(args):
    if (not math.isfinite(args.rotation) or min(args.expected_pairs, args.num_overview,
                                               args.num_freeu_per_category) < 1 or args.seed < 0):
        raise ValueError("Counts must be positive, seed nonnegative and rotation finite")
    root, scores, differences, output = [p.resolve() for p in
        (args.data_root, args.scores, args.differences, args.output_dir)]
    if args.num_overview > args.expected_pairs:
        raise ValueError("--num-overview cannot exceed --expected-pairs")
    protected = [root / name for name in ("G0", "G1", "Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "manifests")]
    if output == root or any(within(output, p.resolve()) for p in protected):
        raise ValueError("Output must be separate from image/experiment/manifest directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty; existing artifacts are never overwritten")
    pairs, audit = load_inputs(root, scores, differences, args.expected_pairs)
    stats, descriptors, quality, readability = prepare_features(pairs, args.rotation)
    # Independent RNG streams avoid changing one figure when the other count changes.
    streams = np.random.SeedSequence(args.seed).spawn(2)
    overview, overview_info = select_overview(stats, descriptors, quality, readability,
                                             args.num_overview, np.random.default_rng(streams[0]))
    freeu, freeu_info, effect, rms, deltas = select_freeu(pairs, args.num_freeu_per_category,
                                                       np.random.default_rng(streams[1]))
    selections = overview + freeu
    config = {"data_root": str(root), "scores": str(scores), "differences": str(differences),
              "output_dir": str(output), "scores_sha256": sha256(scores),
              "differences_sha256": sha256(differences), "rotation": args.rotation,
              "seed": args.seed, "num_overview": args.num_overview,
              "num_freeu_per_category": args.num_freeu_per_category,
              "expected_pairs": args.expected_pairs, "script_sha256": sha256(Path(__file__)),
              "python": sys.version.split()[0], "pillow": PIL.__version__, "numpy": np.__version__,
              "argv": sys.argv[1:], "rng": "NumPy default_rng / SeedSequence.spawn(2)"}
    # Stage all outputs before publishing; use exclusive creation to protect
    # existing files, including when another run competes for the same output.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".paper-figure-selector-", dir=output.parent) as temporary:
        stage = Path(temporary)
        render_overview(pairs, overview, args.rotation, stage / OUTPUTS[0])
        render_freeu(pairs, freeu, effect, stage / OUTPUTS[1])
        write_candidates(stage / OUTPUTS[2], pairs, selections, stats, quality, readability, effect, rms, deltas)
        write_protocol(stage / OUTPUTS[3], config, audit, overview_info, freeu_info, pairs, selections)
        output.mkdir(exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("Output directory became nonempty during selection")
        for name in OUTPUTS:
            with (output / name).open("xb") as handle:
                handle.write((stage / name).read_bytes())
    for warning in freeu_info["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Validated {len(pairs)} wm pairs; selected {len(overview)} Overview and {len(freeu)} FreeU candidates.")
    print(f"Outputs: {output}")


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError, TypeError, csv.Error, Image.DecompressionBombError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
