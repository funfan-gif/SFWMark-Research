"""Explicit acceptance report. Skips and absent CUDA never count as PASS."""
import argparse
import ast
import io
from pathlib import Path
import unittest

from paper_protocol import (PROTOCOL, artifact, environment, git_state, load_document,
                            model_identity, write_once, verify_artifact, require_equal)


def validate_runtime(run, profiles):
    expected = 2 * len(run["sample_ids"]) * len(run["requested_attacks"])
    for profile in profiles:
        if not profile.get("paper_runtime_eligible"):
            continue
        require_equal(run["signature_payload"], profile["base_protocol"], "Runtime protocol")
        require_equal(profile, {"stage": "evaluate", "cache_policy": "no_cache",
            "warmup_policy": "one_synthetic_gray_image", "batch_size": 1,
            "num_requested": expected, "num_actual_inverted": expected,
            "num_cache_hits": 0, "dtype": "float32"}, "Complete runtime benchmark")
        if profile["inversion_total_seconds"] <= 0 or profile["unet_forward_calls"] <= 0:
            raise ValueError("Runtime benchmark has no measured inversion work")
        return True
    raise ValueError("No complete no-cache/warmed runtime benchmark linked to this run")


def gpu_smoke(model_id, revision):
    """Small, explicitly requested real-model check; never invoked by unit tests."""
    import torch
    from diffusers import DiffusionPipeline, DDIMScheduler
    from PIL import Image
    from inversion import invert_image, InversionConfig
    from dataclasses import fields
    from freeu import configure_freeu
    from research_config import EXPERIMENTS
    from paper_generation import sample_latent
    from hsqr_metrics import extract_hsqr_feature
    from utils import make_hsqr_pattern, inject_hsqr
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable: real-model acceptance cannot pass")
    identity = model_identity(model_id, revision)
    kwargs = {"torch_dtype": torch.float32, "local_files_only": True}
    if revision and identity["kind"] != "local":
        kwargs["revision"] = revision
    pipe = DiffusionPipeline.from_pretrained(model_id, **kwargs).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    latents, images = {}, {}
    pattern = make_hsqr_pattern(7433).unsqueeze(0)
    for group, q in (("G0", "Q0"), ("G1", "Q3")):
        configure_freeu(pipe, EXPERIMENTS[q].generation_freeu)
        no = sample_latent(pipe, 918273)
        wm = inject_hsqr(no, pattern, center=True)
        latents[group] = (no, wm)
        images[group] = pipe(["A ceramic cup on a wooden table"] * 2,
            latents=torch.cat([no, wm]), guidance_scale=7.5, num_inference_steps=50).images
    for i in (0, 1):
        torch.testing.assert_close(latents["G0"][i], latents["G1"][i], rtol=0, atol=0)
    # Exercise injection and extraction through actual CUDA half -> FP32 FFT boundaries.
    half_wm = inject_hsqr(latents["G0"][0].half(), pattern, center=True)
    if not torch.isfinite(extract_hsqr_feature(half_wm)).all():
        raise ValueError("Half FFT boundary failed")
    results = {}
    for name in ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5"):
        q = EXPERIMENTS[name]
        config = InversionConfig(method=q.inversion_method, inversion_freeu=q.inversion_freeu)
        kwargs = {f.name: getattr(config, f.name) for f in fields(config) if f.name != "method"}
        report = []
        recovered = invert_image(pipe, images[q.generation_group], method=config.method, provenance=report, **kwargs)
        if not torch.isfinite(recovered).all():
            raise ValueError(f"{name} non-finite inversion")
        if any(not r["inversion_success"] for r in report):
            raise ValueError(f"{name} numerical inversion failure: {report}")
        if config.method == "gnri":
            alone = invert_image(pipe, images[q.generation_group][0], method=config.method, **kwargs)
            torch.testing.assert_close(alone, recovered[:1], rtol=1e-5, atol=1e-6)
        results[name] = {"finite": True, "provenance": report}
    return {"status": "PASS", "model": identity, "results": results,
            "scope": "one synthetic generated pair per pool; six actual inverter paths; not a robustness experiment"}


def run_tests():
    repo = Path(__file__).resolve().parent.parent
    files = list((repo / "src").rglob("*.py")) + list((repo / "tests").rglob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    suite = unittest.defaultTestLoader.discover(str(repo / "tests"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue())
    return {"syntax_files": len(files), "tests_run": result.testsRun,
            "skipped": [(str(test), reason) for test, reason in result.skipped],
            "failures": [(str(test), reason) for test, reason in result.failures + result.errors],
            "status": "PASS" if result.wasSuccessful() and not result.skipped else "FAIL"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--gpu_smoke", action="store_true")
    p.add_argument("--model_id")
    p.add_argument("--model_revision")
    p.add_argument("--run_manifest", nargs="*")
    p.add_argument("--export_manifest", help="Required for the final paper gate, after locked raw export")
    p.add_argument("--acceptance_report", help="Reuse an unchanged clean-commit/environment real-model smoke report")
    args = p.parse_args()
    repo = Path(__file__).resolve().parent.parent
    code = git_state(repo)
    tests = run_tests()
    smoke = {"status": "NOT_RUN"}
    if args.acceptance_report:
        previous = load_document(args.acceptance_report)
        require_equal(previous, {"code": code, "environment": environment(),
                                "implementation_acceptance": "PASS"}, "Prior acceptance")
        smoke = previous["gpu_smoke"]
    if args.gpu_smoke:
        if not args.model_id:
            p.error("--gpu_smoke requires --model_id (local files only)")
        try:
            smoke = gpu_smoke(args.model_id, args.model_revision)
        except Exception as exc:
            smoke = {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"}
    run_checks = []
    for path in args.run_manifest or []:
        try:
            from paper_export import validate_run
            run, _, _ = validate_run(path)
            require_equal(run["signature_payload"], {"code": code, "environment": environment()},
                          "Final gate run/code/environment")
            import json
            profiles = [json.loads(verify_artifact(ref).read_text()) for ref in run.get("runtime", [])]
            validate_runtime(run, profiles)
            run_checks.append({"status": "PASS", "run": artifact(path), "experiment": run["signature_payload"]["experiment"]})
        except Exception as exc:
            run_checks.append({"status": "FAIL", "path": path, "reason": str(exc)})
    ready = tests["status"] == "PASS" and smoke["status"] == "PASS" and not code["git_dirty"]
    exported = {"status": "NOT_RUN"}
    if args.export_manifest:
        try:
            document = load_document(args.export_manifest)
            if document["status"] != "VALIDATED_RAW_EXPORT" or document["source"]["diagnostic"]:
                raise ValueError("Diagnostic exports cannot pass the final gate")
            if not document["source"].get("figures"):
                raise ValueError("Final paper gate requires the declared figure exports")
            if document["source"]["bootstrap_resamples"] < 10000:
                raise ValueError("Final paper gate requires the declared 10000 bootstrap resamples")
            if {r["sha256"] for r in document["source"]["runs"]} != {r["run"]["sha256"] for r in run_checks if r["status"] == "PASS"}:
                raise ValueError("Export run set differs from validated runs")
            for ref in document["outputs"]:
                verify_artifact(ref)
            exported = {"status": "PASS", "export": artifact(args.export_manifest)}
        except Exception as exc:
            exported = {"status": "FAIL", "reason": str(exc)}
    formal = ready and exported["status"] == "PASS" and len(run_checks) == 6 and {r.get("experiment") for r in run_checks} == {f"Q{i}" for i in range(6)} and all(r["status"] == "PASS" for r in run_checks)
    payload = {"protocol": PROTOCOL, "code": code, "environment": environment(),
               "tests": tests, "gpu_smoke": smoke, "run_checks": run_checks,
               "export_check": exported,
               "implementation_acceptance": "PASS" if ready else "FAIL",
               "formal_experiment_gate": "PASS" if formal else "FAIL"}
    write_once(args.report, payload)
    print("IMPLEMENTATION ACCEPTANCE: " + payload["implementation_acceptance"])
    print("FORMAL EXPERIMENT GATE: " + payload["formal_experiment_gate"])
    if not ready or (args.run_manifest and not formal):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
