"""Explicit 4090 orchestration. prepare never starts the full experiment.

All subprocesses use the current interpreter and local model files only.
Artifact pointers are captured from completed stages, never chosen by mtime.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from paper_protocol import (PROTOCOL, artifact, load_document, verify_artifact,
    write_once, create_splits, generation_plan, git_state)


def invoke(script, args, log):
    src = Path(__file__).resolve().parent
    log.parent.mkdir(parents=True, exist_ok=True)
    found = {}
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen([sys.executable, "-u", str(src / script), *map(str, args)],
                                   cwd=src, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
            for key in ("WHITENING_MODEL", "THRESHOLD_MODEL", "RUN_MANIFEST"):
                if line.startswith(key + "="):
                    found[key] = line.strip().split("=", 1)[1]
        if process.wait() != 0:
            raise RuntimeError(f"{script} failed; see {log}")
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run", "export"))
    p.add_argument("--output_dir", default="paper_v1")
    p.add_argument("--metadata")
    p.add_argument("--group_field")
    p.add_argument("--model_id")
    p.add_argument("--model_revision")
    p.add_argument("--diff_attack_model_id")
    p.add_argument("--diff_attack_model_revision")
    p.add_argument("--generation_batch_size", type=int, default=1)
    p.add_argument("--whitening_name", choices=("W_clean", "W_robust"), default="W_robust")
    args = p.parse_args()
    repo = Path(__file__).resolve().parent.parent
    git_state(repo, require_clean=True)
    root = Path(args.output_dir).resolve()
    manifest_dir = root / "manifests"
    config_path = manifest_dir / "workflow_config.json"
    split_path, plan_path = manifest_dir / "formal_splits.json", manifest_dir / "generation_plan.json"
    acceptance = root / "logs" / "implementation_acceptance.json"
    if args.command == "prepare":
        if not args.metadata or not args.model_id or not args.diff_attack_model_id:
            p.error("prepare requires --metadata, --model_id and --diff_attack_model_id; use cached local model paths")
        def model_path(value):
            return str(Path(value).resolve()) if Path(value).is_dir() else value
        config = {"protocol": PROTOCOL, "metadata": str(Path(args.metadata).resolve()),
            "model_id": model_path(args.model_id), "model_revision": args.model_revision,
            "diff_attack_model_id": model_path(args.diff_attack_model_id),
            "diff_attack_model_revision": args.diff_attack_model_revision,
            "generation_batch_size": args.generation_batch_size, "whitening_name": args.whitening_name}
        write_once(config_path, config)
        splits = write_once(split_path, create_splits(config["metadata"], group_field=args.group_field))
        write_once(plan_path, generation_plan(splits))
        cmd = ["--gpu_smoke", "--model_id", config["model_id"], "--report", acceptance]
        if config["model_revision"]:
            cmd += ["--model_revision", config["model_revision"]]
        if not acceptance.exists():
            invoke("paper_gate.py", cmd, root / "logs" / "prepare.log")
        if load_document(acceptance)["implementation_acceptance"] != "PASS":
            raise ValueError("Implementation acceptance failed. Read the report; do not start formal run.")
        print("Preparation accepted. Full experiments have NOT started. Explicitly run: paper_workflow.py run")
        return
    config = load_document(config_path)
    common = ["--formal", "--dataset_id", "coco", "--output_dir", root,
        "--split_manifest", split_path, "--generation_plan", plan_path,
        "--dataset_metadata", config["metadata"], "--acceptance_report", acceptance,
        "--model_id", config["model_id"], "--local_files_only", "--torch_dtype", "float32",
        "--generation_batch_size", config["generation_batch_size"], "--whitening_name", config["whitening_name"],
        "--diff_attack_model_id", config["diff_attack_model_id"], "--diff_attack_local_files_only"]
    if config["model_revision"]:
        common += ["--model_revision", config["model_revision"]]
    # For a local attack model, omit a revision argument; the original default
    # fp16 is ignored for local model loading and fingerprinting.
    if config["diff_attack_model_revision"]:
        common += ["--diff_attack_model_revision", config["diff_attack_model_revision"]]
    def stage(q, name, split, extra=()):
        return invoke("research_hsqr.py", common + ["--experiment", q, "--stage", name,
            "--split_name", split, *extra], root / "logs" / f"{q}-{name}-{split}.log")
    def pointer(q, name):
        return manifest_dir / f"{q}-{name}.json"
    if args.command == "run":
        if not (manifest_dir / "pairing_validation.json").exists():
            for q in ("Q0", "Q3"):
                for split in ("fit", "calibration", "test"):
                    stage(q, "generate", split)
            invoke("validate_generation_pairing.py", ["--output_dir", root, "--generation_plan", plan_path], root / "logs" / "pairing.log")
        for q in ("Q0", "Q3"):
            stage(q, "diff_attack", "test")
        for q in ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5"):
            if not pointer(q, "whitening").exists():
                result = stage(q, "fit", "fit")
                write_once(pointer(q, "whitening"), {"artifact": artifact(result["WHITENING_MODEL"])})
            whitening = verify_artifact(load_document(pointer(q, "whitening"))["artifact"])
            if not pointer(q, "thresholds").exists():
                result = stage(q, "calibrate", "calibration", ["--whitening_model", whitening])
                write_once(pointer(q, "thresholds"), {"artifact": artifact(result["THRESHOLD_MODEL"])})
            thresholds = verify_artifact(load_document(pointer(q, "thresholds"))["artifact"])
            if not pointer(q, "run").exists():
                result = stage(q, "evaluate", "test", ["--whitening_model", whitening,
                    "--threshold_model", thresholds, "--attack", "paper_all",
                    "--cache_policy", "no_cache", "--runtime_benchmark"])
                write_once(pointer(q, "run"), {"artifact": artifact(result["RUN_MANIFEST"])})
        print("Raw runs finished. Export is explicit: paper_workflow.py export")
        return
    runs = [verify_artifact(load_document(pointer(q, "run"))["artifact"]) for q in ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5")]
    invoke("paper_export.py", ["--run_manifest", *runs, "--output_dir", root,
           "--bootstrap_resamples", "10000"], root / "logs" / "export.log")
    invoke("paper_gate.py", ["--acceptance_report", acceptance, "--run_manifest", *runs,
        "--export_manifest", root / "export_manifest.json", "--report", root / "logs" / "formal_gate.json"], root / "logs" / "formal-gate.log")


if __name__ == "__main__":
    main()
