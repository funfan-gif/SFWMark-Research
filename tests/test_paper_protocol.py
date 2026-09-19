"""CPU protocol acceptance tests; ML-dependent checks explicitly skip if absent."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import paper_protocol as protocol
from paper_protocol import (digest, write_once, load_document, artifact, create_splits,
    validate_splits, fit_assignment, require_equal, validate_diff_record,
    validate_generation_pairing, seal, generation_plan)
from paper_statistics import (strict_roc, calibrate_clean_negatives, bootstrap_metrics,
                              METRICS, point_metrics)
from paper_export import validate_raw, summarize, export


def has_modules(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def metadata(self, n=80):
        path = self.root / "metadata.json"
        path.write_text(json.dumps({"annotations": [{"id": i, "image_id": i // 2,
            "caption": f"caption {i}"} for i in range(n)]}), encoding="utf-8")
        return path

    def test_split_sample_and_group_disjoint_counts(self):
        counts = {"dev": 4, "fit": 8, "calibration": 12, "test": 6}
        a = create_splits(self.metadata(), counts=counts)
        b = create_splits(self.root / "metadata.json", counts=counts)
        self.assertEqual(a, b)
        self.assertEqual([len(a["splits"][k]) for k in counts], list(counts.values()))
        broken = deepcopy(a)
        broken["splits"]["test"][0]["group_id"] = broken["splits"]["fit"][0]["group_id"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits(broken, counts)
        broken = deepcopy(a)
        broken["splits"]["test"][0]["sample_id"] = broken["splits"]["fit"][0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits(broken, counts)

    def test_split_missing_group_and_insufficient_groups_fail(self):
        path = self.root / "no-group.json"
        path.write_text(json.dumps({"annotations": [{"id": 1, "caption": "a"}]}))
        with self.assertRaisesRegex(ValueError, "Available fields"):
            create_splits(path)
        with self.assertRaisesRegex(ValueError, "Need"):
            create_splits(self.metadata())

    def test_robust_one_attack_per_sample_balanced_and_no75(self):
        rows = [{"sample_id": i} for i in range(1024)]
        a = fit_assignment(rows)
        self.assertEqual(a, fit_assignment(rows[::-1]))
        self.assertEqual(len(a["assignments"]), 1024)
        bins = {}
        for v in a["assignments"].values():
            key = (v["name"], abs(v["angle"]))
            bins[key] = bins.get(key, 0) + 1
            self.assertNotEqual(abs(v["angle"]), 75)
        self.assertEqual(set(bins.values()), {128})
        self.assertTrue(all(v["name"] == "clean" for v in fit_assignment(rows, "W_clean")["assignments"].values()))

    def test_split_claims_must_match_real_metadata(self):
        counts = {"dev": 4, "fit": 8, "calibration": 12, "test": 6}
        with patch.dict(protocol.FORMAL_COUNTS, counts, clear=True):
            manifest = create_splits(self.metadata())
            protocol.validate_split_source(manifest)
            broken = deepcopy(manifest)
            broken["splits"]["test"][0]["group_id"] = "invented-but-disjoint"
            validate_splits(broken)  # Disjointness alone cannot catch this.
            with self.assertRaisesRegex(ValueError, "metadata-derived"):
                protocol.validate_split_source(broken)
            broken = deepcopy(manifest)
            broken["splits"]["fit"][0]["prompt"] = "changed caption"
            with self.assertRaisesRegex(ValueError, "metadata-derived"):
                protocol.validate_split_source(broken)

    def test_runtime_gate_rejects_partial_and_mixed_cache(self):
        from paper_gate import validate_runtime
        run = {"sample_ids": [1, 2], "requested_attacks": ["clean", "jpeg"],
               "signature_payload": {"experiment": "Q0"}}
        profile = {"paper_runtime_eligible": True, "base_protocol": {"experiment": "Q0"},
            "stage": "evaluate", "cache_policy": "no_cache", "warmup_policy": "one_synthetic_gray_image",
            "batch_size": 1, "num_requested": 8, "num_actual_inverted": 8, "num_cache_hits": 0,
            "dtype": "float32", "inversion_total_seconds": 1., "unet_forward_calls": 8}
        self.assertTrue(validate_runtime(run, [profile]))
        for change in ({"num_actual_inverted": 4}, {"num_cache_hits": 1},
                       {"base_protocol": {"experiment": "Q2"}}, {"warmup_policy": "none"}):
            with self.assertRaises(ValueError):
                validate_runtime(run, [{**profile, **change}])

    def test_immutable_and_hash_tampering(self):
        path = self.root / "document.json"
        write_once(path, {"hello": 1})
        write_once(path, {"hello": 1})
        with self.assertRaises(ValueError):
            write_once(path, {"hello": 2})
        path.write_text(path.read_text().replace('"hello":1', '"hello":3'))
        with self.assertRaises(ValueError):
            load_document(path)

    def test_diff_changed_source_config_or_output_fail(self):
        source, output = self.root / "source.png", self.root / "diff.png"
        source.write_bytes(b"source")
        output.write_bytes(b"output")
        record = {"source_image_sha256": protocol.file_hash(source),
                  "source_generation_manifest_sha256": "pool", "config": {"noise": 60},
                  "output": artifact(output)}
        validate_diff_record(record, source, "pool", {"noise": 60})
        with self.assertRaises(ValueError):
            validate_diff_record(record, source, "pool", {"noise": 80})
        source.write_bytes(b"regenerated")
        with self.assertRaises(ValueError):
            validate_diff_record(record, source, "pool", {"noise": 60})

    def test_pairing_rejects_latent_key_prompt_mismatch(self):
        blob = self.root / "blob.npy"
        blob.write_bytes(b"test-artifact")
        row = {"sample_id": 0, "prompt_hash": "p", "key_index": 12, "latent_seed": 42}
        record = {**row, "no_wm_latent_hash": protocol.file_hash(blob), "wm_latent_hash": protocol.file_hash(blob),
                  "images": {k: artifact(blob) for k in ("no_wm", "wm")},
                  "latents": {k: artifact(blob) for k in ("no_wm", "wm")}}
        common = {k: "same" for k in ("generation_protocol", "model", "steps", "guidance", "model_dtype", "fft_dtype", "vae_slicing", "code", "environment", "geometry")}
        common.update(generation_plan_hash="plan", freeu=protocol.FREEU, records={"0": record})
        a = {**deepcopy(common), "generation_pool": "G0", "freeu_enabled": False}
        b = {**deepcopy(common), "generation_pool": "G1", "freeu_enabled": True}
        plan = {"sha256": "plan", "records": [row]}
        self.assertTrue(validate_generation_pairing(a, b, plan))
        for field in ("no_wm_latent_hash", "wm_latent_hash", "key_index", "prompt_hash"):
            bad = deepcopy(b)
            bad["records"]["0"][field] = "changed"
            with self.assertRaises(ValueError):
                validate_generation_pairing(a, bad, plan)

    def test_wrong_whitening_protocol_hard_error(self):
        from paper_runner import validate_whitening
        path = self.root / "W.npz"
        path.write_bytes(b"placeholder")
        split = {"splits": {"fit": [{"sample_id": 1}, {"sample_id": 2}]}}
        write_once(path.with_suffix(".meta.json"), {"base_protocol": {"experiment": "Q0"},
            "fit_sample_ids_hash": digest([1, 2]), "fitting_residual_count": 2})
        with self.assertRaisesRegex(ValueError, "Whitening model incompatible"):
            validate_whitening(path, {"experiment": "Q2"}, split)

    def test_strict_roc_boundary(self):
        # One negative ranks above every positive: ROC reaches TPR=1 at FPR=.01.
        # Strict original definition must exclude that point.
        result = strict_roc(np.r_[-1., np.repeat(2., 99)], np.repeat(0., 100))
        self.assertEqual(result["TPR@1%FPR"], 0.)

    def test_calibration_ties_and_no_test_access(self):
        for values in (np.ones(2000), np.arange(2000), np.repeat(np.arange(100), 20)):
            result = calibrate_clean_negatives(values)
            self.assertLess(result["calibration_empirical_fpr"], .01)
            self.assertEqual(result, calibrate_clean_negatives(values[::-1]))
        import inspect
        self.assertEqual(list(inspect.signature(calibrate_clean_negatives).parameters), ["distances"])
        self.assertEqual(calibrate_clean_negatives(np.arange(2000))["threshold"], 19.)

    def toy_raw(self, attack="clean"):
        run = {"protocol_signature": "sig", "sample_ids": [1, 2, 3, 4],
               "sample_groups": {str(i): str(i) for i in range(1, 5)},
               "signature_payload": {"experiment": "Q0", "display_name": "DDIM"}}
        records = []
        for sid in run["sample_ids"]:
            r = {"sample_id": sid, "group_id": str(sid), "protocol_signature": "sig", "experiment": "Q0", "attack": attack, "gt_key": sid}
            for kind in ("wm", "no_wm"):
                r[kind] = {"inversion_success": True, "distances": {m: float(sid + (10 if kind == "no_wm" else 0)) for m in METRICS}, "predicted": {m: sid for m in METRICS}}
            records.append(r)
        return run, {"protocol_signature": "sig", "sample_ids": run["sample_ids"],
                     "attack": {"slug": attack}, "records": records}

    def test_raw_failure_nan_missing_and_mixed_protocol_rejected(self):
        run, raw = self.toy_raw()
        validate_raw(run, raw)
        bad = deepcopy(raw)
        bad["records"][0]["wm"]["inversion_success"] = False
        with self.assertRaisesRegex(ValueError, "failed"):
            validate_raw(run, bad)
        counts, _ = validate_raw(run, bad, True)
        self.assertEqual(counts["n_failed"], 1)
        self.assertEqual(counts["coverage"], .75)
        bad = deepcopy(raw)
        bad["records"][0]["wm"]["distances"]["l2"] = float("nan")
        with self.assertRaises(ValueError):
            validate_raw(run, bad)
        bad = deepcopy(raw)
        bad["records"].pop()
        with self.assertRaises(ValueError):
            validate_raw(run, bad)
        bad = deepcopy(raw)
        bad["protocol_signature"] = "different steps"
        with self.assertRaises(ValueError):
            validate_raw(run, bad)

    def test_original12_rotation_and_reproducible_bootstrap(self):
        from paper_export import ORIGINAL12
        attacks = list(ORIGINAL12) + ["rot75_nn"]
        run, _ = self.toy_raw()
        run["requested_attacks"] = attacks
        raw = {a: self.toy_raw(a)[1] for a in attacks}
        threshold = {"thresholds": {m: {"threshold": 7.} for m in METRICS}}
        a = summarize(run, raw, threshold, resamples=10)
        b = summarize(run, raw, threshold, resamples=10)
        self.assertEqual(a, b)
        self.assertEqual(len([r for r in a if r["attack"] == "Original-12 Avg"]), 4)
        self.assertTrue(all(r["group"] == "Rotation" for r in a if r["attack"] == "rot75_nn"))
        for m in METRICS:
            avg = next(r for r in a if r["attack"] == "Original-12 Avg" and r["metric"] == m)
            self.assertEqual(avg["statistics"]["Id-Acc"]["estimate"], 1.)
        del raw["diff"]
        with self.assertRaises((ValueError, KeyError)):
            summarize(run, raw, threshold, resamples=10)

    def test_paired_bootstrap_same_indices(self):
        no, wm, correct = np.arange(8) + 2, np.arange(8), np.arange(8) % 2
        result = bootstrap_metrics(no, wm, correct, 4., resamples=20,
                                   comparison=(no, wm, correct, 4.))
        self.assertTrue(all(v["estimate"] == v["ci_low"] == v["ci_high"] == 0 for v in result.values()))

    def test_legacy_without_evidence_not_promoted(self):
        from paper_generation import audit_legacy
        self.assertEqual(audit_legacy()["status"], "legacy_not_formal")

    def test_formal_parameter_locks(self):
        from paper_runner import formal_guard
        from types import SimpleNamespace
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            formal_guard(SimpleNamespace(experiment="D0"))
        with self.assertRaisesRegex(ValueError, "float32"):
            formal_guard(SimpleNamespace(experiment="Q0", torch_dtype="float16"))

    def test_raw_summary_export_end_to_end_deterministic(self):
        """Synthetic small manifests exercise the real validator and exporter.

        Only required split counts are reduced, not hashing or validation logic.
        No diffusion or actual covariance fitting is represented by this fixture.
        """
        from paper_export import ORIGINAL12, validate_run
        counts = {"dev": 2, "fit": 2, "calibration": 4, "test": 4}
        with patch.dict(protocol.FORMAL_COUNTS, counts, clear=True):
            split = create_splits(self.metadata(), counts=counts)
            sp = self.root / "splits.json"; write_once(sp, split)
            plan = generation_plan(split)
            pp = self.root / "plan.json"; write_once(pp, plan)
            blob = self.root / "blob"; blob.write_bytes(b"synthetic fixture only")
            bref = artifact(blob)
            records = {}
            for row in plan["records"]:
                records[str(row["sample_id"])] = {**row, "images": {k:bref for k in ("wm","no_wm")},
                    "latents": {k:bref for k in ("wm","no_wm")}, "wm_latent_hash": bref["sha256"], "no_wm_latent_hash": bref["sha256"]}
            code={"git_commit":"fixture", "git_dirty":False}; env={"fixture":True}
            common={"protocol":protocol.PROTOCOL,"generation_protocol":protocol.GENERATION,
                "generation_plan_hash":plan["sha256"], "model":{"fixture":True}, "steps":50,"guidance":7.5,
                "model_dtype":"float32","fft_dtype":"float32","vae_slicing":True,"code":code,"environment":env,
                "geometry":protocol.GEOMETRY,"freeu":protocol.FREEU,"records":records,"patterns":bref}
            pools={}
            for g in ("G0","G1"):
                p=self.root/f'{g}.json'; write_once(p,{**common,"generation_pool":g,"freeu_enabled":g=="G1"}); pools[g]=artifact(p)
            pairing=self.root/'pairing.json'; write_once(pairing,{"protocol":protocol.PROTOCOL,"status":"PASS","pools":pools,"generation_plan":artifact(pp)})
            base={"protocol":protocol.PROTOCOL,"experiment":"Q0","display_name":"DDIM baseline",
                "generation_pool":"G0","generation_manifest_hash":load_document(pools['G0']['path'])['sha256'],
                "generation_plan_hash":plan['sha256'],"split_manifest_hash":split['sha256'],"dataset_hash":split['dataset_hash'],
                "metric_version":protocol.METRIC_VERSION,"code":code,"environment":env}
            acceptance=self.root/'acceptance.json'; write_once(acceptance,{"protocol":protocol.PROTOCOL,"implementation_acceptance":"PASS","code":code,"environment":env})
            assignment=fit_assignment(split['splits']['fit'],'W_clean'); ap=self.root/'assignment.json'; write_once(ap,assignment)
            wp=self.root/'W.npz'; wp.write_bytes(b'fake covariance bytes, no fitting claimed')
            write_once(wp.with_suffix('.meta.json'),{"base_protocol":base,"fit_sample_ids_hash":digest([r['sample_id'] for r in split['splits']['fit']]),
                "fitting_residual_count":2,"whitening_model_sha256":protocol.file_hash(wp),"assignment":artifact(ap),"fit_attack_assignment_hash":assignment['sha256']})
            calraw=self.root/'calibration-raw.json'; write_once(calraw,{"base_protocol":base,"split":"calibration","records":[{**r,"provenance":{"inversion_success":True,"image_kind":"no_wm"},"distances":{m:10.+i for m in METRICS}} for i,r in enumerate(split['splits']['calibration'])]})
            tp=self.root/'thresholds.json'; write_once(tp,{"base_protocol":base,"calibration_attack":"clean",
                "whitening_model_sha256":protocol.file_hash(wp),"calibration_sample_ids_hash":digest([r['sample_id'] for r in split['splits']['calibration']]),
                "raw_calibration":artifact(calraw),"thresholds":{m:calibrate_clean_negatives([10.,11.,12.,13.]) for m in METRICS}})
            testrows=split['splits']['test']; ids=[r['sample_id'] for r in testrows]
            diffrefs=[]; diffconfig={"noise_step":60}
            for sid in ids:
                for kind in ('no_wm','wm'):
                    p=self.root/f'diff-{sid}-{kind}.json'; write_once(p,{"sample_id":sid,"image_kind":kind,
                        "source_image_sha256":bref['sha256'],"source_generation_manifest_sha256":base['generation_manifest_hash'],"config":diffconfig,"output":bref})
                    diffrefs.append(artifact(p))
            sig={**base,"split":"test","test_sample_ids_hash":digest(ids),"whitening_model_sha256":protocol.file_hash(wp),
                "threshold_model_sha256":protocol.file_hash(tp),"diff_provenance_hash":digest(diffrefs)}
            rawrefs={}
            for attack in ORIGINAL12:
                rr=[]
                for i,row in enumerate(testrows):
                    key=records[str(row['sample_id'])]['key_index']
                    item={"sample_id":row['sample_id'],"group_id":row['group_id'],"gt_key":key,"experiment":"Q0","attack":attack,"protocol_signature":digest(sig)}
                    for kind in ('no_wm','wm'):
                        item[kind]={"inversion_success":True,"distances":{m:float(i+(20 if kind=='no_wm' else 0)) for m in METRICS},"predicted":{m:key for m in METRICS}}
                    rr.append(item)
                npz=self.root/f'raw-{attack}.npz'
                aa={"sample_ids":np.array(ids),"protocol_signature":np.array(digest(sig))}
                for m in METRICS:
                    for kind in ('wm','no_wm'): aa[f'{kind}_{m}']=np.array([r[kind]['distances'][m] for r in rr])
                    aa[f'id_correct_{m}']=np.ones(len(ids),dtype=bool)
                np.savez_compressed(npz,**aa)
                p=self.root/f'raw-{attack}.json'; write_once(p,{"protocol_signature":digest(sig),"sample_ids":ids,"records":rr,"attack":{"slug":attack},"npz":artifact(npz)}); rawrefs[attack]=artifact(p)
            rp=self.root/'run.json'; write_once(rp,{"protocol":protocol.PROTOCOL,"signature_payload":sig,"protocol_signature":digest(sig),"run_id":"synthetic",
                "sample_ids":ids,"sample_groups":{str(r['sample_id']):r['group_id'] for r in testrows},"requested_attacks":list(ORIGINAL12),"raw_results":rawrefs,
                "split_manifest":artifact(sp),"generation_plan":artifact(pp),"pairing_validation":artifact(pairing),"whitening":artifact(wp),"whitening_metadata":artifact(wp.with_suffix('.meta.json')),
                "thresholds":artifact(tp),"acceptance_report":artifact(acceptance),"diff_records":diffrefs,"diff_config":diffconfig})
            validate_run(rp)
            first=export([rp],self.root/'export-a',resamples=10,figures=False)
            second=export([rp],self.root/'export-a',resamples=10,figures=False)
            self.assertEqual(first,second)
            export([rp],self.root/'export-b',resamples=10,figures=False)
            self.assertEqual((self.root/'export-a/tables/main_results.csv').read_bytes(),(self.root/'export-b/tables/main_results.csv').read_bytes())
            bad=load_document(rp); bad['raw_results'].pop('diff'); badpath=self.root/'bad-run.json'; write_once(badpath,bad)
            with self.assertRaises(ValueError): validate_run(badpath)


@unittest.skipUnless(has_modules("torch", "diffusers", "torchvision"), "Torch/Diffusers/torchvision not installed; no dependency installation authorized")
class TorchAcceptanceTests(unittest.TestCase):
    def test_gnri_batch_independence(self):
        import torch
        import inversion
        from PIL import Image
        from types import SimpleNamespace
        a, b = Image.new("RGB", (512, 512), "white"), Image.new("RGB", (512, 512), "black")
        cfg = inversion.InversionConfig(method="gnri", gnri_lambda=0.)
        scheduler = SimpleNamespace(timesteps=[0], step=lambda output, timestep, target: SimpleNamespace(prev_sample=.5 * output + 1))
        def encode(pipe, images):
            return torch.stack([torch.full((4, 2, 2), float(im.getpixel((0, 0))[0]) / 255) for im in images])
        with patch.object(inversion, "encode_images", side_effect=encode), patch.object(inversion, "_encode_prompt", return_value=None), patch.object(inversion, "_make_schedulers", return_value=(scheduler, None)), patch.object(inversion, "_predict_model_output", side_effect=lambda pipe, sched, latent, *args: latent), patch.object(inversion, "_ddim_timestamp_log_probability", side_effect=lambda sched, value, *args: torch.zeros_like(value)):
            single = inversion.gnri_invert(None, a, cfg)
            batch = inversion.gnri_invert(None, [a, b], cfg)
            # These are exactly the cache-hit/miss input shapes used by the legacy runner.
            retry = inversion.gnri_invert(None, [a], cfg)
            torch.testing.assert_close(single, batch[:1])
            torch.testing.assert_close(single, retry)
            reports = []
            inversion.gnri_invert(None, [a, b], cfg, reports)
            self.assertEqual(len(reports), 2)
            self.assertEqual(reports[0]["gnri_execution"], "per_sample_v2")
            # Exercise the actual legacy cache dispatcher (without importing its
            # unrelated SciPy/attack dependencies) against the real GNRI wrapper.
            source = Path(__file__).resolve().parents[1] / 'src/research_hsqr.py'
            node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_features_for_pair')
            ns = {'_load_attacked_pair':lambda *args:(a,b), '_image_sha256':lambda im:'pixels',
                  '_cache_key':lambda args,config,attack,index,kind,sha:kind,
                  '_inversion_kwargs':lambda config:{},
                  'invert_image':lambda pipe,images,**kw:inversion.gnri_invert(pipe,images,cfg),
                  'extract_hsqr_feature':lambda z,**kw:z.flatten(1)}
            exec(compile(ast.Module(body=[node],type_ignores=[]), str(source), 'exec'), ns)
            class Cache:
                def __init__(self,values): self.values=values
                def contains(self,key): return key in self.values
                def load(self,key): return torch.as_tensor(self.values[key])
                def store(self,key,value): self.values[key]=value
            all_miss=ns['_features_for_pair'](None,Cache({}),None,cfg,None,0)
            partial=ns['_features_for_pair'](None,Cache({'wm':all_miss[1]}),None,cfg,None,0)
            np.testing.assert_allclose(all_miss[0],partial[0],rtol=1e-6,atol=1e-7)
            self.assertEqual(cfg.cache_dict()['gnri_execution'],'per_sample_v2')

    def test_generation_partition_invariance(self):
        import torch
        from paper_generation import sample_latent
        from types import SimpleNamespace
        pipe = SimpleNamespace(device="cpu", unet=SimpleNamespace(config=SimpleNamespace(in_channels=4), dtype=torch.float32),
            prepare_latents=lambda batch, channels, h, w, dtype, device, gen: torch.randn((batch, channels, h//8, w//8), generator=gen, dtype=dtype))
        torch.manual_seed(99)
        before = torch.get_rng_state().clone()
        full = torch.cat([sample_latent(pipe, 42 + i) for i in range(8)])
        retry = sample_latent(pipe, 45)
        shuffled = torch.cat([sample_latent(pipe, 42 + i) for i in [7, 3, 1]])
        torch.testing.assert_close(full[3:4], retry, rtol=0, atol=0)
        torch.testing.assert_close(full[3:4], shuffled[1:2], rtol=0, atol=0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    @unittest.skipUnless(has_modules("scipy", "sklearn"), "SciPy/sklearn unavailable")
    def test_half_44_fft_and_full_identification(self):
        import torch
        from hsqr_metrics import extract_hsqr_complex, HSQRDistanceModel
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            x = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            f = extract_hsqr_complex(x)
            self.assertEqual(f.shape, (1, 42, 21))
            self.assertEqual(f.dtype, torch.complex64)
            source=Path(__file__).resolve().parents[1]/'src/utils.py'
            names={'inject_hsqr','qr_abs','rfft','irfft'}
            nodes=[n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
            ns={'torch':torch,'center_slice':(slice(None),slice(None),slice(10,54),slice(10,54)),
                'HSQR_WATERMARK_CHANNEL':[3],'delta':0}
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
            injected=ns['inject_hsqr'](x,torch.ones((1,1,42,42),dtype=torch.bool,device=device),center=True,device=device)
            self.assertEqual(injected.dtype,torch.float16)
            self.assertTrue(torch.isfinite(injected).all())
        # All 2048 keys are evaluated; GT cannot be passed into identify().
        bank = np.zeros((2048, 1764)); bank[:, 0] = np.arange(2048)
        self.assertEqual(HSQRDistanceModel().identify(bank[1731], bank, "l2")[0], 1731)


@unittest.skipUnless(has_modules("torch", "scipy", "sklearn"), "Torch/SciPy/sklearn unavailable")
class WhiteningTests(unittest.TestCase):
    def test_ledoitwolf_cholesky_consistency(self):
        from hsqr_metrics import HSQRDistanceModel
        rng = np.random.default_rng(7)
        refs = rng.normal(size=(8, 1764))
        q = refs + rng.normal(size=(8, 1764)) + .3
        model = HSQRDistanceModel().fit(q, refs)
        direct = model.distances(q[:1], refs[:3])
        prepared = model.distances_to_prepared(q[:1], model.prepare_references(refs[:3]))
        for metric in METRICS:
            np.testing.assert_allclose(direct[metric], prepared[metric])
        v = q[0] - refs[0] - model.residual_mean
        expected = np.linalg.norm(np.linalg.solve(model.cholesky, v))
        self.assertAlmostEqual(direct["mahalanobis"][0, 0], expected)


if __name__ == "__main__":
    unittest.main()
