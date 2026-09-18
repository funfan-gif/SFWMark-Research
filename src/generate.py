import os
import itertools
import json
from tqdm import tqdm
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from diffusers import DiffusionPipeline, DDIMScheduler

from utils import *
from freeu import configure_freeu
from research_config import FREEU_DEFAULTS, FreeUConfig
from runtime_profiling import RuntimeProfiler

# main 함수
def main(args):
    set_random_seed(42)
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    save_dir = output_root / args.dataset_id / args.wm_type
    os.makedirs(os.path.join(save_dir, "img_pil"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "img_pil_wm"), exist_ok=True)

    # [Datasets]
    meta_annot, prompt_key, gt_folder = get_text_dataset(args.dataset_id)

    # [Evaluation Settings]
    num_dataset = len(meta_annot)
    sample_start = int(getattr(args, "sample_start", 0))
    sample_count = getattr(args, "sample_count", None)
    if sample_start < 0 or sample_start >= num_dataset:
        raise ValueError(f"sample_start must be in [0,{num_dataset}), got {sample_start}")
    sample_stop = num_dataset if sample_count is None else min(
        sample_start + int(sample_count), num_dataset
    )
    if sample_count is not None and int(sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    RANGE_EVAL = range(sample_start, sample_stop)
    w_seed_list = [*range(w_seed, w_seed + wm_capacity)] # 2048 seed numbers
    identify_gt_indices = np.random.choice(wm_capacity, size=num_dataset).tolist()
    np.save(os.path.join(save_dir, f"identify_gt_indices_{num_dataset}.npy"), identify_gt_indices)

    # [Stable-Diffusion-v2-1-base Settings]
    model_id = getattr(args, "model_id", "stabilityai/stable-diffusion-2-1-base")
    model_revision = getattr(args, "model_revision", None)
    local_files_only = bool(getattr(args, "local_files_only", False))
    resolution = 512
    dtype_name = getattr(args, "torch_dtype", "float32")
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]
    target_device = getattr(args, "device", device)
    profiler = RuntimeProfiler(target_device)

    # [Load Stable-Diffusion pipeline]
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "local_files_only": local_files_only,
    }
    if model_revision is not None:
        load_kwargs["revision"] = model_revision
    profiler.start_model_load()
    pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(target_device)
    pipe.set_progress_bar_config(disable=True)
    configure_freeu(pipe, FreeUConfig(
        enabled=getattr(args, "generation_freeu", False),
        s1=getattr(args, "freeu_s1", FREEU_DEFAULTS.s1),
        s2=getattr(args, "freeu_s2", FREEU_DEFAULTS.s2),
        b1=getattr(args, "freeu_b1", FREEU_DEFAULTS.b1),
        b2=getattr(args, "freeu_b2", FREEU_DEFAULTS.b2),
    ))
    profiler.finish_model_load()
    profiler.attach_unet(pipe.unet)

    # [Make GT patterns] wm_capacity=2048
    if args.wm_type == "Tree-Ring":
        masks = tree_masks
        Fourier_watermark_pattern_list = [make_Fourier_treering_pattern(pipe, shape, this_w_seed) for this_w_seed in w_seed_list]
    elif args.wm_type == "RingID":
        # Following the official RingID implementation
        masks = ringid_masks
        single_channel_num_slots = RADIUS - RADIUS_CUTOFF # int(math.log2(wm_capacity))
        key_value_list = [[list(combo) for combo in itertools.product(np.linspace(-64, 64, 2).tolist(), repeat=len(RING_WATERMARK_CHANNEL))] for _ in range(single_channel_num_slots)]
        key_value_combinations = list(itertools.product(*key_value_list))
        Fourier_watermark_pattern_list = [make_Fourier_ringid_pattern(pipe, shape, list(combo), w_seed=w_seed_list[i],
            radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
            ring_watermark_channel=RING_WATERMARK_CHANNEL, heter_watermark_channel=HETER_WATERMARK_CHANNEL,
            heter_watermark_region_mask=heter_watermark_region_mask if len(HETER_WATERMARK_CHANNEL)>0 else None)
            for i, combo in enumerate(key_value_combinations)]
        # A. fix_gt (from official implementation)
        Fourier_watermark_pattern_list = [fft(ifft(Fourier_watermark_pattern).real) for Fourier_watermark_pattern in Fourier_watermark_pattern_list]
        # B. time_shift (from official implementation)
        for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
            Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim=(-1, -2)))
    elif args.wm_type == "HSTR":
        masks = tree_masks
        masks[:, HETER_WATERMARK_CHANNEL] = single_channel_heter_watermark_mask # (64,64) RounderRingMask for Hetero Watermark (noise)
        Fourier_watermark_pattern_list = [make_Fourier_treering_pattern(pipe, shape, this_w_seed, 
            hs=True, center=True, heter=True) for this_w_seed in w_seed_list]
    elif args.wm_type == "HSQR":
        assert box_size == 2
        Fourier_watermark_pattern_list = [make_hsqr_pattern(idx=this_w_seed) for this_w_seed in w_seed_list]
    assert len(Fourier_watermark_pattern_list) == wm_capacity
    
    # [Save Fourier_watermark_pattern_list]
    torch.save(torch.stack(Fourier_watermark_pattern_list, 0).detach(), os.path.join(save_dir, f"pattern_list-{wm_capacity}.pt"))

    save_gt_latents = bool(getattr(args, "save_gt_latents", False))
    if save_gt_latents:
        os.makedirs(save_dir / "gt_latents_no_wm", exist_ok=True)
        os.makedirs(save_dir / "gt_latents_wm", exist_ok=True)

    manifest = {
        "sample_start": sample_start,
        "sample_stop": sample_stop,
        "sample_count": sample_stop - sample_start,
        "dataset_size": num_dataset,
        "model_id": str(model_id),
        "model_revision": model_revision,
        "local_files_only": local_files_only,
        "torch_dtype": dtype_name,
        "generation_freeu": bool(getattr(args, "generation_freeu", False)),
        "freeu": {
            "s1": getattr(args, "freeu_s1", FREEU_DEFAULTS.s1),
            "s2": getattr(args, "freeu_s2", FREEU_DEFAULTS.s2),
            "b1": getattr(args, "freeu_b1", FREEU_DEFAULTS.b1),
            "b2": getattr(args, "freeu_b2", FREEU_DEFAULTS.b2),
        },
        "save_gt_latents": save_gt_latents,
    }
    with open(save_dir / f"generation_manifest-{sample_start}-{sample_stop}.json", "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print("Generation Starts")
    batch_size = 8
    profiler.record_requested_images(2 * len(RANGE_EVAL))
    profiler.start_processing()
    for batch_start in tqdm(range(0, len(RANGE_EVAL), batch_size)):
        batch_indices = RANGE_EVAL[batch_start:batch_start+batch_size]
        batch_size_actual = len(batch_indices) # N
        # File inputs
        gen_prompts = [meta_annot[idx][prompt_key] for idx in batch_indices]
        file_names = [f"{idx}.png" for idx in batch_indices]
        # Set random seeds
        # Seed by the first absolute sample id so G0/G1 stay paired even when
        # generation is executed in a requested sub-range.
        set_random_seed(42 + batch_indices[0])

        with torch.no_grad():
            key_indices = [identify_gt_indices[key] for key in batch_indices]
            pattern_gt_batch = [Fourier_watermark_pattern_list[key_index] for key_index in key_indices]
            # adjust dims of pattern_gt_batch
            if len(pattern_gt_batch[0].shape) == 4:
                pattern_gt_batch = torch.cat(pattern_gt_batch, dim=0) # (N,4,64,64) for Tree-Ring, RingID, HSTR
            elif len(pattern_gt_batch[0].shape) == 3:
                pattern_gt_batch = torch.stack(pattern_gt_batch, dim=0) # (N,c_wm,42,42) for HSQR
            else:
                raise ValueError(f"Unexpected pattern_gt_batch shape: {pattern_gt_batch[0].shape}")
            assert len(pattern_gt_batch.shape) == 4

            # get random latents ~ N(0,I)
            no_watermark_latents = get_random_latents(pipe, batch_size=batch_size_actual) # (N,4,64,64)
            # watermark injection
            if args.wm_type in ["Tree-Ring", "RingID"]:
                Fourier_watermark_latents, _ = inject_wm(no_watermark_latents, pattern_gt_batch, masks, cut_real=True, device=target_device)
            elif args.wm_type == "HSTR":
                Fourier_watermark_latents, _ = inject_wm(no_watermark_latents, pattern_gt_batch, masks, center=True, cut_real=False, device=target_device)
            elif args.wm_type == "HSQR":
                Fourier_watermark_latents = inject_hsqr(
                    no_watermark_latents, pattern_gt_batch, center=True,
                    device=target_device,
                )

            if save_gt_latents:
                for row, idx in enumerate(batch_indices):
                    torch.save(
                        no_watermark_latents[row].detach().cpu(),
                        save_dir / "gt_latents_no_wm" / f"{idx}.pt",
                    )
                    torch.save(
                        Fourier_watermark_latents[row].detach().cpu(),
                        save_dir / "gt_latents_wm" / f"{idx}.pt",
                    )
            
            # generate images
            batched_latents = torch.cat([no_watermark_latents, Fourier_watermark_latents], dim=0) # (2N,4,64,64)
            generated_images = pipe(gen_prompts*2, latents=batched_latents, guidance_scale=7.5,
                num_inference_steps=50, num_images_per_prompt=1).images
            
            # [Free GPU Memory]
            torch.cuda.empty_cache()
        
        # Save images
        img_pils, img_pil_wms = generated_images[:batch_size_actual], generated_images[batch_size_actual:]
        for i, idx in enumerate(batch_indices):
            img_pils[i].save(os.path.join(save_dir, f"img_pil/{file_names[i]}"))
            img_pil_wms[i].save(os.path.join(save_dir, f"img_pil_wm/{file_names[i]}"))

    profiler.record_processed_images(2 * len(RANGE_EVAL))
    profiler.finish_processing()
    profiler.close()
    profiler.save(
        save_dir / "research_results",
        {
            "experiment": getattr(args, "experiment", None),
            "generation_pool": getattr(args, "generation_pool", None),
            "stage": "generate",
            "inversion": None,
            "freeu_generation": bool(getattr(args, "generation_freeu", False)),
            "freeu_inversion": None,
            "freeu": manifest["freeu"],
            "model_id": str(model_id),
            "model_revision": model_revision,
            "torch_dtype": dtype_name,
            "sample_start": sample_start,
            "sample_stop": sample_stop,
            "source_sample_count": len(RANGE_EVAL),
            "git_commit": getattr(args, "git_commit", "unknown"),
        },
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--wm_type", choices=["Tree-Ring", "RingID", "HSTR", "HSQR"], required=True, help="Choose semantic watermarking methods following merged-in-generation scheme")
    parser.add_argument("--dataset_id", choices=["coco", "Gustavo", "DB1k"], required=True, help="Choose dataset_id")
    parser.add_argument("--output_dir", default="outputs", help="output directory: ./[output_dir]/")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_count", type=int)
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--model_revision")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--torch_dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_gt_latents", action="store_true")
    parser.add_argument("--generation_freeu", action="store_true", help="Enable FreeU during generation")
    parser.add_argument("--freeu_s1", type=float, default=FREEU_DEFAULTS.s1)
    parser.add_argument("--freeu_s2", type=float, default=FREEU_DEFAULTS.s2)
    parser.add_argument("--freeu_b1", type=float, default=FREEU_DEFAULTS.b1)
    parser.add_argument("--freeu_b2", type=float, default=FREEU_DEFAULTS.b2)
    args = parser.parse_args()
    main(args)
