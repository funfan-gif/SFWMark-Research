"""Unified DDIM and research inversion adaptations for Stable Diffusion 2.1.

The VAE path intentionally matches the original SFWMark implementation: resize
to 512, map to [-1, 1], use the posterior mode, and apply the VAE scaling
factor.  Decoder inversion is deliberately outside the scope of this module.
"""

import json
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Union

import torch
import torchvision.transforms as tforms
from PIL import Image
from diffusers import DDIMInverseScheduler, DDIMScheduler

from freeu import configure_freeu
from research_config import FreeUConfig


Prompt = Union[str, Sequence[str]]


@dataclass(frozen=True)
class InversionConfig:
    method: str = "ddim"
    num_inference_steps: int = 50
    guidance_scale: float = 0.0
    prompt: Prompt = ""
    negative_prompt: Optional[Prompt] = None

    # Exact-DPM-inspired first-order forward-step DDIM refinement.  This is not
    # the complete higher-order Exact-DPM pipeline or decoder inversion.
    exact_max_iterations: int = 5
    exact_tolerance: float = 1e-5
    exact_step_size: float = 0.5
    exact_update: str = "forward"

    # GNRI official-semantics adaptation for the SD2.1 DDIM scheduler.
    gnri_max_iterations: int = 2
    gnri_tolerance: float = 1e-4
    gnri_lambda: float = 0.1
    # Official code divides directly by the component gradient.  A non-zero
    # eta is retained only as an explicit numerical-stability experiment.
    gnri_eta: float = 0.0
    gnri_step_scale: float = 1.0
    gnri_max_update_norm: Optional[float] = None
    diagnostics: bool = False

    inversion_freeu: FreeUConfig = FreeUConfig()

    def cache_dict(self):
        result = asdict(self)
        # Diagnostics only change logging, never recovered features.
        result.pop("diagnostics", None)
        result["prompt"] = list(self.prompt) if not isinstance(self.prompt, str) else self.prompt
        if self.negative_prompt is not None and not isinstance(self.negative_prompt, str):
            result["negative_prompt"] = list(self.negative_prompt)
        return result


def _pipe_device(pipe) -> torch.device:
    return torch.device(getattr(pipe, "_execution_device", pipe.device))


def _diagnostic(enabled: bool, event: str, **values) -> None:
    if not enabled:
        return
    serializable = {"event": event}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu()
            serializable[key] = value.tolist() if value.ndim else value.item()
        else:
            serializable[key] = value
    print("RESEARCH_DIAGNOSTIC " + json.dumps(serializable, sort_keys=True))


def _as_image_list(images) -> List[Image.Image]:
    result = [images] if isinstance(images, Image.Image) else list(images)
    if not result or not all(isinstance(image, Image.Image) for image in result):
        raise TypeError("images must be a PIL image or a non-empty sequence of PIL images")
    return result


def _batch_text(value: Optional[Prompt], batch_size: int, default: str = ""):
    if value is None:
        return None
    if isinstance(value, str):
        return [value if value != "" else default] * batch_size
    value = list(value)
    if len(value) != batch_size:
        raise ValueError(f"Expected {batch_size} prompts, received {len(value)}")
    return value


def encode_images(pipe, images, resolution: int = 512) -> torch.Tensor:
    """Run the unchanged SFWMark VAE encoder path."""

    image_list = _as_image_list(images)
    transform = tforms.Compose(
        [tforms.Resize((resolution, resolution)), tforms.ToTensor()]
    )
    image_tensor = torch.stack([2.0 * transform(image) - 1.0 for image in image_list])
    image_tensor = image_tensor.to(device=_pipe_device(pipe), dtype=pipe.unet.dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(image_tensor).latent_dist.mode()
    return latent * pipe.vae.config.scaling_factor


def _encode_prompt(pipe, prompt: Prompt, negative_prompt: Optional[Prompt], batch_size: int,
                   guidance_scale: float) -> torch.Tensor:
    device = _pipe_device(pipe)
    prompts = _batch_text(prompt, batch_size, default="")
    negatives = _batch_text(negative_prompt, batch_size) if negative_prompt is not None else None
    do_cfg = guidance_scale > 1.0

    with torch.no_grad():
        if hasattr(pipe, "encode_prompt"):
            prompt_embeds, negative_embeds = pipe.encode_prompt(
                prompt=prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negatives,
            )
            if do_cfg:
                return torch.cat([negative_embeds, prompt_embeds], dim=0).detach()
            return prompt_embeds.detach()

        # Compatibility with older StableDiffusionPipeline releases.
        return pipe._encode_prompt(prompts, device, 1, do_cfg, negatives).detach()


def _predict_model_output(pipe, scheduler, latent: torch.Tensor, timestep,
                          prompt_embeds: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    do_cfg = guidance_scale > 1.0
    model_input = torch.cat([latent, latent], dim=0) if do_cfg else latent
    model_input = scheduler.scale_model_input(model_input, timestep)
    model_output = pipe.unet(
        model_input,
        timestep,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]
    if do_cfg:
        uncond, text = model_output.chunk(2)
        model_output = uncond + guidance_scale * (text - uncond)
    return model_output


def _baseline_ddim(pipe, images, config: InversionConfig) -> torch.Tensor:
    """Call the original diffusers DDIMInverseScheduler pipeline path."""

    image_list = _as_image_list(images)
    prompts = _batch_text(config.prompt, len(image_list), default="")
    negatives = (
        _batch_text(config.negative_prompt, len(image_list))
        if config.negative_prompt is not None else None
    )
    current_scheduler = pipe.scheduler
    try:
        pipe.scheduler = DDIMInverseScheduler.from_config(current_scheduler.config)
        image_latent = encode_images(pipe, image_list)
        with torch.no_grad():
            return pipe(
                prompt=prompts,
                negative_prompt=negatives,
                latents=image_latent,
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.num_inference_steps,
                output_type="latent",
            ).images
    finally:
        pipe.scheduler = current_scheduler


def _make_schedulers(pipe, num_steps: int):
    inverse = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    forward = DDIMScheduler.from_config(pipe.scheduler.config)
    device = _pipe_device(pipe)
    inverse.set_timesteps(num_steps, device=device)
    forward.set_timesteps(num_steps, device=device)
    return inverse, forward


def _replace_rows(old: torch.Tensor, new: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    view_shape = (mask.shape[0],) + (1,) * (old.ndim - 1)
    return torch.where(mask.view(view_shape), new, old)


def exact_ddim_invert(pipe, images, config: InversionConfig) -> torch.Tensor:
    """Exact-DPM-inspired first-order forward-step DDIM refinement.

    Each step starts from ordinary DDIM inversion, then refines the candidate
    noisy latent so a deterministic DDIM denoising step maps it back to the
    already recovered lower-noise latent.  This is iteration refinement, not an
    increase in the number of DDIM steps.
    """

    if config.exact_max_iterations < 1:
        raise ValueError("exact_max_iterations must be positive")
    if config.exact_update not in {"forward", "gradient"}:
        raise ValueError("exact_update must be 'forward' or 'gradient'")

    latent = encode_images(pipe, images)
    prompt_embeds = _encode_prompt(
        pipe, config.prompt, config.negative_prompt, latent.shape[0], config.guidance_scale
    )
    inverse, forward = _make_schedulers(pipe, config.num_inference_steps)

    for timestep in inverse.timesteps:
        target = latent.detach()
        with torch.no_grad():
            initial_output = _predict_model_output(
                pipe, inverse, target, timestep, prompt_embeds, config.guidance_scale
            )
            candidate = inverse.step(initial_output, timestep, target).prev_sample.detach()

        best = candidate
        best_error = torch.full(
            (candidate.shape[0],), float("inf"), device=candidate.device, dtype=torch.float32
        )
        iterations_run = 0
        for iteration in range(config.exact_max_iterations):
            iterations_run = iteration + 1
            candidate = candidate.detach().requires_grad_(config.exact_update == "gradient")
            context = torch.enable_grad() if config.exact_update == "gradient" else torch.no_grad()
            with context:
                model_output = _predict_model_output(
                    pipe, forward, candidate, timestep, prompt_embeds, config.guidance_scale
                )
                reconstructed = forward.step(
                    model_output, timestep, candidate, eta=0.0
                ).prev_sample
                residual = reconstructed - target
                mse = residual.float().square().flatten(1).mean(dim=1)

            improved = mse < best_error
            best = _replace_rows(best, candidate.detach(), improved)
            best_error = torch.where(improved, mse.detach(), best_error)
            residual_rms = torch.sqrt(mse.detach())
            _diagnostic(
                config.diagnostics,
                "exact_ddim_refinement",
                timestep=int(timestep),
                iteration=iterations_run,
                reconstruction_residual_rms=residual_rms,
                best_residual_rms=torch.sqrt(best_error),
            )
            if torch.sqrt(mse).max().item() <= config.exact_tolerance:
                break

            if config.exact_update == "gradient":
                gradient = torch.autograd.grad(mse.sum(), candidate, only_inputs=True)[0]
                candidate = candidate - config.exact_step_size * gradient
            else:
                # Forward-step update from On Exact Inversion of DPM-Solvers.
                candidate = candidate - config.exact_step_size * residual

        latent = best.detach()
        _diagnostic(
            config.diagnostics,
            "exact_ddim_timestep_complete",
            timestep=int(timestep),
            iteration_count=iterations_run,
            best_residual_rms=torch.sqrt(best_error),
        )
    return latent


def _ddim_timestamp_log_probability(inverse, value: torch.Tensor,
                                    image_latent: torch.Tensor, timestep):
    """DDIM adaptation of GNRI's ``get_timestamp_dist``.

    The official SDXL/Euler implementation evaluates a detached implicit-map
    output under the timestamp Gaussian around the encoded image latent.  A
    DDIM scheduler has no public ``sigmas`` array in that parameterization, so
    this adaptation uses the standard forward-diffusion marginal
    q(z_t | z_0) = N(sqrt(alpha_bar_t) z_0, (1-alpha_bar_t) I).  The returned
    log probability remains detached and only guides the scalar Newton
    numerator, as in the official implementation.
    """

    t = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
    alpha_bar = inverse.alphas_cumprod[t].to(device=value.device, dtype=value.dtype)
    variance = (1.0 - alpha_bar).clamp_min(1e-8)
    mean = alpha_bar.sqrt() * image_latent
    return -0.5 * (value - mean).square() / variance


def gnri_invert(pipe, images, config: InversionConfig) -> torch.Tensor:
    """Official-semantics GNRI adaptation for DDIM / Stable Diffusion 2.1.

    Like the official ICLR 2025 implementation, U-Net prediction and the
    implicit scheduler mapping are evaluated under ``no_grad`` and detached.
    The candidate latent is the sole Newton variable.  This deliberately avoids
    differentiating through the U-Net and is not ordinary latent gradient
    descent.  The scheduler-specific Gaussian is adapted as documented in
    :func:`_ddim_timestamp_log_probability`.
    """

    if config.gnri_max_iterations < 1:
        raise ValueError("gnri_max_iterations must be positive")
    image_latent = encode_images(pipe, images)
    latent = image_latent
    prompt_embeds = _encode_prompt(
        pipe, config.prompt, config.negative_prompt, latent.shape[0], config.guidance_scale
    )
    inverse, _ = _make_schedulers(pipe, config.num_inference_steps)

    for timestep in inverse.timesteps:
        target = latent.detach()
        candidate = target.clone()
        dimension = candidate[0].numel()
        best = candidate
        best_objective = float("inf")

        for iteration in range(config.gnri_max_iterations):
            candidate = candidate.detach().requires_grad_(True)
            # Official GNRI semantics: the learned model and implicit mapping
            # are constants for the component-wise Newton derivative.
            with torch.no_grad():
                model_output = _predict_model_output(
                    pipe, inverse, candidate, timestep, prompt_embeds, config.guidance_scale
                )
                implicit_map = inverse.step(
                    model_output, timestep, target
                ).prev_sample.detach()
                log_probability = _ddim_timestamp_log_probability(
                    inverse, implicit_map, image_latent, timestep
                ).detach()

            root = implicit_map - candidate
            root_residual = root.detach().float().abs().flatten(1).mean(dim=1)
            # Official structure: |f(x)-x| - alpha * log p_t(f(x)).  The
            # detached prior changes the scalar numerator, not its derivative.
            objective_components = root.abs() - config.gnri_lambda * log_probability
            objective = objective_components.float().sum()
            objective_score = objective.detach() / float(objective_components.numel())

            # The official implementation performs one scalar root solve and
            # keeps the complete mapped latent with the lowest mean score.
            if objective_score.item() < best_objective:
                best = implicit_map.detach()
                best_objective = objective_score.item()
            if root_residual.max().item() <= config.gnri_tolerance:
                _diagnostic(
                    config.diagnostics,
                    "gnri_iteration",
                    timestep=int(timestep),
                    iteration=iteration + 1,
                    objective=objective_score,
                    root_residual=root_residual,
                    update_norm=torch.zeros_like(root_residual),
                    non_finite=False,
                )
                break

            gradient = torch.autograd.grad(objective.sum(), candidate, only_inputs=True)[0]
            denominator = gradient + config.gnri_eta
            # Component-wise Newton update from the official code:
            # x <- x - (1 / D) * objective / grad(objective), D=4*64*64.
            update = (objective / float(dimension)) / denominator

            if config.gnri_max_update_norm is not None:
                flat_norm = update.float().flatten(1).norm(dim=1).clamp_min(1e-12)
                scale = (config.gnri_max_update_norm / flat_norm).clamp(max=1.0)
                update = update * scale.view((-1,) + (1,) * (update.ndim - 1))
            next_candidate = candidate - config.gnri_step_scale * update
            update_norm = update.detach().float().flatten(1).norm(dim=1)
            non_finite = not bool(torch.isfinite(next_candidate).all().item())
            _diagnostic(
                config.diagnostics,
                "gnri_iteration",
                timestep=int(timestep),
                iteration=iteration + 1,
                objective=objective_score,
                root_residual=root_residual,
                update_norm=update_norm,
                non_finite=non_finite,
            )
            if non_finite:
                break
            candidate = next_candidate

        latent = best.detach()
    return latent


def invert_image(pipe, images, method: str = "ddim", **kwargs) -> torch.Tensor:
    """Unified inversion entry point returning the recovered final ``z_T``.

    ``method`` is one of ``ddim``, ``exact_ddim``, or ``gnri``.  Keyword
    arguments are fields of :class:`InversionConfig`.
    """

    if "method" in kwargs:
        raise TypeError("Pass method only through invert_image(..., method=...)")
    config = InversionConfig(method=method, **kwargs)
    configure_freeu(pipe, config.inversion_freeu)

    if method == "ddim":
        return _baseline_ddim(pipe, images, config)
    if method == "exact_ddim":
        return exact_ddim_invert(pipe, images, config)
    if method == "gnri":
        return gnri_invert(pipe, images, config)
    raise ValueError(f"Unknown inversion method: {method}")
