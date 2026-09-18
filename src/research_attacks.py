"""Original SFWMark attacks plus separately reported rotation diagnostics."""

import io
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance


CANVAS_SIZE = (512, 512)

ORIGINAL_ATTACKS = (
    "clean", "brightness", "contrast", "jpeg", "blur", "noise", "bm3d",
    "vae_b", "vae_c", "diff", "cc", "rc",
)

ATTACK_DISPLAY_NAMES = {
    "clean": "Clean",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "jpeg": "JPEG",
    "blur": "Blur",
    "noise": "Noise",
    "bm3d": "BM3D",
    "vae_b": "VAE-B",
    "vae_c": "VAE-C",
    "diff": "Diff",
    "cc": "CC",
    "rc": "RC",
    "rot75_nn": "Rot75",
    "rotation_bilinear": "Rotation-Bilinear",
    "orthogonal": "Orthogonal-Rotation",
}

BASELINE_ATTACK_PARAMETERS = {
    "brightness": {"brightness_factor": 6},
    "contrast": {"contrast_factor": 0.5},
    "jpeg": {"jpeg_ratio": 25},
    "blur": {"gaussian_blur_r": 5},
    "noise": {"gaussian_std": 0.05},
    "bm3d": {"bm3d_sigma": 0.1},
    "vae_b": {"vaeb_quality": 3},
    "vae_c": {"vaec_quality": 3},
    "diff": {"precomputed": "img_pil-diffatt_fp16"},
    "cc": {"center_crop_area_ratio": 0.5},
    "rc": {"random_crop_area_ratio": 0.7},
}

_COMPRESSION_MODELS = {}


def attack_display_name(name: str, angle: float = 0.0) -> str:
    base = ATTACK_DISPLAY_NAMES[name]
    if name in {"rotation_bilinear", "orthogonal"}:
        return f"{base}({angle:g})"
    return base


def attack_parameters(name: str, angle: float = 0.0):
    if name == "clean":
        return {"canvas": [512, 512]}
    if name in BASELINE_ATTACK_PARAMETERS:
        return dict(BASELINE_ATTACK_PARAMETERS[name])
    if name == "rot75_nn":
        return {
            "angle": 75.0, "interpolation": "nearest", "expand": False,
            "canvas": [512, 512], "fill": 0,
        }
    if name == "orthogonal":
        return {
            "angle": angle, "interpolation": "none", "expand": False,
            "canvas": [512, 512], "fill": 0,
        }
    if name == "rotation_bilinear":
        return {
            "angle": angle, "interpolation": "bilinear", "expand": False,
            "canvas": [512, 512], "fill": 0,
        }
    raise ValueError(f"Unknown attack: {name}")


def _set_baseline_seed(seed: int) -> None:
    """Match ``utils.set_random_seed`` without importing its eager GPU models."""

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + 1)
        torch.cuda.manual_seed_all(seed + 2)
        torch.cuda.manual_seed_all(seed + 4)
    np.random.seed(seed + 3)
    random.seed(seed + 5)


def _fixed_black_canvas(image: Image.Image, size=CANVAS_SIZE) -> Image.Image:
    """Center-crop/pad to a fixed canvas without resampling pixels."""

    if image.size == size:
        return image.copy()
    canvas = Image.new(image.mode, size, color=0)
    src_left = max((image.width - size[0]) // 2, 0)
    src_top = max((image.height - size[1]) // 2, 0)
    src_right = min(src_left + size[0], image.width)
    src_bottom = min(src_top + size[1], image.height)
    cropped = image.crop((src_left, src_top, src_right, src_bottom))
    dst_left = max((size[0] - cropped.width) // 2, 0)
    dst_top = max((size[1] - cropped.height) // 2, 0)
    canvas.paste(cropped, (dst_left, dst_top))
    return canvas


def rotate_image(image: Image.Image, angle: float, interpolation: str = "bilinear",
                 orthogonal_exact: bool = False) -> Image.Image:
    """Rotate on a fixed 512x512 black canvas with ``expand=False``."""

    image = _fixed_black_canvas(image)
    normalized = int(angle) % 360 if float(angle).is_integer() else None
    if orthogonal_exact:
        if normalized == 0:
            return image
        transpose = {
            90: Image.Transpose.ROTATE_90,
            180: Image.Transpose.ROTATE_180,
            270: Image.Transpose.ROTATE_270,
        }
        if normalized not in transpose:
            raise ValueError("Exact orthogonal rotation only supports 0/90/180/270 degrees")
        return image.transpose(transpose[normalized])

    resampling = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
    }
    interpolation = interpolation.lower()
    if interpolation not in resampling:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    return image.rotate(
        angle=float(angle), resample=resampling[interpolation],
        expand=False, fillcolor=0,
    )


def rotate_pair(img_no_wm: Image.Image, img_wm: Image.Image, angle: float,
                interpolation: str = "bilinear", orthogonal_exact: bool = False
                ) -> Tuple[Image.Image, Image.Image]:
    return (
        rotate_image(img_no_wm, angle, interpolation, orthogonal_exact)
        if img_no_wm is not None else None,
        rotate_image(img_wm, angle, interpolation, orthogonal_exact),
    )


def rot75_nn(img_no_wm: Image.Image, img_wm: Image.Image):
    return rotate_pair(img_no_wm, img_wm, 75.0, interpolation="nearest")


def rotation_bilinear(img_no_wm: Image.Image, img_wm: Image.Image, angle: float):
    return rotate_pair(img_no_wm, img_wm, angle, interpolation="bilinear")


def orthogonal_rotation(img_no_wm: Image.Image, img_wm: Image.Image, angle: int):
    return rotate_pair(
        img_no_wm, img_wm, angle, interpolation="nearest", orthogonal_exact=True
    )


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    result = Image.open(buffer).convert("RGB")
    result.load()
    return result


def _compression_model(kind: str, device: str):
    key = (kind, str(device))
    if key not in _COMPRESSION_MODELS:
        from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor

        factory = bmshj2018_hyperprior if kind == "vae_b" else cheng2020_anchor
        _COMPRESSION_MODELS[key] = factory(quality=3, pretrained=True).to(device).eval()
    return _COMPRESSION_MODELS[key]


def _compression_attack(image: Image.Image, kind: str, device: str) -> Image.Image:
    import torch
    import torchvision.transforms as tforms

    model = _compression_model(kind, device)
    tensor = tforms.Compose([tforms.Resize((512, 512)), tforms.ToTensor()])(
        image
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        encoded = model.compress(tensor)
        decoded = model.decompress(encoded["strings"], encoded["shape"])["x_hat"]
    return tforms.ToPILImage()(decoded.squeeze().cpu())


def _random_crop_original_position(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    cropped = image.crop((left, top, left + crop_size, top + crop_size))
    padded = Image.new(image.mode, CANVAS_SIZE, 0)
    padded.paste(cropped, (left, top))
    return padded


def apply_attack_pair(img_no_wm: Image.Image, img_wm: Image.Image, attack: str,
                      angle: float = 0.0, seed: int = 42,
                      device: str = "cuda"):
    """Apply one original-baseline or rotation attack to an image pair.

    Numeric parameters and random seeding match ``detect.py`` and
    ``utils.image_distortion``.  ``diff`` expects the caller to provide the
    already regenerated pair from the baseline Diff attack directories.
    """

    _set_baseline_seed(seed)
    img_no_wm = img_no_wm.convert("RGB") if img_no_wm is not None else None
    img_wm = img_wm.convert("RGB")

    if attack in {"clean", "diff"}:
        return _fixed_black_canvas(img_no_wm), _fixed_black_canvas(img_wm)
    if attack == "rot75_nn":
        return rot75_nn(img_no_wm, img_wm)
    if attack == "rotation_bilinear":
        return rotation_bilinear(img_no_wm, img_wm, angle)
    if attack == "orthogonal":
        return orthogonal_rotation(img_no_wm, img_wm, int(angle))
    if attack == "brightness":
        import torchvision.transforms as tforms
        return (
            tforms.ColorJitter(brightness=6)(img_no_wm),
            tforms.ColorJitter(brightness=6)(img_wm),
        )
    if attack == "contrast":
        return (
            ImageEnhance.Contrast(img_no_wm).enhance(0.5),
            ImageEnhance.Contrast(img_wm).enhance(0.5),
        )
    if attack == "jpeg":
        return _jpeg(img_no_wm, 25), _jpeg(img_wm, 25)
    if attack == "blur":
        import cv2
        return (
            Image.fromarray(cv2.GaussianBlur(np.asarray(img_no_wm), (5, 5), 1)),
            Image.fromarray(cv2.GaussianBlur(np.asarray(img_wm), (5, 5), 1)),
        )
    if attack == "noise":
        # Preserve the baseline's uint8 conversion semantics exactly.
        noise = (np.random.normal(0, 0.05, np.asarray(img_no_wm).shape) * 255).astype(np.uint8)
        return (
            Image.fromarray(np.clip(np.asarray(img_no_wm) + noise, 0, 255)),
            Image.fromarray(np.clip(np.asarray(img_wm) + noise, 0, 255)),
        )
    if attack == "bm3d":
        from bm3d import bm3d_rgb
        return (
            Image.fromarray(
                (np.clip(bm3d_rgb(np.asarray(img_no_wm) / 255, 0.1), 0, 1) * 255)
                .astype(np.uint8)
            ),
            Image.fromarray(
                (np.clip(bm3d_rgb(np.asarray(img_wm) / 255, 0.1), 0, 1) * 255)
                .astype(np.uint8)
            ),
        )
    if attack in {"vae_b", "vae_c"}:
        return (
            _compression_attack(img_no_wm, attack, device),
            _compression_attack(img_wm, attack, device),
        )
    if attack == "cc":
        import torchvision.transforms as tforms
        crop_len = int(512 * (0.5 ** 0.5))
        left = (512 - crop_len) // 2
        right = (512 - crop_len) - left
        transform = tforms.Compose([
            tforms.CenterCrop((crop_len, crop_len)),
            tforms.Pad((left, left, right, right), fill=0),
        ])
        return transform(img_no_wm), transform(img_wm)
    if attack == "rc":
        crop_len = int(512 * (0.7 ** 0.5))
        return (
            _random_crop_original_position(img_no_wm, crop_len),
            _random_crop_original_position(img_wm, crop_len),
        )
    raise ValueError(f"Unknown attack: {attack}")


def apply_rotation_attack(img_no_wm: Image.Image, img_wm: Image.Image,
                          attack: str, angle: float = 0.0):
    """Backward-compatible wrapper used by the baseline utility module."""

    return apply_attack_pair(img_no_wm, img_wm, attack, angle=angle)
