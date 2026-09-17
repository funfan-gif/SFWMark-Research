"""Rotation attacks used by the SFWMark research experiments."""

from typing import Tuple

from PIL import Image


CANVAS_SIZE = (512, 512)


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
        angle=float(angle),
        resample=resampling[interpolation],
        expand=False,
        fillcolor=0,
    )


def rotate_pair(img_no_wm: Image.Image, img_wm: Image.Image, angle: float,
                interpolation: str = "bilinear", orthogonal_exact: bool = False
                ) -> Tuple[Image.Image, Image.Image]:
    """Apply exactly the same deterministic rotation settings to both images."""

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


def apply_rotation_attack(img_no_wm: Image.Image, img_wm: Image.Image,
                          attack: str, angle: float = 0.0):
    if attack == "clean":
        return _fixed_black_canvas(img_no_wm), _fixed_black_canvas(img_wm)
    if attack == "rot75_nn":
        return rot75_nn(img_no_wm, img_wm)
    if attack == "rotation_bilinear":
        return rotation_bilinear(img_no_wm, img_wm, angle)
    if attack == "orthogonal":
        return orthogonal_rotation(img_no_wm, img_wm, int(angle))
    raise ValueError(f"Unknown rotation attack: {attack}")
