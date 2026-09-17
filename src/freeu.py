"""FreeU controls shared by generation and inversion."""

from research_config import FreeUConfig


def configure_freeu(pipe, config: FreeUConfig) -> None:
    """Configure FreeU on the U-Net actually owned by ``pipe``.

    Diffusers exposes these methods on the pipeline in current releases and on
    the U-Net in some older releases.  No post-processing approximation is used.
    """

    owner = pipe if hasattr(pipe, "enable_freeu") else getattr(pipe, "unet", None)
    if owner is None or not hasattr(owner, "enable_freeu"):
        raise RuntimeError("This diffusers pipeline does not expose FreeU controls")

    if config.enabled:
        owner.enable_freeu(s1=config.s1, s2=config.s2, b1=config.b1, b2=config.b2)
        return

    if hasattr(owner, "disable_freeu"):
        owner.disable_freeu()
    else:
        raise RuntimeError("This diffusers pipeline cannot disable FreeU")
