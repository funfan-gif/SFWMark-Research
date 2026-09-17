"""Central configuration for the SFWMark HSQR research extensions."""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class FreeUConfig:
    enabled: bool = False
    s1: float = 0.9
    s2: float = 0.2
    b1: float = 1.1
    b2: float = 1.2


@dataclass(frozen=True)
class ExperimentConfig:
    inversion_method: str
    generation_freeu: FreeUConfig
    inversion_freeu: FreeUConfig


FREEU_DEFAULTS = FreeUConfig(enabled=False)
ROTATION_BILINEAR_ANGLES: Tuple[int, ...] = (
    -75, -45, -30, -15, -10, -5, 0, 5, 10, 15, 30, 45, 75
)


def _freeu(enabled: bool) -> FreeUConfig:
    return FreeUConfig(enabled=enabled)


EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "Q0": ExperimentConfig("ddim", _freeu(False), _freeu(False)),
    "Q1": ExperimentConfig("exact_ddim", _freeu(False), _freeu(False)),
    "Q2": ExperimentConfig("gnri", _freeu(False), _freeu(False)),
    "Q3": ExperimentConfig("ddim", _freeu(True), _freeu(True)),
    "Q4": ExperimentConfig("exact_ddim", _freeu(True), _freeu(True)),
    "Q5": ExperimentConfig("gnri", _freeu(True), _freeu(True)),
}
