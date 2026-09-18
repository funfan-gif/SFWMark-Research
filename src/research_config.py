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
    generation_group: str
    display_name: str


FREEU_DEFAULTS = FreeUConfig(enabled=False)
ROTATION_BILINEAR_ANGLES: Tuple[int, ...] = (
    -75, -45, -30, -15, -10, -5, 0, 5, 10, 15, 30, 45, 75
)


def _freeu(enabled: bool) -> FreeUConfig:
    return FreeUConfig(enabled=enabled)


EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "Q0": ExperimentConfig("ddim", _freeu(False), _freeu(False), "G0", "DDIM baseline"),
    "Q1": ExperimentConfig(
        "exact_ddim", _freeu(False), _freeu(False), "G0",
        "Exact-DPM-inspired first-order forward-step DDIM refinement",
    ),
    "Q2": ExperimentConfig(
        "gnri", _freeu(False), _freeu(False), "G0",
        "GNRI official-semantics SD2.1/DDIM adaptation",
    ),
    "Q3": ExperimentConfig("ddim", _freeu(True), _freeu(True), "G1", "DDIM + FreeU"),
    "Q4": ExperimentConfig(
        "exact_ddim", _freeu(True), _freeu(True), "G1",
        "Exact-DPM-inspired first-order forward-step DDIM refinement + FreeU",
    ),
    "Q5": ExperimentConfig(
        "gnri", _freeu(True), _freeu(True), "G1",
        "GNRI official-semantics SD2.1/DDIM adaptation + FreeU",
    ),
    # Diagnostic-only phase ablations.  They are intentionally excluded from
    # the formal Q0-Q5 table.
    "D0": ExperimentConfig(
        "ddim", _freeu(False), _freeu(True), "G0",
        "Diagnostic: generation FreeU OFF / inversion FreeU ON",
    ),
    "D1": ExperimentConfig(
        "ddim", _freeu(True), _freeu(False), "G1",
        "Diagnostic: generation FreeU ON / inversion FreeU OFF",
    ),
}
