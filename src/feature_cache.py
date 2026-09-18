"""Disk interface for inversion-independent HSQR feature reuse."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _canonical(value: Any):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class FeatureCacheKey:
    inversion_method: str
    inversion_hyperparameters: Mapping[str, Any]
    generation_freeu: Mapping[str, Any]
    inversion_freeu: Mapping[str, Any]
    generation_group: str
    model_id: str
    model_revision: Any
    diffusers_version: str
    torch_version: str
    torch_dtype: str
    num_inference_steps: int
    invert_guidance: float
    invert_prompt: Any
    git_commit: str
    source_image_sha256: str
    attack: str
    attack_parameters: Mapping[str, Any]
    dataset_id: str
    sample_id: str
    wm_type: str
    image_kind: str

    def metadata(self):
        return _canonical(asdict(self))

    def digest(self) -> str:
        payload = json.dumps(
            self.metadata(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class FeatureCache:
    def __init__(self, root):
        self.root = Path(root)

    def path_for(self, key: FeatureCacheKey) -> Path:
        digest = key.digest()
        return self.root / key.dataset_id / key.wm_type / digest[:2] / f"{digest}.npz"

    def contains(self, key: FeatureCacheKey) -> bool:
        return self.path_for(key).is_file()

    def load(self, key: FeatureCacheKey) -> torch.Tensor:
        path = self.path_for(key)
        with np.load(path, allow_pickle=False) as data:
            stored = json.loads(str(data["metadata"].item()))
            if stored != key.metadata():
                raise RuntimeError(f"Feature-cache metadata mismatch: {path}")
            feature = np.asarray(data["feature"], dtype=np.float32)
        return torch.from_numpy(feature.copy())

    def store(self, key: FeatureCacheKey, feature) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(feature, torch.Tensor):
            feature = feature.detach().cpu().numpy()
        feature = np.asarray(feature, dtype=np.float32)
        if feature.ndim != 1:
            raise ValueError("A cache record must contain one flattened feature")
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            feature=feature,
            metadata=np.array(json.dumps(key.metadata(), sort_keys=True)),
        )
        os.replace(temporary, path)
        return path
