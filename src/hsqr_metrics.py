"""HSQR feature extraction and shared-covariance distance evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf


HSQR_QR_SIZE = 42
HSQR_HALF_WIDTH = 21
HSQR_COMPLEX_DIM = HSQR_QR_SIZE * HSQR_HALF_WIDTH
HSQR_FEATURE_DIM = 2 * HSQR_COMPLEX_DIM
HSQR_CHANNEL = 3
CENTER_START = 10
CENTER_END = 54


def extract_hsqr_complex(recovered_z_t: torch.Tensor, center: bool = True,
                         channel: int = HSQR_CHANNEL) -> torch.Tensor:
    """Extract the 42x21 complex HSQR region from a recovered latent batch."""

    if recovered_z_t.ndim == 3:
        recovered_z_t = recovered_z_t.unsqueeze(0)
    if recovered_z_t.ndim != 4:
        raise ValueError("recovered_z_t must have shape (B,C,H,W) or (C,H,W)")

    if center:
        spatial = recovered_z_t[..., CENTER_START:CENTER_END, CENTER_START:CENTER_END]
        spectrum = torch.fft.fftshift(torch.fft.fft2(spatial), dim=(-1, -2))
        region = spectrum[:, channel, 1:1 + HSQR_QR_SIZE, 23:23 + HSQR_HALF_WIDTH]
    else:
        spectrum = torch.fft.fftshift(torch.fft.fft2(recovered_z_t), dim=(-1, -2))
        center_row = spectrum.shape[-2] // 2
        region = spectrum[
            :, channel,
            center_row - HSQR_HALF_WIDTH:center_row + HSQR_HALF_WIDTH,
            center_row + 1:center_row + 1 + HSQR_HALF_WIDTH,
        ]
    if region.shape[-2:] != (HSQR_QR_SIZE, HSQR_HALF_WIDTH):
        raise ValueError(f"Unexpected HSQR region shape: {tuple(region.shape)}")
    return region


def hsqr_complex_to_feature(region: torch.Tensor) -> torch.Tensor:
    """Map 42x21 complex values to [real.flatten(), imag.flatten()] (1764-D)."""

    if region.ndim == 2:
        region = region.unsqueeze(0)
    if region.shape[-2:] != (HSQR_QR_SIZE, HSQR_HALF_WIDTH):
        raise ValueError(f"Unexpected HSQR region shape: {tuple(region.shape)}")
    feature = torch.cat(
        [region.real.reshape(region.shape[0], -1), region.imag.reshape(region.shape[0], -1)],
        dim=1,
    ).float()
    if feature.shape[1] != HSQR_FEATURE_DIM:
        raise AssertionError(f"HSQR feature dimension must be {HSQR_FEATURE_DIM}")
    return feature


def extract_hsqr_feature(recovered_z_t: torch.Tensor, center: bool = True,
                         channel: int = HSQR_CHANNEL) -> torch.Tensor:
    return hsqr_complex_to_feature(extract_hsqr_complex(recovered_z_t, center, channel))


def hsqr_reference_complex(qr_bool: torch.Tensor) -> torch.Tensor:
    """Build the exact complex reference used by baseline get_distance_hsqr()."""

    if qr_bool.ndim == 2:
        qr_bool = qr_bool.unsqueeze(0).unsqueeze(0)
    elif qr_bool.ndim == 3:
        qr_bool = qr_bool.unsqueeze(0)
    if qr_bool.ndim != 4 or qr_bool.shape[-2:] != (HSQR_QR_SIZE, HSQR_QR_SIZE):
        raise ValueError("qr_bool must end in shape (42,42)")
    # The current SFWMark paper configuration has one HSQR watermark channel.
    if qr_bool.shape[1] != 1:
        raise ValueError("The 1764-D HSQR feature requires exactly one watermark channel")
    values = torch.where(
        qr_bool[:, 0].bool(),
        torch.tensor(45.0, device=qr_bool.device),
        torch.tensor(-45.0, device=qr_bool.device),
    )
    return torch.complex(values[..., :HSQR_HALF_WIDTH], values[..., HSQR_HALF_WIDTH:])


def hsqr_reference_feature(qr_bool: torch.Tensor) -> torch.Tensor:
    return hsqr_complex_to_feature(hsqr_reference_complex(qr_bool))


def _as_2d_float64(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[1] != HSQR_FEATURE_DIM:
        raise ValueError(f"Expected shape (N,{HSQR_FEATURE_DIM}), got {value.shape}")
    return value


def complex_l1_from_features(query, reference) -> np.ndarray:
    """Original HSQR complex L1: mean(abs(complex_query-complex_reference))."""

    query = _as_2d_float64(query)
    reference = _as_2d_float64(reference)
    residual = query[:, None, :] - reference[None, :, :]
    real = residual[..., :HSQR_COMPLEX_DIM]
    imag = residual[..., HSQR_COMPLEX_DIM:]
    return np.sqrt(real * real + imag * imag).mean(axis=-1)


def l2_from_features(query, reference) -> np.ndarray:
    query = _as_2d_float64(query)
    reference = _as_2d_float64(reference)
    residual = query[:, None, :] - reference[None, :, :]
    return np.linalg.norm(residual, axis=-1)


@dataclass
class HSQRDistanceModel:
    """Diagonal and full-whitening models fitted on correct-key residuals."""

    diagonal_variance: Optional[np.ndarray] = None
    cholesky: Optional[np.ndarray] = None
    shrinkage: Optional[float] = None
    jitter: float = 0.0

    def fit(self, query_features, correct_reference_features) -> "HSQRDistanceModel":
        queries = _as_2d_float64(query_features)
        references = _as_2d_float64(correct_reference_features)
        if queries.shape != references.shape:
            raise ValueError("Fitting queries and correct references must have identical shapes")
        if queries.shape[0] < 2:
            raise ValueError("At least two fitting residuals are required")
        residuals = queries - references

        estimator = LedoitWolf(assume_centered=False).fit(residuals)
        covariance = np.asarray(estimator.covariance_, dtype=np.float64)
        diagonal = np.diag(covariance).copy()
        positive = diagonal[diagonal > 0]
        floor = max(float(np.median(positive)) * 1e-8 if positive.size else 0.0, 1e-12)
        self.diagonal_variance = np.maximum(diagonal, floor)
        identity = np.eye(covariance.shape[0], dtype=np.float64)
        jitter = 0.0
        for candidate_jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
            try:
                self.cholesky = np.linalg.cholesky(
                    covariance + candidate_jitter * identity
                )
                jitter = candidate_jitter
                break
            except np.linalg.LinAlgError:
                continue
        if self.cholesky is None:
            raise np.linalg.LinAlgError("Ledoit-Wolf covariance is not Cholesky-factorizable")
        self.shrinkage = float(estimator.shrinkage_)
        self.jitter = jitter
        return self

    def _require_fit(self):
        if self.diagonal_variance is None or self.cholesky is None:
            raise RuntimeError("Call fit() or load() before covariance distances")

    def diagonal_distance(self, query, references, chunk_size: int = 128) -> np.ndarray:
        self._require_fit()
        return self._chunked_distance(query, references, "diag", chunk_size)

    def mahalanobis_distance(self, query, references, chunk_size: int = 128) -> np.ndarray:
        self._require_fit()
        return self._chunked_distance(query, references, "mahalanobis", chunk_size)

    def _chunked_distance(self, query, references, metric: str, chunk_size: int) -> np.ndarray:
        queries = _as_2d_float64(query)
        references = _as_2d_float64(references)
        chunks = []
        for start in range(0, references.shape[0], chunk_size):
            ref = references[start:start + chunk_size]
            residual = queries[:, None, :] - ref[None, :, :]
            if metric == "diag":
                distance = np.sqrt(
                    np.sum(residual * residual / self.diagonal_variance, axis=-1)
                )
            else:
                flat = residual.reshape(-1, residual.shape[-1]).T
                whitened = solve_triangular(
                    self.cholesky, flat, lower=True, check_finite=False
                )
                distance = np.linalg.norm(whitened, axis=0).reshape(
                    queries.shape[0], ref.shape[0]
                )
            chunks.append(distance)
        return np.concatenate(chunks, axis=1)

    def distances(self, query, references,
                  metrics: Sequence[str] = ("complex_l1", "l2", "diag", "mahalanobis"),
                  chunk_size: int = 128) -> Dict[str, np.ndarray]:
        result = {}
        for metric in metrics:
            if metric == "complex_l1":
                result[metric] = complex_l1_from_features(query, references)
            elif metric == "l2":
                result[metric] = l2_from_features(query, references)
            elif metric == "diag":
                result[metric] = self.diagonal_distance(query, references, chunk_size)
            elif metric == "mahalanobis":
                result[metric] = self.mahalanobis_distance(query, references, chunk_size)
            else:
                raise ValueError(f"Unknown HSQR distance: {metric}")
        return result

    def identify(self, query, candidate_references, metric: str,
                 chunk_size: int = 128) -> np.ndarray:
        """Argmin over every candidate key; no ground-truth key is accepted here."""

        return np.argmin(
            self.distances(query, candidate_references, (metric,), chunk_size)[metric],
            axis=1,
        )

    def save(self, path) -> None:
        self._require_fit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            diagonal_variance=self.diagonal_variance,
            cholesky=self.cholesky,
            shrinkage=np.array(self.shrinkage),
            jitter=np.array(self.jitter),
        )

    @classmethod
    def load(cls, path) -> "HSQRDistanceModel":
        with np.load(path) as data:
            return cls(
                diagonal_variance=data["diagonal_variance"].copy(),
                cholesky=data["cholesky"].copy(),
                shrinkage=float(data["shrinkage"]),
                jitter=float(data["jitter"]),
            )
