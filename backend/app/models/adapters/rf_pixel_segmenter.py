"""Supervised random-forest pixel segmenter trained on real annotated data.

WHAT THIS IS
------------
A per-pixel classifier over a bank of multiscale filter responses — the
approach popularised by Ilastik (Berg et al., Nature Methods 2019) and still a
strong baseline for medical segmentation when training data is limited. Each
pixel is described by intensity, Gaussian smoothings, gradient magnitudes,
Laplacians and local standard deviation at several scales; a random forest maps
that vector to `P(foreground | pixel)`.

WHY IT IS THE RIGHT BASELINE FOR PUMS
-------------------------------------
1. It is genuinely *supervised*, so it learns real anatomy from real expert
   masks. The unsupervised intensity mixture cannot — a hippocampus is not
   separable by intensity alone — so it produced honest but useless Dice on
   real MRI.
2. It trains in seconds on CPU with no GPU, no PyTorch and no pretrained
   weights, so the whole project stays reproducible on a laptop.
3. **Random-forest vote fractions are known to be poorly calibrated.** That is
   not a defect here, it is the point: it gives PUMS a real, measurable
   miscalibration to correct, on real data, rather than a contrived one.

WHAT IT IS NOT
--------------
Not a deep network and not a clinically validated model. It has no notion of
anatomy beyond local texture, so it makes spatially incoherent errors that a
U-Net would not. It is a defensible research baseline, not a deployable tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.errors import ModelUnavailableError
from app.core.logging import get_logger
from app.models.base import (
    BaseMedicalModel,
    ModelDescriptor,
    SegmentationOutput,
)
from app.preprocessing.pipeline import PreprocessedImage
from app.schemas.common import ImplementationStatus, TaskType

logger = get_logger(__name__)

#: Scales (in pixels, at the 256x256 working resolution) for the filter bank.
#: Spans fine texture to organ-scale context; more scales cost inference time
#: for diminishing returns.
SCALES: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

FEATURE_NAMES: list[str] = (
    ["intensity"]
    + [f"gaussian_s{s:g}" for s in SCALES]
    + [f"gradient_magnitude_s{s:g}" for s in SCALES]
    + [f"laplacian_s{s:g}" for s in SCALES]
    + [f"local_std_s{s:g}" for s in (2.0, 4.0)]
)


def compute_features(image: np.ndarray) -> np.ndarray:
    """(H*W, F) feature matrix. Must be identical at train and inference time."""
    from scipy import ndimage

    features: list[np.ndarray] = [image]

    smoothed: dict[float, np.ndarray] = {}
    for sigma in SCALES:
        smoothed[sigma] = ndimage.gaussian_filter(image, sigma=sigma)
        features.append(smoothed[sigma])

    for sigma in SCALES:
        gy = ndimage.gaussian_filter(image, sigma=sigma, order=(1, 0))
        gx = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 1))
        features.append(np.sqrt(gy**2 + gx**2))

    for sigma in SCALES:
        features.append(ndimage.gaussian_laplace(image, sigma=sigma))

    for sigma in (2.0, 4.0):
        mean = smoothed.get(sigma, ndimage.gaussian_filter(image, sigma=sigma))
        mean_sq = ndimage.gaussian_filter(image**2, sigma=sigma)
        features.append(np.sqrt(np.clip(mean_sq - mean**2, 0.0, None)))

    return np.stack([f.ravel() for f in features], axis=1).astype(np.float32)


@dataclass(slots=True)
class TrainingReport:
    """What actually happened during training — reported, never assumed."""

    dataset_id: str
    n_train_slices: int
    n_train_cases: int
    n_pixels_sampled: int
    foreground_fraction: float
    n_features: int
    n_estimators: int
    max_depth: int
    train_seconds: float
    oob_accuracy: float | None


class RandomForestPixelSegmenter(BaseMedicalModel):
    """Trained pixel classifier. Weights are produced by `scripts/train_pixel_model.py`."""

    #: Pixels sampled across the training split. 300k is ample for a forest of
    #: this depth and keeps training under a minute on a laptop CPU.
    MAX_TRAIN_PIXELS = 300_000
    N_ESTIMATORS = 120
    MAX_DEPTH = 16
    #: Foreground is a small fraction of every slice. Without rebalancing, the
    #: forest maximises accuracy by predicting background everywhere.
    CLASS_WEIGHT = "balanced_subsample"

    def __init__(self, model_path: Path | None = None, seed: int = 1337) -> None:
        super().__init__()
        from app.core.config import BACKEND_ROOT

        self.seed = seed
        self.model_path = model_path or (BACKEND_ROOT / "var" / "models" / "rf_pixel_seg.joblib")
        self._forest: Any = None
        self._report: TrainingReport | None = None
        #: Small JSON sidecar so `metadata()` can describe the trained model
        #: without deserialising the whole forest on every /api/models call.
        self.report_path = self.model_path.with_suffix(".report.json")

        self.descriptor = ModelDescriptor(
            id="rf-pixel-seg-v1",
            name="Random-Forest Pixel Segmenter (trained)",
            model_type="classical-supervised",
            tasks=[TaskType.SEGMENTATION],
            status=ImplementationStatus.BASELINE,
            description=(
                "Random forest over a multiscale filter bank (intensity, Gaussian, "
                "gradient magnitude, Laplacian, local std at 4 scales), trained on "
                "real expert annotations. Ilastik-style. Supervised, CPU-only, no "
                "pretrained weights."
            ),
            trained_on="Not trained yet — run scripts/train_pixel_model.py",
            is_medical=False,
            class_labels=["Background", "Foreground"],
            input_channels=1,
            parameters={
                "n_estimators": self.N_ESTIMATORS,
                "max_depth": self.MAX_DEPTH,
                "scales": list(SCALES),
                "n_features": len(FEATURE_NAMES),
                "class_weight": self.CLASS_WEIGHT,
                "seed": seed,
            },
            requires=["sklearn", "scipy", "joblib"],
        )

    # ------------------------------------------------------------ lifecycle --
    @property
    def is_trained(self) -> bool:
        return self.model_path.exists()

    def availability(self) -> tuple[bool, str | None]:
        available, reason = super().availability()
        if not available:
            return available, reason
        if not self.is_trained:
            return False, (
                "Not trained. Run: python scripts/train_pixel_model.py "
                "--dataset msd-hippocampus"
            )
        return True, None

    def _load(self) -> None:
        if not self.is_trained:
            raise ModelUnavailableError(
                "rf-pixel-seg-v1 has no trained weights. Run "
                "`python scripts/train_pixel_model.py --dataset msd-hippocampus`."
            )
        import joblib

        bundle = joblib.load(self.model_path)
        self._forest = bundle["forest"]
        self._report = bundle.get("report")
        self._describe_training()
        logger.info("Loaded trained forest from %s", self.model_path)

    def _describe_training(self, summary: dict | None = None) -> None:
        """Make the descriptor state what this model was ACTUALLY trained on.

        A stale `trained_on` string on a trained model would misrepresent the
        result, so the descriptor is rewritten from the training report rather
        than left at its constructor default.
        """
        if summary is None:
            if self._report is None:
                return
            summary = {
                "dataset_id": self._report.dataset_id,
                "n_train_cases": self._report.n_train_cases,
                "n_train_slices": self._report.n_train_slices,
                "n_pixels_sampled": self._report.n_pixels_sampled,
                "oob_accuracy": self._report.oob_accuracy,
                "train_seconds": self._report.train_seconds,
            }

        from app.datasets.registry import DATASETS

        spec = DATASETS.get(summary["dataset_id"])
        source = spec.name if spec else summary["dataset_id"]
        self.descriptor.trained_on = (
            f"{source} — {summary['n_train_cases']} patient-disjoint cases / "
            f"{summary['n_train_slices']} slices, "
            f"{summary['n_pixels_sampled']:,} sampled pixels, expert annotations"
        )
        # Trained on real medical imaging data — which is NOT the same as being
        # validated for clinical use. `clinically_validated` stays False.
        self.descriptor.is_medical = True
        self.descriptor.parameters = {
            **self.descriptor.parameters,
            "trained_dataset": summary["dataset_id"],
            "train_cases": summary["n_train_cases"],
            "train_slices": summary["n_train_slices"],
            "oob_accuracy": round(float(summary["oob_accuracy"]), 4),
            "train_seconds": round(float(summary["train_seconds"]), 1),
        }

    def metadata(self) -> ModelDescriptor:
        """Describe the adapter, reading the training sidecar if present."""
        if self._report is None and self.report_path.exists():
            import json

            try:
                self._describe_training(json.loads(self.report_path.read_text("utf-8")))
            except Exception as exc:
                logger.warning("Could not read training report: %s", exc)
        return self.descriptor

    # -------------------------------------------------------------- training --
    def train(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        *,
        dataset_id: str,
        n_cases: int,
    ) -> TrainingReport:
        """Fit the forest on (image, mask) pairs and persist it."""
        import time

        import joblib
        from sklearn.ensemble import RandomForestClassifier

        if not images:
            raise ValueError("No training slices supplied")

        rng = np.random.default_rng(self.seed)
        feature_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []

        per_slice = max(1, self.MAX_TRAIN_PIXELS // len(images))
        for image, mask in zip(images, masks):
            features = compute_features(image)
            labels = mask.ravel().astype(np.int8)

            # Sample foreground and background separately: foreground is ~5-15%
            # of a slice, so uniform sampling would starve the positive class.
            positive = np.flatnonzero(labels == 1)
            negative = np.flatnonzero(labels == 0)
            n_pos = min(len(positive), per_slice // 2)
            n_neg = min(len(negative), per_slice - n_pos)
            if n_pos == 0 or n_neg == 0:
                continue

            idx = np.concatenate(
                [
                    rng.choice(positive, size=n_pos, replace=False),
                    rng.choice(negative, size=n_neg, replace=False),
                ]
            )
            feature_blocks.append(features[idx])
            label_blocks.append(labels[idx])

        if not feature_blocks:
            raise ValueError("No usable training pixels (all slices were empty)")

        X = np.concatenate(feature_blocks)
        y = np.concatenate(label_blocks)
        logger.info("Training on %s pixels, %d features", f"{len(y):,}", X.shape[1])

        started = time.perf_counter()
        forest = RandomForestClassifier(
            n_estimators=self.N_ESTIMATORS,
            max_depth=self.MAX_DEPTH,
            min_samples_leaf=4,
            class_weight=self.CLASS_WEIGHT,
            n_jobs=-1,
            random_state=self.seed,
            oob_score=True,
            bootstrap=True,
        )
        forest.fit(X, y)
        elapsed = time.perf_counter() - started

        report = TrainingReport(
            dataset_id=dataset_id,
            n_train_slices=len(images),
            n_train_cases=n_cases,
            n_pixels_sampled=int(len(y)),
            foreground_fraction=float((y == 1).mean()),
            n_features=int(X.shape[1]),
            n_estimators=self.N_ESTIMATORS,
            max_depth=self.MAX_DEPTH,
            train_seconds=elapsed,
            oob_accuracy=float(getattr(forest, "oob_score_", float("nan"))),
        )

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        self.report_path.write_text(
            json.dumps(
                {
                    "dataset_id": report.dataset_id,
                    "n_train_cases": report.n_train_cases,
                    "n_train_slices": report.n_train_slices,
                    "n_pixels_sampled": report.n_pixels_sampled,
                    "foreground_fraction": report.foreground_fraction,
                    "oob_accuracy": report.oob_accuracy,
                    "train_seconds": report.train_seconds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        joblib.dump(
            {"forest": forest, "report": report, "feature_names": FEATURE_NAMES},
            self.model_path,
            compress=3,
        )
        self._forest = forest
        self._report = report
        self._loaded = True
        self._describe_training()
        logger.info("Trained in %.1fs, OOB accuracy %.4f", elapsed, report.oob_accuracy)
        return report

    def feature_importances(self) -> list[tuple[str, float]]:
        """Which filter responses the forest actually relies on."""
        self.load()
        if self._forest is None:
            return []
        pairs = list(zip(FEATURE_NAMES, self._forest.feature_importances_.tolist()))
        return sorted(pairs, key=lambda item: item[1], reverse=True)

    # ------------------------------------------------------------- inference --
    def predict_segmentation(self, image: PreprocessedImage) -> SegmentationOutput:
        if self._forest is None:
            self._load()

        height, width = image.gray.shape
        features = compute_features(image.gray)
        probabilities = self._forest.predict_proba(features)  # (N, 2)

        # sklearn orders columns by `classes_`; do not assume [0, 1].
        classes = list(self._forest.classes_)
        background = probabilities[:, classes.index(0)] if 0 in classes else 1 - probabilities[:, 0]
        foreground = probabilities[:, classes.index(1)] if 1 in classes else probabilities[:, 0]

        stacked = np.stack(
            [background.reshape(height, width), foreground.reshape(height, width)]
        ).astype(np.float64)
        stacked = np.clip(stacked, 1e-9, 1.0)
        stacked /= stacked.sum(axis=0, keepdims=True)

        return SegmentationOutput(
            probabilities=stacked,
            class_labels=self.descriptor.class_labels,
            foreground_channels=[1],
        )
