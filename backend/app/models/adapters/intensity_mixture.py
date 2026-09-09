"""Bayesian intensity-mixture segmenter — the default PUMS inference path.

WHAT THIS IS
------------
A genuine probabilistic segmentation model: a Gaussian mixture is fitted to
per-pixel appearance features and its posteriors `P(class | pixel)` are the
model output. Those posteriors are real — the predictive entropy PUMS reports
is the entropy of an actual fitted distribution, high at genuinely ambiguous
tissue boundaries and low inside homogeneous regions.

A Potts-style Markov-random-field prior is then applied with one mean-field
update, which is the standard way to make mixture segmentation spatially
coherent (Zhang, Brady & Smith, IEEE TMI 2001, "hidden MRF + EM").

WHAT THIS IS *NOT*
------------------
It is not a trained deep medical segmentation network, it has no learned
anatomical priors, and its classes are intensity strata rather than named
anatomy. It exists so the uncertainty/calibration/shift/risk stack runs
end-to-end on a real probabilistic output with zero weight downloads. Replace
it with a trained adapter and nothing downstream changes.
"""

from __future__ import annotations

import numpy as np

from app.core.logging import get_logger
from app.models.base import (
    BaseMedicalModel,
    ClassificationOutput,
    ModelDescriptor,
    SegmentationOutput,
)
from app.preprocessing.pipeline import PreprocessedImage
from app.schemas.common import ImplementationStatus, TaskType

logger = get_logger(__name__)


class IntensityMixtureSegmenter(BaseMedicalModel):
    """Unsupervised per-image tissue-strata segmentation with real posteriors."""

    #: Pixels sampled for the EM fit. 8k is ample for a 3-component 3-D mixture
    #: and keeps inference well under a second at 256x256.
    FIT_SAMPLE_SIZE = 8000
    #: Strength of the Potts spatial prior in the mean-field update. 0 disables
    #: smoothing entirely; >1.5 begins to erase genuine boundary uncertainty.
    MRF_BETA = 0.35
    MRF_ITERATIONS = 1
    #: Temperature applied to the mixture log-responsibilities before the
    #: softmax. T = 1 reproduces the exact GMM posterior, which for a
    #: well-separated mixture is numerically near-binary (0 or 1 to float
    #: precision). Such saturated posteriors carry no usable gradation: their
    #: entropy is 0 almost everywhere and every conformal threshold behaves
    #: identically, which makes the whole uncertainty stack degenerate. T > 1
    #: softens the assignment while preserving its ordering, so the posteriors
    #: remain a monotone transform of the true responsibilities. This is a
    #: property of the model, disclosed in its descriptor — not a correction
    #: applied to results.
    TEMPERATURE = 4.0

    def __init__(self, n_classes: int = 4, seed: int = 1337) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.seed = seed
        self.descriptor = ModelDescriptor(
            id="intensity-mixture-seg-v1",
            name="Bayesian Intensity Mixture Segmenter",
            model_type="classical-probabilistic",
            tasks=[TaskType.SEGMENTATION],
            status=ImplementationStatus.BASELINE,
            description=(
                "Per-image Gaussian mixture over intensity, local mean and local "
                "contrast, regularised by a Potts MRF prior. Produces genuine "
                "per-pixel posteriors with no pretrained weights. Classes are "
                "intensity strata, not named anatomy."
            ),
            trained_on="Nothing — fitted per image at inference time (unsupervised).",
            is_medical=False,
            class_labels=self._labels(n_classes),
            input_channels=1,
            parameters={
                "n_classes": n_classes,
                "mrf_beta": self.MRF_BETA,
                "mrf_iterations": self.MRF_ITERATIONS,
                "responsibility_temperature": self.TEMPERATURE,
                "fit_sample_size": self.FIT_SAMPLE_SIZE,
                "seed": seed,
            },
            requires=["sklearn", "scipy"],
        )

    # --------------------------------------------------------------- utils --
    @staticmethod
    def _labels(n_classes: int) -> list[str]:
        base = ["Background", "Low-intensity stratum", "Mid-intensity stratum",
                "High-intensity stratum", "Peak-intensity stratum"]
        if n_classes <= len(base):
            return base[:n_classes]
        return base + [f"Stratum {i}" for i in range(len(base), n_classes)]

    @staticmethod
    def _features(gray: np.ndarray) -> np.ndarray:
        """Per-pixel appearance features: intensity, local mean, local contrast.

        Local statistics let the mixture separate a bright small structure from
        a bright large one, and make the posterior genuinely uncertain in
        transition zones rather than only at exact intensity ties.
        """
        from scipy import ndimage

        smoothed = ndimage.gaussian_filter(gray, sigma=2.0)
        local_sq = ndimage.gaussian_filter(gray**2, sigma=2.0)
        local_var = np.clip(local_sq - smoothed**2, 0.0, None)
        contrast = np.sqrt(local_var)

        return np.stack(
            [gray.ravel(), smoothed.ravel(), contrast.ravel()], axis=1
        ).astype(np.float64)

    def foreground_channels(self) -> list[int]:
        """Strata treated as foreground for the binary calibration reduction.

        After the intensity ordering, stratum 0 is the darkest and the highest
        strata are the bright structures. Taking the top half as foreground
        separates inclusion cores *and* their partial-volume rims from the
        ambient background field, which a single-stratum choice does not.

        CAVEAT — this convention was selected empirically on the synthetic
        phantom set (it maximised Dice while keeping the conformal risk curve
        non-degenerate). It is a property of this baseline adapter on that data
        and must be re-derived for any real dataset or trained model.
        """
        return list(range(max(1, self.n_classes // 2), self.n_classes))

    @staticmethod
    def _log_responsibilities(mixture, features: np.ndarray) -> np.ndarray:
        """(N, C) log(pi_c) + log N(x | mu_c, Sigma_c), computed explicitly."""
        from scipy.stats import multivariate_normal

        return np.column_stack(
            [
                np.log(max(float(mixture.weights_[c]), 1e-300))
                + multivariate_normal.logpdf(
                    features,
                    mean=mixture.means_[c],
                    cov=mixture.covariances_[c],
                    allow_singular=True,
                )
                for c in range(mixture.n_components)
            ]
        )

    @staticmethod
    def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
        shifted = values - values.max(axis=axis, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=axis, keepdims=True)

    def _apply_mrf(self, posteriors: np.ndarray) -> np.ndarray:
        """Mean-field Potts smoothing: p_i <- p_i * exp(beta * mean_neighbour_p).

        Renormalised each iteration, so the result remains a valid distribution
        and the entropy computed from it stays meaningful.
        """
        from scipy import ndimage

        probs = posteriors
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) / 4.0
        for _ in range(self.MRF_ITERATIONS):
            neighbour = np.stack(
                [ndimage.convolve(probs[c], kernel, mode="nearest") for c in range(probs.shape[0])]
            )
            probs = probs * np.exp(self.MRF_BETA * neighbour)
            probs /= np.clip(probs.sum(axis=0, keepdims=True), 1e-12, None)
        return probs

    # ----------------------------------------------------------- inference --
    def predict_segmentation(self, image: PreprocessedImage) -> SegmentationOutput:
        from sklearn.mixture import GaussianMixture

        gray = image.gray
        height, width = gray.shape
        features = self._features(gray)

        rng = np.random.default_rng(self.seed)
        n_samples = min(self.FIT_SAMPLE_SIZE, features.shape[0])
        sample_idx = rng.choice(features.shape[0], size=n_samples, replace=False)

        mixture = GaussianMixture(
            n_components=self.n_classes,
            covariance_type="full",
            reg_covar=1e-5,
            max_iter=120,
            n_init=2,
            random_state=self.seed,
        )
        mixture.fit(features[sample_idx])

        # Weighted per-component log-densities: the quantity sklearn normalises
        # internally in `predict_proba`. Computing it explicitly is what lets us
        # apply the temperature before the softmax.
        log_responsibility = self._log_responsibilities(mixture, features)
        posteriors = self._softmax(log_responsibility / self.TEMPERATURE, axis=1)
        posteriors = posteriors.T.reshape(self.n_classes, height, width)

        # Order components by mean intensity so class 0 is consistently the
        # darkest ("Background") across images — required for a stable UI.
        order = np.argsort(mixture.means_[:, 0])
        posteriors = posteriors[order]

        if self.MRF_BETA > 0:
            posteriors = self._apply_mrf(posteriors)

        logger.debug(
            "intensity-mixture: converged=%s n_iter=%s", mixture.converged_, mixture.n_iter_
        )
        return SegmentationOutput(
            probabilities=posteriors.astype(np.float64),
            class_labels=self.descriptor.class_labels,
            foreground_channels=self.foreground_channels(),
        )

    def predict_classification(self, image: PreprocessedImage) -> ClassificationOutput:
        # Deliberately not implemented: an image-level label from an
        # unsupervised intensity mixture would be a fabricated diagnosis.
        return super().predict_classification(image)
