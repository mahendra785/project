"""Concrete uncertainty estimators.

All of these are single-forward-pass, post-hoc estimators — no ensembling, no
MC dropout, no retraining — which is the "lightweight post-hoc" objective from
the project proposal. Each derives strictly from the model's own probabilities.
"""

from __future__ import annotations

import numpy as np

from app.schemas.common import ImplementationStatus
from app.uncertainty.base import BaseUncertaintyEstimator, UncertaintyMap


class PredictiveEntropyEstimator(BaseUncertaintyEstimator):
    r"""Normalised Shannon entropy of the per-pixel predictive distribution.

    .. math::
        H(x) = -\sum_c p_c(x)\,\log p_c(x), \qquad
        \tilde H(x) = \frac{H(x)}{\log C} \in [0, 1]

    Dividing by :math:`\log C` makes maps comparable across models with
    different class counts — essential for a model-agnostic framework, and the
    reason a 3-class and a 21-class adapter produce readable maps on the same
    colour scale.
    """

    id = "normalised_predictive_entropy"
    name = "Normalised Predictive Entropy"
    status = ImplementationStatus.IMPLEMENTED

    def estimate(self, probabilities: np.ndarray) -> UncertaintyMap:
        probs = self._validate(probabilities)
        n_classes = probs.shape[0]

        entropy = -(probs * np.log(probs)).sum(axis=0)
        normalised = np.clip(entropy / np.log(n_classes), 0.0, 1.0)

        return UncertaintyMap(
            values=normalised,
            method=self.id,
            method_name=self.name,
            status=self.status,
            confidence=probs.max(axis=0),
            notes=[
                "Aleatoric-dominated: single-pass entropy conflates data ambiguity "
                "with model ignorance and does not isolate epistemic uncertainty.",
                f"Normalised by log(C) with C={n_classes}.",
            ],
        )


class MarginConfidenceEstimator(BaseUncertaintyEstimator):
    """1 - (top-1 probability - top-2 probability).

    A decision-boundary-sensitive alternative to entropy: it saturates only
    where the two leading classes are genuinely tied, so it highlights
    boundaries more sharply and ignores diffuse low-probability mass.
    """

    id = "margin_uncertainty"
    name = "Top-2 Margin Uncertainty"
    status = ImplementationStatus.IMPLEMENTED

    def estimate(self, probabilities: np.ndarray) -> UncertaintyMap:
        probs = self._validate(probabilities)
        top2 = np.sort(probs, axis=0)[-2:]
        margin = top2[1] - top2[0]
        return UncertaintyMap(
            values=np.clip(1.0 - margin, 0.0, 1.0),
            method=self.id,
            method_name=self.name,
            status=self.status,
            confidence=probs.max(axis=0),
            notes=["Sensitive only to the top-2 classes; ignores tail mass."],
        )


class LeastConfidenceEstimator(BaseUncertaintyEstimator):
    """1 - max_c p_c, rescaled so a uniform distribution maps to exactly 1."""

    id = "least_confidence"
    name = "Least Confidence"
    status = ImplementationStatus.IMPLEMENTED

    def estimate(self, probabilities: np.ndarray) -> UncertaintyMap:
        probs = self._validate(probabilities)
        n_classes = probs.shape[0]
        maximum = probs.max(axis=0)
        scaled = (1.0 - maximum) * (n_classes / (n_classes - 1))
        return UncertaintyMap(
            values=np.clip(scaled, 0.0, 1.0),
            method=self.id,
            method_name=self.name,
            status=self.status,
            confidence=maximum,
            notes=["Rescaled so a uniform predictive distribution yields 1.0."],
        )


ESTIMATORS: dict[str, BaseUncertaintyEstimator] = {
    estimator.id: estimator
    for estimator in (
        PredictiveEntropyEstimator(),
        MarginConfidenceEstimator(),
        LeastConfidenceEstimator(),
    )
}

DEFAULT_ESTIMATOR_ID = PredictiveEntropyEstimator.id


def get_estimator(estimator_id: str | None = None) -> BaseUncertaintyEstimator:
    return ESTIMATORS[estimator_id or DEFAULT_ESTIMATOR_ID]
