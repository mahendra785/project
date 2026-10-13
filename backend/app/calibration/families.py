r"""Nested prediction-set families for segmentation risk control.

Every family here is a *fixed* score transform :math:`s = T(\hat p)` with values
in :math:`[0, 1]`, and its prediction sets are upper level sets

.. math:: S_\lambda = \{\, u : s(u) \ge 1 - \lambda \,\}, \qquad \lambda \in [0, 1].

Because :math:`T` does not depend on the label or on :math:`\lambda`, the sets are
nested in :math:`\lambda` for every image, which is all that Lemma 1 of the theory
memo needs for any set-monotone loss (pixel FNR, lesion miss, small-lesion miss)
to be non-increasing in :math:`\lambda`. Validity of CRC therefore holds for
every family; families differ only in *efficiency* — how large the set must be to
reach a given risk.

Families
--------
``threshold``          :math:`s=\hat p`. Standard CRC (Angelopoulos et al., 2024).
``dilation``           :math:`s = 1/(1+d(u,\hat M))`, :math:`\hat M=\{\hat p\ge\frac12\}`:
                       the level sets are :math:`\hat M\oplus B_r` — the morphological
                       prediction sets of Mossina & Friedrich (MICCAI 2025).
``lesion_equalized``   the contribution of this work, defined below.

Lesion-equalized scores
-----------------------
Let :math:`C_1,\dots,C_J` be the connected components of the *candidate* mask
:math:`\{\hat p \ge t_0\}` and :math:`P_j=\max_{u\in C_j}\hat p(u)` their peaks. Then

.. math::
   s_{\rho,t_0}(u) = \begin{cases}
       \hat p(u)\,/\,P_j^{\rho} & u \in C_j \\
       \hat p(u)               & \text{otherwise}
   \end{cases}\qquad \rho\in[0,1].

Properties (proved in ``docs/research/theory-memo-v2.md``):

* :math:`s\in[0,1]`, and every candidate pixel outranks every non-candidate
  pixel (inside, :math:`s \ge \hat p \ge t_0`; outside, :math:`s=\hat p<t_0`).
* :math:`\rho=0` recovers thresholding exactly, so the family strictly contains
  the standard CRC baseline.
* **Amplitude invariance at** :math:`\rho=1`: if a lesion's posterior is a scaled
  profile :math:`\hat p = a_k\,g` on its candidate component, then
  :math:`s = g/\max g`, independent of the confidence amplitude :math:`a_k`. A
  faint lesion is then hit at the *same* cutoff as a confident lesion of the same
  shape, so the global cutoff no longer has to fall to the faintest lesion's
  level — the mechanism by which pixel thresholding inflates sets everywhere to
  protect small, low-confidence lesions.

The pair :math:`(\rho, t_0)` must be chosen on data **disjoint** from the
calibration split (see :func:`app.calibration.joint.select_family`); choosing it
on the calibration split would void the guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

#: 8-connectivity (2-D) / 26-connectivity (3-D) structuring elements.
def connectivity(ndim: int) -> np.ndarray:
    return np.ones((3,) * ndim, dtype=bool)


@dataclass(frozen=True, slots=True)
class ScoreFamily:
    """A fixed, label-free score transform defining nested sets {s >= 1 - lambda}."""

    kind: str
    rho: float = 0.0
    t0: float = 0.5

    @property
    def label(self) -> str:
        if self.kind == "lesion_equalized":
            return f"lesion_equalized(rho={self.rho:g},t0={self.t0:g})"
        return self.kind

    def score(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
        if self.kind == "threshold":
            return p
        if self.kind == "dilation":
            return dilation_score(p)
        if self.kind == "lesion_equalized":
            return lesion_equalized_score(p, rho=self.rho, t0=self.t0)
        raise ValueError(f"Unknown family '{self.kind}'")


def dilation_score(p: np.ndarray) -> np.ndarray:
    """1/(1+EDT to the argmax mask). Level sets are dilations of M_hat.

    If the argmax mask is empty the dilation family degenerates to the empty set
    for every radius (the 'feasibility floor' of theory memo §7). We seed it
    with the single highest-probability pixel instead so the family is never
    empty; this is disclosed and is the top-q seeding with q = 1 pixel.
    """
    seed = p >= 0.5
    if not seed.any():
        seed = p >= p.max()
    distance = ndimage.distance_transform_edt(~seed)
    return 1.0 / (1.0 + distance)


def lesion_equalized_score(p: np.ndarray, *, rho: float, t0: float) -> np.ndarray:
    """Peak-normalise each candidate component of {p >= t0} by P_j ** rho."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if not 0.0 < t0 <= 1.0:
        raise ValueError("t0 must lie in (0, 1]")
    s = p.copy()
    if rho == 0.0:
        return s
    candidate = p >= t0
    labels, count = ndimage.label(candidate, structure=connectivity(p.ndim))
    if count == 0:
        return s
    peaks = ndimage.maximum(p, labels, index=np.arange(1, count + 1))
    divisor = np.ones(count + 1, dtype=np.float64)
    divisor[1:] = np.power(np.maximum(peaks, t0), rho)
    inside = labels > 0
    s[inside] = np.minimum(p[inside] / divisor[labels[inside]], 1.0)
    return s


def default_family_grid() -> list[ScoreFamily]:
    """The tuning grid for lesion_equalized (rho = 0 is plain thresholding)."""
    grid = [ScoreFamily("threshold")]
    for t0 in (0.05, 0.1, 0.2, 0.3):
        for rho in (0.25, 0.5, 0.75, 1.0):
            grid.append(ScoreFamily("lesion_equalized", rho=rho, t0=t0))
    return grid
