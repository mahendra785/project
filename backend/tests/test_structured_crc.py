"""Tests for multi-granularity CRC: families, structured losses, joint selection."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from app.calibration.families import ScoreFamily, default_family_grid, lesion_equalized_score
from app.calibration.joint import (
    crc_index,
    evaluate,
    ltt_index,
    max_index,
    select_family,
    union_index,
)
from app.calibration.losses import gt_components, lambda_grid, stack, unit_stats


def _blob_image(rng, size=64, k=3):
    """Random image with k Gaussian blobs of random size and amplitude."""
    yy, xx = np.mgrid[:size, :size]
    p = rng.uniform(0, 0.05, (size, size))
    mask = np.zeros((size, size), bool)
    for _ in range(k):
        cy, cx = rng.integers(8, size - 8, 2)
        r = rng.uniform(2, 8)
        a = rng.uniform(0.2, 1.0)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        p = np.maximum(p, a * np.exp(-d2 / (2 * r**2)))
        mask |= d2 <= (r * rng.uniform(0.8, 1.4)) ** 2
    return np.clip(p, 0, 1), mask


FAMILIES = [ScoreFamily("threshold"), ScoreFamily("dilation"),
            ScoreFamily("lesion_equalized", rho=0.5, t0=0.1),
            ScoreFamily("lesion_equalized", rho=1.0, t0=0.2)]


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.label)
def test_losses_are_monotone_for_every_family(family):
    rng = np.random.default_rng(0)
    units = []
    for _ in range(10):
        p, mask = _blob_image(rng)
        units.append(unit_stats(family.score(p), gt_components(mask)))
    st = stack(units, small_max_size=60)
    for m in (st.pix_rate, st.les_rate, np.nan_to_num(st.small_rate)):
        assert np.all(np.diff(m, axis=1) <= 1e-7)
    assert np.all(np.diff(st.set_size, axis=1) >= 0)
    # lambda = 1 -> cutoff 0 -> full image -> all losses zero
    assert np.allclose(st.pix_rate[:, -1], 0) and np.allclose(st.les_rate[:, -1], 0)


def test_scores_in_unit_interval_and_candidates_outrank_background():
    rng = np.random.default_rng(1)
    p, _ = _blob_image(rng)
    s = lesion_equalized_score(p, rho=0.7, t0=0.15)
    assert s.min() >= 0 and s.max() <= 1
    cand = p >= 0.15
    assert s[cand].min() >= s[~cand].max()


def test_rho_zero_is_thresholding():
    rng = np.random.default_rng(2)
    p, _ = _blob_image(rng)
    assert np.array_equal(lesion_equalized_score(p, rho=0.0, t0=0.1), p)


def test_amplitude_invariance_at_rho_one():
    """A scaled profile a*g on its candidate component maps to g/max g for any a."""
    yy, xx = np.mgrid[:48, :48]
    g = np.exp(-((yy - 24) ** 2 + (xx - 24) ** 2) / (2 * 5.0**2))
    t0 = 0.05
    # Inside the core shared by every amplitude, s = g / max g regardless of a.
    core = 0.3 * g >= t0
    for a in (0.3, 0.6, 0.95):
        s = lesion_equalized_score(a * g, rho=1.0, t0=t0)
        assert np.allclose(s[core], (g / g.max())[core])


def test_lemma2_order_statistic_matches_brute_force():
    rng = np.random.default_rng(3)
    lambdas = lambda_grid(41)
    for _ in range(5):
        p, mask = _blob_image(rng, k=4)
        labels = gt_components(mask)
        u = unit_stats(p, labels, lambdas=lambdas)
        st = stack([u], small_max_size=1e9, lambdas=lambdas)
        k = labels.max()
        for g, lam in enumerate(lambdas):
            in_set = p >= 1 - lam
            missed = sum(
                (in_set & (labels == c)).sum() < 0.5 * (labels == c).sum() for c in range(1, k + 1)
            )
            assert st.les_rate[0, g] == pytest.approx(missed / k)
            assert st.pix_rate[0, g] == pytest.approx(1 - (in_set & mask).sum() / mask.sum(), abs=1e-6)


def test_gt_components_min_size_filter():
    m = np.zeros((20, 20), bool)
    m[1:3, 1:3] = True  # 4 px
    m[10:16, 10:16] = True  # 36 px
    assert gt_components(m, 0).max() == 2
    assert gt_components(m, 10).max() == 1
    # diagonal neighbours are one component under 8-connectivity
    d = np.zeros((5, 5), bool)
    d[1, 1] = d[2, 2] = True
    assert ndimage.label(d)[1] == 2 and gt_components(d).max() == 1


def _random_stats(rng, n):
    units = []
    for _ in range(n):
        p, mask = _blob_image(rng, size=40, k=int(rng.integers(1, 4)))
        units.append(unit_stats(p, gt_components(mask), lambdas=lambda_grid(201)))
    return stack(units, small_max_size=40, lambdas=lambda_grid(201))


def test_union_dominates_each_and_max_dominates_union():
    st = _random_stats(np.random.default_rng(4), 40)
    for alpha in (0.05, 0.1, 0.2):
        iu, _ = union_index(st, alpha, risks=("pix", "les"))
        im, _ = max_index(st, alpha, risks=("pix", "les"))
        for r_curve in (st.pix_rate, st.les_rate):
            assert iu >= crc_index(r_curve, alpha)[0]
        assert im >= iu


def test_crc_union_validity_by_simulation():
    """Exchangeable units: mean test risk over random splits must be <= alpha (+MC slack)."""
    rng = np.random.default_rng(5)
    st = _random_stats(rng, 160)
    lambdas = lambda_grid(201)
    alpha = 0.15
    pix, les = [], []
    for _ in range(300):
        perm = rng.permutation(len(st))
        cal, test = st.subset(perm[:80]), st.subset(perm[80:])
        idx, _ = union_index(cal, alpha, risks=("pix", "les"))
        ev = evaluate(test, idx, lambdas)
        pix.append(ev.risk_pix)
        les.append(ev.risk_les)
    assert np.mean(pix) <= alpha + 0.01
    assert np.mean(les) <= alpha + 0.01


def test_ltt_is_at_least_as_conservative_as_crc_at_moderate_n():
    st = _random_stats(np.random.default_rng(6), 60)
    for alpha in (0.1, 0.2):
        assert ltt_index(st, alpha, delta=0.1)[0] >= union_index(st, alpha)[0]


def test_empty_or_small_subgroup_falls_back_to_whole_unit():
    """Eq. (crc): with n_r + 1 < B / alpha, including n_r = 0, every rule returns lambda = 1."""
    st = _random_stats(np.random.default_rng(7), 30)
    last = st.pix_rate.shape[1] - 1
    no_small = st.subset(np.flatnonzero(~st.has_small))
    assert no_small.has_small.sum() == 0
    assert crc_index(no_small.small_rate[no_small.has_small], 0.1) == (last, False)
    assert union_index(no_small, 0.1) == (last, False)
    assert ltt_index(no_small, 0.1) == (last, False)
    # 18 eligible units: (0 + 1) / 19 > 0.05, so no grid point is feasible at alpha = 0.05 (needs 19)
    assert crc_index(st.pix_rate[:18], 0.05) == (last, False)
    assert crc_index(st.pix_rate[:19], 0.05)[1]  # (0 + 1) / 20 = alpha: lambda = 1 is feasible


def test_select_family_returns_member_of_grid():
    rng = np.random.default_rng(7)
    raw = [_blob_image(rng, size=40, k=3) for _ in range(20)]
    tuning = {}
    for fam in default_family_grid()[:5]:
        units = [unit_stats(fam.score(p), gt_components(m), lambdas=lambda_grid(201)) for p, m in raw]
        tuning[fam.label] = stack(units, small_max_size=40, lambdas=lambda_grid(201))
    assert select_family(tuning, 0.1) in tuning


def test_app_method_calibrates_and_applies():
    from app.calibration.datasets import CalibrationSample, CalibrationSet
    from app.calibration.registry import get_method
    from app.schemas.common import DataProvenance

    rng = np.random.default_rng(8)
    samples = []
    for _ in range(40):
        p, m = _blob_image(rng, size=48, k=2)
        samples.append(CalibrationSample(foreground_probability=p, ground_truth=m))
    cs = CalibrationSet(samples=samples, name="t", provenance=DataProvenance.SYNTHETIC, description="")
    for mid in ("lesion_aware_crc", "morphological_conformal"):
        method = get_method(mid)
        res = method.calibrate(cs, target_risk=0.2, n_bins=10)
        assert 0.0 <= res.threshold <= 1.0
        assert any("lesion miss" in w for w in res.validity_warnings)
        applied = method.apply(res, samples[0].foreground_probability)
        assert applied.abstain_mask.shape == (48, 48)
