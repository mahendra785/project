"""Live lesion-level CRC demo: service and API on a small synthetic bundle.

The real bundle (var/demo/…) is built from frozen-model posteriors and is not part of
the repository, so these tests build a bundle of Gaussian-bump scans with the same
layout and check the calibration contract and the binary payloads.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.services import lesion_demo
from app.services.lesion_demo import METHODS, LesionDemo

SIZE = 48


def _make_bundle(path, n=60, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    prob, labels, valid = [], [], []
    for _ in range(n):
        p = rng.uniform(0, 0.05, (SIZE, SIZE))
        lab = np.zeros((SIZE, SIZE), np.uint8)
        k = rng.integers(1, 4)
        for j in range(k):
            cy, cx, r = rng.integers(8, SIZE - 8), rng.integers(8, SIZE - 8), rng.integers(2, 7)
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            amp = rng.uniform(0.2, 1.0) * (r / 6) ** 1.5
            p = np.maximum(p, amp * np.exp(-d2 / (2 * r * r)))
            lab[(d2 <= r * r) & (lab == 0)] = j + 1
        # renumber so labels are 1..K connected components, as the pipeline produces
        from scipy import ndimage

        lab = ndimage.label(lab > 0, structure=np.ones((3, 3)))[0].astype(np.uint8)
        prob.append(p)
        labels.append(lab)
        valid.append(np.ones((SIZE, SIZE), bool))
    jpeg = np.frombuffer(b"\xff\xd8fakejpeg\xff\xd9" * n, dtype=np.uint8)
    offsets = np.arange(n + 1, dtype=np.int64) * 13
    manifest = dict(dataset="synthetic", arch="none", seed=seed, epochs=0, best_val_dice=1.0,
                    pool_metrics=dict(dice=1.0, lesion_recall=1.0), small_cuts_from_train=dict(q33=30.0), env={})
    np.savez_compressed(path, prob=np.asarray(prob, np.float16), labels=np.asarray(labels),
                        valid=np.asarray(valid), ids=np.asarray([f"u{i}" for i in range(n)]),
                        jpeg=jpeg, jpeg_offsets=offsets, manifest=np.asarray(json.dumps(manifest)))
    return path


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return LesionDemo(_make_bundle(tmp_path_factory.mktemp("demo") / "bundle.npz"))


def test_run_reports_every_method_on_held_out_units(demo):
    out = demo.run(0.1, seed=3)
    assert out["n_cal"] + out["n_test"] + out["n_tune"] == demo.data().n
    assert {m["id"] for m in out["methods"]} == {m.id for m in METHODS}
    joint = next(m for m in out["methods"] if m["id"] == "joint")
    pixel = next(m for m in out["methods"] if m["id"] == "pixel")
    # the joint threshold is the largest per-risk threshold, so its set is never smaller
    assert joint["lam"][0] >= pixel["lam"][0]
    assert joint["lam"][0] >= max(out["curves"][r]["lam_hat"] for r in ("pix", "les", "small"))
    assert len(out["units"]) == out["n_test"]


def test_lesion_flags_follow_lemma2(demo):
    out = demo.run(0.1, seed=5)
    cut = {m["id"]: m["cutoff"][0] for m in out["methods"] if m["id"] != "strata"}
    for unit in out["units"]:
        for lesion in unit["lesions"]:
            assert lesion["found"]["pixel"] == (lesion["stat"] >= cut["pixel"] - 1e-12)
            # a lesion found by pixel CRC is found by the (larger) joint set too
            assert not lesion["found"]["pixel"] or lesion["found"]["joint"]
        assert unit["rescued"] == sum(le["found"]["joint"] and not le["found"]["pixel"] for le in unit["lesions"])


def test_validity_means_respect_alpha_for_joint(demo):
    out = demo.validity(0.15, reps=40)
    joint = next(m for m in out["methods"] if m["id"] == "joint")
    for risk in ("pix", "les", "small"):
        assert joint["stats"][risk]["mean"] <= 0.15 + 0.03
    assert joint["stats"]["rel_vs_joint"]["mean"] == pytest.approx(0.0)


def test_rescues_match_the_split_they_point_to(demo):
    out = demo.rescues(0.1, splits=30)
    for hit in out["units"][:5]:
        run = demo.run(0.1, seed=hit["seed"])
        unit = next(u for u in run["units"] if u["uid"] == hit["uid"])
        assert unit["rescued"] >= 1
        assert 1 <= hit["n_rescued"] <= hit["n_test"]


def test_api_endpoints(client, demo, monkeypatch):
    monkeypatch.setattr(lesion_demo, "_demo", demo)
    status = client.get("/api/demo/status").json()
    assert status["available"] and status["size"] == SIZE
    run = client.get("/api/demo/run", params={"alpha": 0.1, "seed": 1})
    assert run.status_code == 200 and run.json()["units"]
    score = client.get("/api/demo/units/0/score.bin", params={"family": "lesion_equalized"})
    assert score.status_code == 200 and len(score.content) == SIZE * SIZE * 2
    labels = client.get("/api/demo/units/0/labels.bin")
    assert len(labels.content) == SIZE * SIZE
    assert client.get("/api/demo/units/9999/labels.bin").status_code == 404
    assert client.get("/api/demo/rescues", params={"alpha": 0.1, "splits": 10}).status_code == 200


def test_api_reports_missing_bundle(client, tmp_path, monkeypatch):
    monkeypatch.setattr(lesion_demo, "_demo", LesionDemo(tmp_path / "missing.npz"))
    assert client.get("/api/demo/status").json()["available"] is False
    assert client.get("/api/demo/run").status_code == 503
