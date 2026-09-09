"""API contract and end-to-end pipeline tests."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "intensity-mixture-seg-v1" in body["available_models"]
    assert isinstance(body["capabilities"], dict)


def test_root_carries_disclaimer(client):
    body = client.get("/").json()
    assert "not a medical device" in body["disclaimer"].lower()


def test_models_endpoint_reports_honest_provenance(client):
    body = client.get("/api/models").json()
    assert body["default_model_id"] in {m["id"] for m in body["models"]}

    for model in body["models"]:
        assert model["trained_on"], "every adapter must declare what it was trained on"
        assert model["status"] in ("implemented", "baseline", "experimental", "planned")
        # An adapter may legitimately be trained on medical data, but NOTHING in
        # this project may claim clinical validation.
        assert model["clinically_validated"] is False


def test_unknown_model_returns_404(client):
    response = client.get("/api/models/does-not-exist")
    assert response.status_code == 404
    assert response.json()["code"] == "model_not_found"


# ---------------------------------------------------------------- pipeline --
@pytest.fixture(scope="module")
def analysis(client, phantom_png):
    response = client.post(
        "/api/analyze",
        files={"file": ("phantom.png", phantom_png, "image/png")},
        data={"options": json.dumps({"task": "segmentation", "target_risk": 0.15})},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_analysis_has_the_full_hierarchy(analysis):
    assert analysis["status"] == "completed"
    for key in (
        "analysis_id",
        "metadata",
        "prediction",
        "uncertainty",
        "calibration",
        "distribution_shift",
        "risk_assessment",
    ):
        assert key in analysis, f"missing top-level key: {key}"

    uncertainty = analysis["uncertainty"]
    assert set(uncertainty) >= {"global", "objects", "pixel"}


def test_prediction_is_well_formed(analysis):
    prediction = analysis["prediction"]
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["inference_time_ms"] > 0

    probabilities = [c["probability"] for c in prediction["classes"]]
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-4)

    segmentation = prediction["segmentation"]
    assert sum(segmentation["class_pixel_fraction"]) == pytest.approx(1.0, abs=1e-4)
    # RLE run lengths must account for exactly every pixel.
    assert sum(segmentation["rle"][1::2]) == segmentation["height"] * segmentation["width"]


def test_segmentation_rle_decodes_to_the_declared_shape(analysis):
    from app.utils.arrays import rle_decode

    segmentation = analysis["prediction"]["segmentation"]
    labels = rle_decode(
        segmentation["rle"], segmentation["height"], segmentation["width"]
    )
    assert labels.shape == (segmentation["height"], segmentation["width"])
    assert labels.max() < segmentation["num_classes"]


def test_uncertainty_levels_are_bounded_and_consistent(analysis):
    global_level = analysis["uncertainty"]["global"]
    assert 0.0 <= global_level["score"] <= 1.0
    assert 0.0 <= global_level["high_uncertainty_fraction"] <= 1.0

    statistics = global_level["statistics"]
    assert statistics["minimum"] <= statistics["median"] <= statistics["maximum"]
    assert statistics["p90"] <= statistics["p95"] <= statistics["p99"]

    pixel = analysis["uncertainty"]["pixel"]
    assert pixel["heatmap_url"] and pixel["raw_url"]
    assert len(pixel["preview_grid"]) == 32

    for obj in analysis["uncertainty"]["objects"]:
        assert 0.0 <= obj["uncertainty"] <= 1.0
        assert obj["risk_level"] in ("LOW", "MODERATE", "HIGH", "REVIEW_REQUIRED")
        assert obj["source"] in ("high_uncertainty_region", "detection")


def test_calibration_result_is_honest(analysis):
    calibration = analysis["calibration"]
    assert calibration["method"] == "conformal_risk_control"
    assert calibration["status"] == "implemented"
    assert calibration["provenance"] == "synthetic"
    assert calibration["assumptions"], "assumptions must travel with the result"
    assert calibration["applied"] is True
    assert 0.0 <= calibration["coverage"] <= 1.0


def test_shift_result_is_present_and_bounded(analysis):
    shift = analysis["distribution_shift"]
    assert shift["status"] in ("ID", "MILD_SHIFT", "MODERATE_SHIFT", "SEVERE_SHIFT")
    assert 0.0 <= shift["shift_score"] <= 1.0
    assert len(shift["indicators"]) == 4
    assert shift["reference_distribution"]["histogram"]


def test_risk_assessment_is_transparent(analysis):
    risk = analysis["risk_assessment"]
    assert risk["risk_level"] in ("LOW", "MODERATE", "HIGH", "REVIEW_REQUIRED")
    assert risk["factors"] and risk["reasons"]
    assert "not a diagnostic result" in risk["disclaimer"].lower()


def test_non_medical_model_produces_a_warning(analysis):
    assert any("not a validated medical model" in w for w in analysis["warnings"])


def test_media_endpoints_return_png(client, analysis):
    analysis_id = analysis["analysis_id"]
    for key in ("source", "mask", "uncertainty-raw", "uncertainty-heatmap"):
        response = client.get(f"/api/analyses/{analysis_id}/media/{key}.png")
        assert response.status_code == 200, key
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_uncertainty_png_round_trips_to_the_reported_mean(client, analysis):
    """The transported PNG must actually encode the numbers we reported.

    Decoding the grayscale map and dividing by 255 must reproduce the Level-1
    mean to quantisation precision — proof the image is the data, not a
    decorative render.
    """
    analysis_id = analysis["analysis_id"]
    response = client.get(f"/api/analyses/{analysis_id}/media/uncertainty-raw.png")
    decoded = np.asarray(Image.open(io.BytesIO(response.content))).astype(float) / 255.0

    assert decoded.mean() == pytest.approx(
        analysis["uncertainty"]["global"]["score"], abs=2e-3
    )


def test_sub_resource_endpoints_match_the_full_analysis(client, analysis):
    analysis_id = analysis["analysis_id"]
    assert (
        client.get(f"/api/analyses/{analysis_id}/uncertainty").json()["global"]["score"]
        == analysis["uncertainty"]["global"]["score"]
    )
    assert (
        client.get(f"/api/analyses/{analysis_id}/calibration").json()["threshold"]
        == analysis["calibration"]["threshold"]
    )
    assert (
        client.get(f"/api/analyses/{analysis_id}/shift").json()["shift_score"]
        == analysis["distribution_shift"]["shift_score"]
    )
    assert (
        client.get(f"/api/analyses/{analysis_id}/risk").json()["risk_score"]
        == analysis["risk_assessment"]["risk_score"]
    )


def test_analysis_appears_in_list_and_dashboard(client, analysis):
    listing = client.get("/api/analyses").json()
    assert any(i["analysis_id"] == analysis["analysis_id"] for i in listing["items"])

    dashboard = client.get("/api/dashboard").json()
    assert dashboard["completed_analyses"] >= 1
    assert len(dashboard["uncertainty_histogram"]) == 10


def test_missing_analysis_returns_404(client):
    response = client.get("/api/analyses/pums_doesnotexist")
    assert response.status_code == 404
    assert response.json()["code"] == "analysis_not_found"


# ------------------------------------------------------------- input errors --
def test_invalid_file_is_rejected_with_422(client):
    response = client.post(
        "/api/analyze", files={"file": ("bad.png", b"not an image", "image/png")}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_image"


def test_malformed_options_json_is_rejected(client, phantom_png):
    response = client.post(
        "/api/analyze",
        files={"file": ("p.png", phantom_png, "image/png")},
        data={"options": "{not json"},
    )
    assert response.status_code == 422


def test_unknown_model_in_options_returns_404(client, phantom_png):
    response = client.post(
        "/api/analyze",
        files={"file": ("p.png", phantom_png, "image/png")},
        data={"options": json.dumps({"model_id": "nope"})},
    )
    assert response.status_code == 404


def test_detection_on_a_segmentation_only_model_is_rejected(client, phantom_png):
    response = client.post(
        "/api/analyze",
        files={"file": ("p.png", phantom_png, "image/png")},
        data={"options": json.dumps({"task": "detection"})},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "task_not_supported"


def test_classification_on_an_unsupporting_model_is_rejected(client, phantom_png):
    """An unsupported task must 400, never fall through to segmentation.

    Regression test: the pipeline previously ran segmentation and returned it
    labelled `task: "classification"`, reporting a classification no classifier
    produced.
    """
    response = client.post(
        "/api/analyze",
        files={"file": ("p.png", phantom_png, "image/png")},
        data={"options": json.dumps({"task": "classification"})},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "task_not_supported"
    assert "segmentation" in body["detail"]["supported_tasks"]


def test_no_adapter_silently_relabels_the_requested_task(client, phantom_png, registry):
    """Whatever task comes back must be one the model genuinely supports."""
    for model in registry.list_models():
        if not model.available:
            continue
        for task in ("segmentation", "detection", "classification"):
            response = client.post(
                "/api/analyze",
                files={"file": ("p.png", phantom_png, "image/png")},
                data={"options": json.dumps({"task": task, "model_id": model.id})},
            )
            supported = task in [t.value for t in model.tasks]
            if supported:
                assert response.status_code == 200, (model.id, task, response.text)
                assert response.json()["prediction"]["task"] == task
            else:
                assert response.status_code == 400, (model.id, task)


def test_detection_works_on_the_detection_adapter(client, phantom_png):
    response = client.post(
        "/api/analyze",
        files={"file": ("p.png", phantom_png, "image/png")},
        data={
            "options": json.dumps(
                {"task": "detection", "model_id": "region-proposal-det-v1"}
            )
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction"]["detections"] is not None
    for detection in body["prediction"]["detections"]:
        assert 0.0 <= detection["confidence"] <= 1.0
        assert detection["bbox"]["width"] > 0


# --------------------------------------------------------------- catalogues --
def test_calibration_methods_declare_status(client):
    body = client.get("/api/calibration/methods").json()
    statuses = {m["id"]: m["status"] for m in body["methods"]}
    assert statuses["conformal_risk_control"] == "implemented"
    assert statuses["spatially_aware_conformal"] == "planned"


def test_running_a_planned_method_returns_501(client):
    response = client.post(
        "/api/calibration/run", params={"method": "localized_adaptive_risk_control"}
    )
    assert response.status_code == 501


def test_shift_config_exposes_thresholds_and_limits(client):
    body = client.get("/api/shift/config").json()
    assert set(body["thresholds"]) == {"mild", "moderate", "severe"}
    assert body["limitations"]
    assert body["reference_domains"]


def test_evaluation_reports_unavailable_metrics_honestly(client):
    body = client.get("/api/evaluation", params={"include_calibration_run": False}).json()
    planned = {m["id"]: m for m in body["metrics"]}
    assert planned["detection_map"]["value"] is None
    assert planned["detection_map"]["unavailable_reason"]
    assert body["notes"]


def test_evaluation_with_calibration_run_returns_measured_metrics(client):
    """Exercises the calibration-metrics path, not just the cheap one.

    Regression test: the only evaluation test used
    `include_calibration_run=False`, so a broken return signature in
    `_calibration_set_metrics` reached the running server as a 500.
    """
    response = client.get("/api/evaluation", params={"include_calibration_run": True})
    assert response.status_code == 200, response.text
    body = response.json()

    measured = {m["id"]: m for m in body["metrics"] if m["value"] is not None}
    assert {"dice", "iou", "unc_error_corr", "aurc"} <= set(measured)

    for metric in measured.values():
        assert metric["provenance"] in ("computed", "synthetic")
        assert metric["sample_size"] > 0
    # The report must name the data its numbers describe.
    assert any(len(note) > 40 for note in body["notes"])
    assert body["risk_coverage_curve"]


def test_evaluation_self_test_passes(client):
    body = client.get("/api/evaluation/self-test").json()
    assert body["all_passed"] is True
