"""Research summary for the dashboard.

Aggregates what the project can currently claim, from the files that back it:
the active dataset, the trained model's own training report, the calibration
method catalogue, and the experiment registry (`docs/research/experiments.json`)
joined with the raw result files each experiment names. Nothing here is typed in
by hand except the human verdict record, which is itself a committed file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter

from app.api.dependencies import RegistryDep, SettingsDep
from app.api.routes.datasets import ActiveDatasetInfo, datasets as datasets_route
from app.calibration.registry import list_methods
from app.core.config import BACKEND_ROOT, REPO_ROOT
from app.models.registry import resolve_default_model_id
from app.schemas.common import ApiModel, ImplementationStatus

router = APIRouter(tags=["research"])

EXPERIMENTS_FILE = REPO_ROOT / "docs" / "research" / "experiments.json"


class ExperimentRow(ApiModel):
    alpha: float
    n_cal: int
    multi_lesion_fraction: float
    lambda_pix: float
    R_pix: float
    R_les_at_lambda_pix: float
    G: float
    G_sd: float | None
    R_les_test: float | None
    lambda_les: float
    lambda_union: float
    inflation_union: float


class Experiment(ApiModel):
    id: str
    week: int
    title: str
    dataset: str
    role: str
    verdict_by_rule: str
    recommendation: str
    recommendation_reason: str
    next: str
    pre_registered_statistic: str
    protocol: dict
    report: str | None
    memo: str | None
    rows: list[ExperimentRow]
    seeds: list[int]
    m_justification: str | None


class ModelSummary(ApiModel):
    id: str
    name: str
    trained_on: str
    trained_dataset: str | None
    is_medical: bool
    clinically_validated: bool
    training_report: dict | None


class ResearchSummary(ApiModel):
    dataset: ActiveDatasetInfo
    model: ModelSummary
    method_status_counts: dict[str, int]
    experiments: list[Experiment]
    disclaimer: str


def _aggregate_rows(results_path: Path) -> tuple[list[ExperimentRow], list[int], str | None]:
    if not results_path.exists():
        return [], [], None
    payload = json.loads(results_path.read_text("utf-8"))
    rows = payload.get("rows", [])
    seeds = sorted({int(r["seed"]) for r in rows})
    out: list[ExperimentRow] = []
    for alpha in sorted({float(r["alpha"]) for r in rows}):
        rs = [r for r in rows if float(r["alpha"]) == alpha]

        def mean(key: str) -> float:
            return float(np.mean([r[key] for r in rs]))

        gs = np.asarray([r["G"] for r in rs], dtype=float)
        out.append(
            ExperimentRow(
                alpha=alpha,
                n_cal=int(rs[0]["n_cal"]),
                multi_lesion_fraction=round(mean("multi_lesion_fraction"), 3),
                lambda_pix=round(mean("lambda_pix"), 3),
                R_pix=round(mean("R_pix_at_lpix"), 4),
                R_les_at_lambda_pix=round(mean("R_les_at_lpix"), 4),
                G=round(float(gs.mean()), 2),
                G_sd=round(float(gs.std()), 2) if len(gs) > 1 else None,
                R_les_test=round(mean("R_les_test_at_lpix"), 4) if "R_les_test_at_lpix" in rs[0] else None,
                lambda_les=round(mean("lambda_les"), 3),
                lambda_union=round(mean("lambda_union"), 3),
                inflation_union=round(mean("infl_union"), 2),
            )
        )
    return out, seeds, payload.get("m_justification")


@router.get("/research/summary", response_model=ResearchSummary, summary="Everything the dashboard needs")
def research_summary(settings: SettingsDep, registry: RegistryDep) -> ResearchSummary:
    active = datasets_route(settings).active

    model_id = resolve_default_model_id(settings.default_model_id, settings.fallback_model_id)
    info = registry.describe(model_id)
    report_path = BACKEND_ROOT / "var" / "models" / "rf_pixel_seg.report.json"
    training_report = (
        json.loads(report_path.read_text("utf-8")) if report_path.exists() and model_id == "rf-pixel-seg-v1" else None
    )

    counts = {s.value: 0 for s in ImplementationStatus}
    for method in list_methods():
        counts[method.status.value] += 1

    experiments: list[Experiment] = []
    if EXPERIMENTS_FILE.exists():
        for e in json.loads(EXPERIMENTS_FILE.read_text("utf-8")).get("experiments", []):
            # Audits and design notes have no numeric result file; show their record anyway.
            rows, seeds, m_note = (
                _aggregate_rows(REPO_ROOT / e["results"]) if e.get("results") else ([], [], None)
            )
            experiments.append(
                Experiment(
                    id=e["id"], week=e["week"], title=e["title"], dataset=e["dataset"], role=e["role"],
                    verdict_by_rule=e["verdict_by_rule"], recommendation=e["recommendation"],
                    recommendation_reason=e["recommendation_reason"], next=e["next"],
                    pre_registered_statistic=e["pre_registered_statistic"], protocol=e.get("protocol", {}),
                    report=e.get("report"), memo=e.get("memo"), rows=rows, seeds=seeds, m_justification=m_note,
                )
            )

    return ResearchSummary(
        dataset=active,
        model=ModelSummary(
            id=info.id, name=info.name, trained_on=info.trained_on, trained_dataset=info.trained_dataset,
            is_medical=info.is_medical, clinically_validated=info.clinically_validated,
            training_report=training_report,
        ),
        method_status_counts=counts,
        experiments=experiments,
        disclaimer=(
            "Research prototype. Every figure is a measurement on the named dataset with a "
            "classical baseline segmenter. No clinical validation has been performed."
        ),
    )
