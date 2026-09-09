"""Dataset routes.

Surfaces what data actually backs the calibration and reference domain, with
its licence and citation. A research UI that shows numbers without showing
their source invites exactly the misreading this project is trying to avoid.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import SettingsDep
from app.datasets.download import is_available
from app.datasets.registry import DATASETS, list_specs
from app.schemas.common import ApiModel, DataProvenance

router = APIRouter(tags=["datasets"])


class DatasetInfo(ApiModel):
    id: str
    name: str
    modality: str
    target: str
    use_case: str
    licence: str
    licence_url: str
    citation: str
    homepage: str
    approx_mb: float
    downloadable: bool
    requires_registration: bool
    #: True when it has been downloaded and extracted on this machine.
    available: bool
    notes: str


class ActiveDatasetInfo(ApiModel):
    """What is really backing calibration and the reference domain right now."""

    dataset_id: str | None
    dataset_name: str | None
    available: bool
    provenance: DataProvenance
    #: Human-readable statement of what the current results describe.
    summary: str
    download_command: str | None = None


class DatasetsResponse(ApiModel):
    active: ActiveDatasetInfo
    datasets: list[DatasetInfo]


def _describe(spec) -> DatasetInfo:
    return DatasetInfo(
        id=spec.id,
        name=spec.name,
        modality=spec.modality.value,
        target=spec.target,
        use_case=spec.use_case,
        licence=spec.licence,
        licence_url=spec.licence_url,
        citation=spec.citation,
        homepage=spec.homepage,
        approx_mb=spec.approx_mb,
        downloadable=spec.downloadable,
        requires_registration=spec.requires_registration,
        available=is_available(spec.id),
        notes=spec.notes,
    )


@router.get(
    "/datasets",
    response_model=DatasetsResponse,
    summary="Registered datasets and which one is backing current results",
)
def datasets(settings: SettingsDep) -> DatasetsResponse:
    configured = DATASETS.get(settings.dataset_id)
    available = settings.use_real_dataset and is_available(settings.dataset_id)

    if available and configured is not None:
        active = ActiveDatasetInfo(
            dataset_id=configured.id,
            dataset_name=configured.name,
            available=True,
            provenance=DataProvenance.COMPUTED,
            summary=(
                f"Calibration and the reference domain are built from {configured.name} "
                f"({configured.modality.value}, target: {configured.target}) using "
                "patient-disjoint splits with expert annotations. Reported metrics are "
                "genuine measurements on this dataset; they are not a clinical validation."
            ),
        )
    else:
        active = ActiveDatasetInfo(
            dataset_id=None,
            dataset_name=None,
            available=False,
            provenance=DataProvenance.SYNTHETIC,
            summary=(
                "No real dataset is downloaded, so calibration and the reference "
                "domain fall back to procedurally generated phantoms. Every derived "
                "figure describes phantom behaviour, NOT medical performance."
            ),
            download_command=(
                f"python scripts/download_dataset.py {settings.dataset_id}"
            ),
        )

    return DatasetsResponse(
        active=active,
        datasets=[_describe(spec) for spec in list_specs()],
    )
