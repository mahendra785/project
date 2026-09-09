"""Bundled sample cases: list them, serve their images, run them server-side.

The samples in `data/sample/` are real test-split slices exported by
`scripts/export_samples.py`. Exposing them here lets the dashboard populate
itself with one click instead of asking the user to hunt for files to upload.
Only files named in `manifest.json` are served — the path parameter can never
reach outside the sample directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import Response

from app.api.dependencies import AnalysisServiceDep, RepositoryDep
from app.core.config import REPO_ROOT
from app.core.errors import AnalysisNotFoundError
from app.schemas.analysis import AnalysisOptions, AnalysisResponse
from app.schemas.common import ApiModel

router = APIRouter(tags=["samples"])

SAMPLE_DIR = REPO_ROOT / "data" / "sample"


class SampleCase(ApiModel):
    id: str
    image: str
    ground_truth: str | None
    shifted_variant: str | None
    dataset: str
    dataset_name: str
    modality: str
    target: str
    case_id: str
    split: str
    foreground_fraction: float
    licence: str
    #: Analyses already run on this exact file (or its shifted variant).
    analysis_ids: list[str]
    shifted_analysis_ids: list[str]


class SamplesResponse(ApiModel):
    note: str
    citation: str
    licence: str
    samples: list[SampleCase]


def _manifest() -> dict:
    path = SAMPLE_DIR / "manifest.json"
    if not path.exists():
        return {"note": "No samples exported. Run scripts/export_samples.py.", "citation": "",
                "licence": "", "samples": []}
    return json.loads(path.read_text("utf-8"))


def _allowed_files(manifest: dict) -> set[str]:
    names: set[str] = set()
    for s in manifest.get("samples", []):
        for key in ("sample", "ground_truth", "shifted_variant"):
            if s.get(key):
                names.add(Path(s[key]).name)
    return names


@router.get("/samples", response_model=SamplesResponse, summary="Bundled sample cases")
def list_samples(repository: RepositoryDep) -> SamplesResponse:
    manifest = _manifest()
    records, _ = repository.list(limit=500)
    by_filename: dict[str, list[str]] = {}
    for record in records:
        by_filename.setdefault(record.metadata.image.filename, []).append(record.analysis_id)

    samples = [
        SampleCase(
            id=Path(s["sample"]).stem,
            image=s["sample"],
            ground_truth=s.get("ground_truth"),
            shifted_variant=s.get("shifted_variant"),
            dataset=s["dataset"],
            dataset_name=s["dataset_name"],
            modality=s["modality"],
            target=s["target"],
            case_id=s["case_id"],
            split=s.get("split", "test"),
            foreground_fraction=float(s.get("foreground_fraction", 0.0)),
            licence=s["licence"],
            analysis_ids=by_filename.get(s["sample"], []),
            shifted_analysis_ids=by_filename.get(s.get("shifted_variant", ""), []),
        )
        for s in manifest.get("samples", [])
    ]
    return SamplesResponse(
        note=manifest.get("note", ""),
        citation=manifest.get("citation", ""),
        licence=manifest.get("licence", ""),
        samples=samples,
    )


@router.get("/samples/files/{filename}", summary="Serve a bundled sample image", response_class=Response)
def sample_file(filename: str) -> Response:
    if filename not in _allowed_files(_manifest()):
        raise AnalysisNotFoundError(f"'{filename}' is not a bundled sample")
    path = SAMPLE_DIR / filename
    return Response(path.read_bytes(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.post(
    "/samples/{filename}/analyze",
    response_model=AnalysisResponse,
    summary="Run the full pipeline on a bundled sample",
)
def analyze_sample(filename: str, service: AnalysisServiceDep) -> AnalysisResponse:
    if filename not in _allowed_files(_manifest()):
        raise AnalysisNotFoundError(f"'{filename}' is not a bundled sample")
    if filename.endswith("_groundtruth.png"):
        raise AnalysisNotFoundError("Ground-truth masks are annotations, not inputs")
    data = (SAMPLE_DIR / filename).read_bytes()
    return service.analyze(
        data, filename=filename, content_type="image/png", options=AnalysisOptions()
    )
