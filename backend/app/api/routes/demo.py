"""Live lesion-level CRC demo (frontend route /live-demo).

Every response is computed on request from real frozen-model posteriors by
app.services.lesion_demo: a fresh random partition, the experiment's calibration
rules, and risks measured on that partition's held-out test units.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from app.services.lesion_demo import FAMILIES, get_demo

router = APIRouter(prefix="/demo", tags=["demo"])


def _unavailable() -> JSONResponse:
    return JSONResponse(status_code=503, content={
        "code": "demo_unavailable", "message": get_demo().status()["reason"], "detail": {}})


def _check_uid(uid: int) -> None:
    n = get_demo().data().n
    if not 0 <= uid < n:
        raise HTTPException(status_code=404, detail=f"unit {uid} not in 0..{n - 1}")


@router.get("/status", summary="Demo data set, model and methods")
def demo_status():
    return get_demo().status()


@router.get("/run", summary="Calibrate every method on a fresh random partition and test it")
def demo_run(alpha: float = Query(0.10, ge=0.02, le=0.3), seed: int = Query(0, ge=0, le=1_000_000)):
    if not get_demo().available:
        return _unavailable()
    return get_demo().run(round(alpha, 4), seed)


@router.get("/validity", summary="Repeat calibration over many random partitions")
def demo_validity(alpha: float = Query(0.10, ge=0.02, le=0.3), reps: int = Query(200, ge=10, le=1000),
                  seed: int = Query(0, ge=0, le=1_000_000)):
    if not get_demo().available:
        return _unavailable()
    return get_demo().validity(round(alpha, 4), reps, seed)


@router.get("/rescues", summary="Images where pixel CRC misses a polyp that the joint rule finds")
def demo_rescues(alpha: float = Query(0.05, ge=0.02, le=0.3), splits: int = Query(100, ge=10, le=500)):
    if not get_demo().available:
        return _unavailable()
    return get_demo().rescues(round(alpha, 4), splits)


@router.get("/units/{uid}/image.jpg", summary="Letterboxed colour image of one unit")
def demo_image(uid: int):
    if not get_demo().available:
        return _unavailable()
    _check_uid(uid)
    return Response(get_demo().data().image_jpeg(uid), media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=600", "Vary": "Origin"})


@router.get("/units/{uid}/score.bin", summary="Score map of one unit as little-endian uint16 (score x 65535)")
def demo_score(uid: int, family: Literal["threshold", "dilation", "lesion_equalized"] = "threshold"):
    if not get_demo().available:
        return _unavailable()
    _check_uid(uid)
    assert family in FAMILIES
    s = get_demo().data().score(uid, family)
    q = np.round(np.clip(s, 0.0, 1.0) * 65535).astype("<u2")
    return Response(q.tobytes(), media_type="application/octet-stream",
                    headers={"Cache-Control": "private, max-age=600", "Vary": "Origin"})


@router.get("/units/{uid}/labels.bin", summary="Ground-truth lesion labels of one unit as uint8")
def demo_labels(uid: int):
    if not get_demo().available:
        return _unavailable()
    _check_uid(uid)
    return Response(get_demo().data().labels[uid].astype(np.uint8).tobytes(), media_type="application/octet-stream",
                    headers={"Cache-Control": "private, max-age=600", "Vary": "Origin"})
