"""Analysis repository.

Deliberately not a database. The abstract interface below is the seam a real
store (Postgres, S3 for the media blobs) slots into later; the JSON/in-memory
implementations keep the first phase dependency-free and inspectable.
"""

from __future__ import annotations

import abc
import json
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from app.core.errors import AnalysisNotFoundError
from app.core.logging import get_logger
from app.schemas.analysis import AnalysisResponse, AnalysisSummary

logger = get_logger(__name__)


def summarise(record: AnalysisResponse) -> AnalysisSummary:
    return AnalysisSummary(
        analysis_id=record.analysis_id,
        created_at=record.metadata.created_at,
        filename=record.metadata.image.filename,
        model_id=record.metadata.model_id,
        task=record.metadata.task,
        status=record.status,
        mean_uncertainty=record.uncertainty.global_.score,
        confidence=record.prediction.confidence,
        shift_score=record.distribution_shift.shift_score
        if record.distribution_shift
        else None,
        shift_status=record.distribution_shift.status.value
        if record.distribution_shift
        else None,
        risk_level=record.risk_assessment.risk_level.value,
        risk_score=record.risk_assessment.risk_score,
    )


class AnalysisRepository(abc.ABC):
    """Persistence interface for analysis records and their media blobs."""

    @abc.abstractmethod
    def save(self, record: AnalysisResponse) -> None: ...

    @abc.abstractmethod
    def get(self, analysis_id: str) -> AnalysisResponse: ...

    @abc.abstractmethod
    def list(self, limit: int = 50, offset: int = 0) -> tuple[list[AnalysisResponse], int]: ...

    @abc.abstractmethod
    def put_media(self, analysis_id: str, key: str, data: bytes) -> None: ...

    @abc.abstractmethod
    def get_media(self, analysis_id: str, key: str) -> bytes: ...

    @abc.abstractmethod
    def delete(self, analysis_id: str) -> None: ...


class InMemoryAnalysisRepository(AnalysisRepository):
    """Bounded LRU store. Default for tests; fine for a single-user demo."""

    def __init__(self, max_records: int = 200) -> None:
        self._records: OrderedDict[str, AnalysisResponse] = OrderedDict()
        self._media: dict[str, dict[str, bytes]] = {}
        self._max = max_records
        self._lock = threading.RLock()

    def save(self, record: AnalysisResponse) -> None:
        with self._lock:
            self._records[record.analysis_id] = record
            self._records.move_to_end(record.analysis_id)
            while len(self._records) > self._max:
                evicted, _ = self._records.popitem(last=False)
                self._media.pop(evicted, None)

    def get(self, analysis_id: str) -> AnalysisResponse:
        with self._lock:
            if analysis_id not in self._records:
                raise AnalysisNotFoundError(f"No analysis with id '{analysis_id}'")
            return self._records[analysis_id]

    def list(self, limit: int = 50, offset: int = 0) -> tuple[list[AnalysisResponse], int]:
        with self._lock:
            items = list(reversed(self._records.values()))
            return items[offset : offset + limit], len(items)

    def put_media(self, analysis_id: str, key: str, data: bytes) -> None:
        with self._lock:
            self._media.setdefault(analysis_id, {})[key] = data

    def get_media(self, analysis_id: str, key: str) -> bytes:
        with self._lock:
            blob = self._media.get(analysis_id, {}).get(key)
            if blob is None:
                raise AnalysisNotFoundError(
                    f"No media '{key}' for analysis '{analysis_id}'"
                )
            return blob

    def delete(self, analysis_id: str) -> None:
        with self._lock:
            self._records.pop(analysis_id, None)
            self._media.pop(analysis_id, None)


class JsonFileAnalysisRepository(AnalysisRepository):
    """Writes one JSON document plus a media folder per analysis.

    Survives a backend restart, which matters when demonstrating the dashboard,
    and keeps every result trivially inspectable on disk.
    """

    def __init__(self, directory: Path, max_records: int = 500) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._max = max_records
        self._cache = InMemoryAnalysisRepository(max_records=64)
        self._lock = threading.RLock()

    # ------------------------------------------------------------- paths --
    def _record_path(self, analysis_id: str) -> Path:
        return self.directory / f"{analysis_id}.json"

    def _media_dir(self, analysis_id: str) -> Path:
        return self.directory / analysis_id

    # ------------------------------------------------------------- crud ---
    def save(self, record: AnalysisResponse) -> None:
        with self._lock:
            self._cache.save(record)
            payload = record.model_dump(mode="json", by_alias=True)
            self._record_path(record.analysis_id).write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            self._prune()

    def get(self, analysis_id: str) -> AnalysisResponse:
        try:
            return self._cache.get(analysis_id)
        except AnalysisNotFoundError:
            pass

        path = self._record_path(analysis_id)
        if not path.exists():
            raise AnalysisNotFoundError(f"No analysis with id '{analysis_id}'")
        record = AnalysisResponse.model_validate(json.loads(path.read_text("utf-8")))
        self._cache.save(record)
        return record

    def list(self, limit: int = 50, offset: int = 0) -> tuple[list[AnalysisResponse], int]:
        with self._lock:
            paths = sorted(
                self.directory.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        records: list[AnalysisResponse] = []
        for path in paths[offset : offset + limit]:
            try:
                records.append(self.get(path.stem))
            except Exception as exc:  # a corrupt file must not break the list
                logger.warning("Skipping unreadable analysis %s: %s", path.stem, exc)
        return records, len(paths)

    def put_media(self, analysis_id: str, key: str, data: bytes) -> None:
        media_dir = self._media_dir(analysis_id)
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / f"{key}.png").write_bytes(data)
        self._cache.put_media(analysis_id, key, data)

    def get_media(self, analysis_id: str, key: str) -> bytes:
        try:
            return self._cache.get_media(analysis_id, key)
        except AnalysisNotFoundError:
            pass
        path = self._media_dir(analysis_id) / f"{key}.png"
        if not path.exists():
            raise AnalysisNotFoundError(
                f"No media '{key}' for analysis '{analysis_id}'"
            )
        data = path.read_bytes()
        self._cache.put_media(analysis_id, key, data)
        return data

    def delete(self, analysis_id: str) -> None:
        import shutil

        with self._lock:
            self._cache.delete(analysis_id)
            self._record_path(analysis_id).unlink(missing_ok=True)
            shutil.rmtree(self._media_dir(analysis_id), ignore_errors=True)

    def _prune(self) -> None:
        """Drop the oldest records once the store exceeds `max_records`."""
        paths = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for path in paths[: max(0, len(paths) - self._max)]:
            self.delete(path.stem)


def build_repository(backend: str, directory: Path) -> AnalysisRepository:
    if backend == "memory":
        return InMemoryAnalysisRepository()
    return JsonFileAnalysisRepository(directory)


def iso(value: datetime) -> str:
    return value.isoformat()
