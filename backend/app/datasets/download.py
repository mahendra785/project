"""Dataset acquisition.

Downloads are explicit and auditable: the licence and citation are printed
before any bytes are written, archives are extracted with path-traversal
guards, and nothing is fetched implicitly at import time or during a request.
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from app.core.config import REPO_ROOT
from app.core.logging import get_logger
from app.datasets.registry import DatasetSpec, get_spec

logger = get_logger(__name__)

RAW_DIR = REPO_ROOT / "data" / "raw"
EXTRACT_DIR = REPO_ROOT / "data" / "datasets"


def dataset_root(spec: DatasetSpec) -> Path:
    return EXTRACT_DIR / spec.id


def is_available(dataset_id: str) -> bool:
    """True when the dataset has been downloaded and extracted."""
    try:
        spec = get_spec(dataset_id)
    except KeyError:
        return False
    root = dataset_root(spec)
    return root.exists() and any(root.rglob("*"))


def licence_banner(spec: DatasetSpec) -> str:
    return (
        f"\n{'=' * 74}\n"
        f"{spec.name}\n"
        f"{'=' * 74}\n"
        f"  Modality   : {spec.modality.value}\n"
        f"  Target     : {spec.target}\n"
        f"  Use case   : {spec.use_case}\n"
        f"  Size       : ~{spec.approx_mb} MB\n"
        f"  Licence    : {spec.licence}  ({spec.licence_url})\n"
        f"  Homepage   : {spec.homepage}\n"
        f"  Citation   : {spec.citation}\n"
        f"{'=' * 74}\n"
        "  By downloading you accept the licence above. Cite the source in any\n"
        "  publication. This data is for research use; it is not a substitute\n"
        "  for clinical judgement.\n"
        f"{'=' * 74}\n"
    )


def _report(done: int, total: int) -> None:
    if total <= 0:
        return
    pct = done * 100 // total
    bar = "#" * (pct // 3)
    print(f"\r  [{bar:<33}] {pct:3d}%  {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="")


def download_archive(spec: DatasetSpec, *, force: bool = False) -> Path:
    """Fetch the archive into `data/raw/`, resuming is not attempted."""
    if not spec.downloadable:
        raise RuntimeError(
            f"'{spec.id}' cannot be downloaded automatically: it requires "
            f"registration. See {spec.homepage}"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".tar" if spec.url and spec.url.endswith(".tar") else ".zip"
    target = RAW_DIR / f"{spec.id}{suffix}"

    if target.exists() and not force:
        logger.info("Archive already present: %s", target.name)
        return target

    print(f"Downloading {spec.name} ...")
    with urllib.request.urlopen(spec.url, timeout=120) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with open(target, "wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                _report(done, total)
    print()
    return target


def _is_within(base: Path, candidate: Path) -> bool:
    """Guard against archive members escaping the extraction directory."""
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def extract_archive(spec: DatasetSpec, archive: Path, *, force: bool = False) -> Path:
    root = dataset_root(spec)
    if root.exists() and any(root.rglob("*")) and not force:
        logger.info("Already extracted: %s", root)
        return root

    if force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting to {root} ...")
    if archive.suffix == ".tar":
        with tarfile.open(archive) as tar:
            members = [
                m
                for m in tar.getmembers()
                # Skip macOS resource forks that MSD archives carry.
                if not Path(m.name).name.startswith("._")
                and _is_within(root, root / m.name)
            ]
            tar.extractall(root, members=members, filter="data")
    else:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if Path(member).name.startswith("._"):
                    continue
                if not _is_within(root, root / member):
                    logger.warning("Skipping unsafe archive member: %s", member)
                    continue
                zf.extract(member, root)
    return root


def ensure_dataset(dataset_id: str, *, force: bool = False) -> Path:
    """Download + extract if needed; return the extracted root directory."""
    spec = get_spec(dataset_id)
    if is_available(dataset_id) and not force:
        return dataset_root(spec)
    archive = download_archive(spec, force=force)
    return extract_archive(spec, archive, force=force)
