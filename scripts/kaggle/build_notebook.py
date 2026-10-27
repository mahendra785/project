"""Build the Kaggle notebook for the revision runs from the repository files (verbatim copies).

    python scripts/kaggle/build_notebook.py   ->  scripts/kaggle/pums_revision.ipynb
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BACK = REPO / "backend"
OUT = Path(__file__).resolve().parent / "pums_revision.ipynb"

FILES = {  # path inside the notebook's code tree -> repository file
    "backend/app/__init__.py": None,
    "backend/app/calibration/__init__.py": None,
    "backend/app/core/__init__.py": None,
    "backend/app/calibration/families.py": BACK / "app/calibration/families.py",
    "backend/app/calibration/losses.py": BACK / "app/calibration/losses.py",
    "backend/scripts/experiments/kaggle_train.py": BACK / "scripts/experiments/kaggle_train.py",
    "backend/scripts/experiments/mgcrc_v2.py": BACK / "scripts/experiments/mgcrc_v2.py",
    "backend/scripts/experiments/aggregate_v2.py": BACK / "scripts/experiments/aggregate_v2.py",
}
CONFIG_STUB = '''"""Kaggle paths: large intermediates in /tmp, results in /kaggle/working (kept with the version)."""
from pathlib import Path

BACKEND_ROOT = Path("/tmp/pums_backend")      # posteriors, stats caches (not saved)
REPO_ROOT = Path("/kaggle/working/pums")      # docs/research/results_v2 is saved with the notebook version
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [md("""
# PUMS revision runs: lesion-level joint CRC (pre-registration Amendment 3)

Runs the **frozen-model protocol** on a Kaggle GPU: download data from the official sources, train each
backbone once on 40% of the units, freeze it, predict the other 60%, then run all conformal calibration,
the clustered bootstrap, the ablations, and the aggregation. **Output:** `/kaggle/working/revision_results_<JOB>.zip`
(small; contains no images).

**Settings (right panel):** Accelerator = GPU (T4 x2 or P100) · Internet = On · Persistence = off.
Run with **Save Version → Save & Run All (Commit)**. Do not use interactive mode; a commit runs for up to 12 h unattended.

| Session | `JOB` | Approx. time |
|---|---|---|
| 1 | `"2d"` | Kvasir-SEG + FracAtlas, 2 backbones x 3 seeds, calibration + ablations: ~4-6 h |
| 2 | `"task08"`, `ARCHS=["unet-r34"]` | MSD Task08 liver tumours (patient-disjoint), 3 seeds: ~5-7 h |
| 3 | `"task08"`, `ARCHS=["unet-mitb2"]` | same, second backbone: ~6-8 h |
"""),
         code("""
# ======== CONFIG: edit before each session ========
JOB   = "2d"                              # "2d" or "task08"
ARCHS = ["unet-r34", "unet-mitb2"]        # backbones for this session
SEEDS = [0, 1, 2]                         # model seeds (independent replicates)
RUN_SMOKE = True                          # 1-epoch, <=40-unit smoke test first (outputs deleted)
"""),
         code("""
import subprocess, sys, os, shutil, json, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch", "nibabel"], check=True)
import torch, segmentation_models_pytorch as smp
print("torch", torch.__version__, "| smp", smp.__version__, "| GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), ("NO GPU: set Settings -> Accelerator -> 'GPU T4 x2' in the right panel, "
                                   "then Save Version -> Save & Run All (Commit) again")
os.system("df -h /tmp /kaggle/working | cat; nproc; free -g")
"""),
         md("## Code (verbatim copies of the repository files)"),
         code("""
for d in ["backend/app/calibration", "backend/app/core", "backend/scripts/experiments", "docs/research/results_v2"]:
    os.makedirs(f"/kaggle/working/pums/{d}", exist_ok=True)
""")]

for rel, src in FILES.items():
    text = "# package marker\n" if src is None else src.read_text(encoding="utf-8")
    cells.append(code(f"%%writefile /kaggle/working/pums/{rel}\n{text}"))
cells.append(code(f"%%writefile /kaggle/working/pums/backend/app/core/config.py\n{CONFIG_STUB}"))

cells += [md("## Data (official sources)"), code('''
DATA = "/tmp/pums_data"
os.makedirs(DATA, exist_ok=True)
import hashlib
# Integrity is checked against SHA-256 / size of the copies used in the preliminary study, so falling
# back to --no-check-certificate (Simula's TLS certificate fails verification) cannot admit a wrong file.
EXPECTED = {"kvasir-seg.zip": ("03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7", 46227172),
            "fracatlas.zip": ("b67ec2d290a022b3dcf47f78e9a37f7edcc80592c0571f439355bf00bd9f0e23", 338436460),
            "task08.tar": (None, 9347254784)}
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()
def fetch(url, dest):
    if not os.path.exists(dest):
        base = ["wget", "-q", "--tries=20", "--continue", "-O", dest, url]
        if subprocess.run(base).returncode != 0:
            print("certificate/transfer problem; retrying without TLS verification (checksum enforced below)")
            if os.path.exists(dest):
                os.remove(dest)
            subprocess.run(base[:1] + ["--no-check-certificate"] + base[1:], check=True)
    digest, size = EXPECTED[os.path.basename(dest)]
    got = os.path.getsize(dest)
    assert got == size, f"{dest}: size {got} != expected {size}"
    if digest:
        assert sha256(dest) == digest, f"{dest}: SHA-256 mismatch"
    print(dest, got // 2**20, "MB, verified")
def unpack(archive, target):
    if not os.path.isdir(target):
        os.makedirs(target)
        shutil.unpack_archive(archive, target)
if JOB == "2d":
    fetch("https://datasets.simula.no/downloads/kvasir-seg.zip", f"{DATA}/kvasir-seg.zip")
    unpack(f"{DATA}/kvasir-seg.zip", f"{DATA}/kvasir-seg")
    fetch("https://ndownloader.figshare.com/files/65518038", f"{DATA}/fracatlas.zip")
    unpack(f"{DATA}/fracatlas.zip", f"{DATA}/fracatlas")
else:
    fetch("https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar", f"{DATA}/task08.tar")
    unpack(f"{DATA}/task08.tar", f"{DATA}/msd-hepatic-vessel")
    os.remove(f"{DATA}/task08.tar")
os.system(f"du -sh {DATA}/* | cat")
'''), md("## Smoke test (Amendment 3: <=40 units, 1 epoch; outputs deleted)"), code('''
EXP = "/kaggle/working/pums/backend/scripts/experiments"
DATASETS = ["kvasir-seg", "fracatlas"] if JOB == "2d" else ["msd-hepatic-vessel"]
if RUN_SMOKE:
    for ds in DATASETS:
        for a in ARCHS:
            r = subprocess.run([sys.executable, f"{EXP}/kaggle_train.py", "--dataset", ds, "--arch", a, "--seed", "0",
                                "--smoke", "--data-root", DATA], capture_output=True, text=True)
            print(ds, a, "OK" if r.returncode == 0 else "FAILED"); print(r.stdout[-1500:], r.stderr[-3000:])
            assert r.returncode == 0, "smoke test failed: stop and report the error above"
    shutil.rmtree("/tmp/pums_backend/var/smoke", ignore_errors=True)
'''), md("## Train the frozen backbones"), code('''
LOGS = "/kaggle/working/pums/docs/research/results_v2/logs"
os.makedirs(LOGS, exist_ok=True)
def sh(cmd, log):
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(open(log).read()[-800:]); assert p.returncode == 0, f"failed: {log}"
for ds in DATASETS:
    for a in ARCHS:
        for s in SEEDS:
            t = time.time()
            sh([sys.executable, f"{EXP}/kaggle_train.py", "--dataset", ds, "--arch", a, "--seed", str(s), "--data-root", DATA],
               f"{LOGS}/train_{ds}_{a}_{s}.log")
            print(f"== {ds} {a} seed {s}: {time.time()-t:.0f}s", flush=True)
'''), md("## Conformal calibration, bootstrap, ablations"), code('''
for ds in DATASETS:
    for a in ARCHS:
        for s in SEEDS:
            sh([sys.executable, f"{EXP}/mgcrc_v2.py", "--dataset", ds, "--arch", a, "--seed", str(s), "--tag", "main"],
               f"{LOGS}/main_{ds}_{a}_{s}.log")
ABL = [("abltau25", ["--tau", "0.25"]), ("abltau75", ["--tau", "0.75"]), ("ablq25", ["--small-q", "25"]),
       ("ablq50", ["--small-q", "50"]), ("ablncal50", ["--ncal", "50"]), ("ablncal100", ["--ncal", "100"]),
       ("ablncal200", ["--ncal", "200"])]
if JOB == "2d":
    ABL.append(("ablconn4", ["--conn4"]))
if "unet-r34" in ARCHS:
    for ds in DATASETS:
        for s in SEEDS:
            for tag, extra in ABL:
                sh([sys.executable, f"{EXP}/mgcrc_v2.py", "--dataset", ds, "--arch", "unet-r34", "--seed", str(s),
                    "--tag", tag, "--boot", "50", "--boot-reps", "50", *extra], f"{LOGS}/{tag}_{ds}_{s}.log")
'''), md("## Aggregate and package"), code('''
sh([sys.executable, f"{EXP}/aggregate_v2.py"], f"{LOGS}/aggregate_{JOB}.log")
man = "/kaggle/working/pums/docs/research/results_v2/manifests"
os.makedirs(man, exist_ok=True)
for root, _, files in os.walk("/tmp/pums_backend/var/posteriors_frozen"):
    for f in files:
        if f.endswith(".json"):
            shutil.copy(os.path.join(root, f), f"{man}/{os.path.basename(root)}__{f}")
out = shutil.make_archive(f"/kaggle/working/revision_results_{JOB}_{'-'.join(ARCHS)}", "zip",
                          "/kaggle/working/pums/docs/research", "results_v2")
print(out, os.path.getsize(out) // 2**20, "MB")
'''), md("""
## After the run
Open the version's **Output** tab and download `revision_results_*.zip`. Put every zip (one per session) into
`docs/research/results_v2_kaggle/` in the repository and tell Claude. It merges them and runs the final aggregation,
figures, and paper build locally.
""")]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")

# Ready-to-run copies for each session: only the CONFIG cell differs.
SESSIONS = {
    "pums_session1_2d.ipynb": ('"2d"', '["unet-r34", "unet-mitb2"]'),
    "pums_session2_task08_unet-r34.ipynb": ('"task08"', '["unet-r34"]'),
    "pums_session3_task08_unet-mitb2.ipynb": ('"task08"', '["unet-mitb2"]'),
}
cfg_idx = next(i for i, c in enumerate(cells) if "CONFIG: edit before each session" in "".join(c["source"]))
base_cfg = "".join(cells[cfg_idx]["source"])
for name, (job, archs) in SESSIONS.items():
    src = (base_cfg.replace('JOB   = "2d"', f"JOB   = {job}")
                   .replace('ARCHS = ["unet-r34", "unet-mitb2"]', f"ARCHS = {archs}"))
    assert src != base_cfg or job == '"2d"'
    cells_s = [dict(c) for c in cells]
    cells_s[cfg_idx] = code(src)
    nb_s = dict(nb, cells=cells_s)
    (OUT.parent / name).write_text(json.dumps(nb_s, indent=1), encoding="utf-8")
    print(f"wrote {name}: JOB={job} ARCHS={archs}")
