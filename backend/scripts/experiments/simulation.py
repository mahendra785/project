"""E7 — controlled simulation of the lesion-equalization mechanism.

Generator and decision rule: docs/research/preregistration-mgcrc.md, Amendment 2.
Writes posteriors in the train_backbone.py format (dataset "sim-g<γ>-b<b>",
backbone "sim"), then runs the unchanged mgcrc.py protocol on each cell.

    python scripts/experiments/simulation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mgcrc  # noqa: E402

from app.calibration.losses import gt_components, lambda_grid  # noqa: E402

SIZE = 128
N_UNITS = 600
GAMMAS = (0.0, 0.5, 1.0, 1.5, 2.0)
BACKGROUNDS = (0.05, 0.15)
REPS = 500
A0 = 0.9
R_MIN, R_MAX = 2.0, 12.0


def make_unit(rng, gamma, b):
    yy, xx = np.mgrid[:SIZE, :SIZE]
    p = rng.uniform(0, b, (SIZE, SIZE))
    gt = np.zeros((SIZE, SIZE), bool)
    for _ in range(1 + rng.poisson(1.5)):
        cy, cx = rng.uniform(R_MAX, SIZE - R_MAX, 2)
        r = rng.uniform(R_MIN, R_MAX)
        a = np.clip(A0 * (r / R_MAX) ** gamma + rng.normal(0, 0.05), 0.05, 1.0)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        p = np.maximum(p, a * np.exp(-d2 / (2 * r**2)))
        gt |= d2 <= r**2
    for _ in range(rng.poisson(3)):  # false candidates
        cy, cx = rng.uniform(0, SIZE, 2)
        r = rng.uniform(R_MIN, R_MAX / 2)
        a = rng.uniform(0.1, 0.35)
        p = np.maximum(p, a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2)))
    return np.clip(p, 0, 1), gt_components(gt)


def main():
    out = []
    for gamma in GAMMAS:
        for b in BACKGROUNDS:
            ds = f"sim-g{gamma:g}-b{b:g}"
            rng = np.random.default_rng(int(gamma * 100 + b * 1000))
            units = [make_unit(rng, gamma, b) for _ in range(N_UNITS)]
            path = mgcrc.POST_ROOT / ds
            path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path / "sim_seed0.npz", prob=np.stack([u[0] for u in units]).astype(np.float16),
                                labels=np.stack([u[1] for u in units]).astype(np.uint8),
                                valid=np.ones((N_UNITS, SIZE, SIZE), bool), ids=np.arange(N_UNITS),
                                fold=np.zeros(N_UNITS))
            rows, gap_rows, chosen, meta = mgcrc.run(ds, 0, "sim", REPS, 0.5, mgcrc.ALPHAS, lambda_grid())
            summary = mgcrc.summarise(rows, gap_rows, chosen, meta)
            summary["meta"].update(gamma=gamma, background=b)
            mgcrc.OUT_ROOT.mkdir(parents=True, exist_ok=True)
            (mgcrc.OUT_ROOT / f"sim_{ds}_sim_seed0_tau0.5.json").write_text(json.dumps(summary, indent=2))
            for p in summary["paired_vs_union_threshold"]:
                if p["method"] == "ours_union":
                    out.append(dict(gamma=gamma, background=b, alpha=p["alpha"], rel=p["rel_change_mean"],
                                    lo=p["lo"], hi=p["hi"], p_smaller=p["p_smaller"]))
                    print(f"g={gamma:<4} b={b:<5} a={p['alpha']:.2f}  ours vs joint-threshold "
                          f"{100*p['rel_change_mean']:+6.1f}% [{100*p['lo']:+.1f},{100*p['hi']:+.1f}] "
                          f"P(smaller)={p['p_smaller']:.3f}", flush=True)
    # pre-registered decision rule (Amendment 2)
    at = {(r["gamma"], r["background"], r["alpha"]): r for r in out}
    strong = all(r["rel"] < 0 and r["p_smaller"] >= 0.95 for r in out if r["gamma"] >= 1.0)
    larger = all(at[(2.0, b, a)]["rel"] < at[(0.0, b, a)]["rel"] for b in BACKGROUNDS for a in mgcrc.ALPHAS)
    verdict = "SUPPORTED" if strong and larger else "FAILED"
    (mgcrc.OUT_ROOT / "simulation_E7.json").write_text(json.dumps(dict(verdict=verdict, rows=out), indent=2))
    with open(mgcrc.REPO_ROOT / "paper" / "data" / "sim_E7.csv", "w") as f:
        f.write("gamma,background,alpha,rel,lo,hi,p_smaller\n")
        for r in out:
            f.write(f"{r['gamma']},{r['background']},{r['alpha']},{r['rel']:.4f},{r['lo']:.4f},{r['hi']:.4f},{r['p_smaller']:.3f}\n")
    print("E7 verdict:", verdict)


if __name__ == "__main__":
    main()
