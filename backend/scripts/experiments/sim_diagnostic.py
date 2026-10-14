"""POST-HOC (not pre-registered): why E7's gain vanishes at large gamma.

For each simulation cell: (a) fraction of GT lesions whose peak posterior lies
below the smallest candidate level t0 = 0.05; (b) fraction of SMALL lesions
(r < 5 px) whose peak does not exceed the 99th percentile of that image's
background posterior. A lesion in (b) cannot be separated from background by
any transform of p alone, so no nested family can hit it without admitting
background (Proposition 3's precondition fails).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.core.config import BACKEND_ROOT, REPO_ROOT  # noqa: E402

rows = []
for d in sorted((BACKEND_ROOT / "var" / "posteriors").glob("sim-*")):
    z = np.load(d / "sim_seed0.npz")
    below, small_below, n, n_small, small_bg = 0, 0, 0, 0, 0
    for p, lab in zip(z["prob"].astype(np.float32), z["labels"]):
        k = int(lab.max())
        if not k:
            continue
        peaks = ndimage.maximum(p, lab, index=np.arange(1, k + 1))
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, k + 1))
        below += int((peaks < 0.05).sum()); n += k
        sm = sizes < np.pi * 5**2
        small_below += int((peaks[sm] < 0.05).sum()); n_small += int(sm.sum())
        bg99 = np.percentile(p[lab == 0], 99)
        small_bg += int((peaks[sm] <= bg99).sum())
    _, g, b = d.name.split("-")
    rows.append(dict(gamma=float(g[1:]), background=float(b[1:]), frac_lesions_below_t0=below / n,
                     frac_small_lesions_below_t0=small_below / max(n_small, 1),
                     frac_small_lesions_not_above_background_p99=small_bg / max(n_small, 1)))
    print(rows[-1])
(REPO_ROOT / "docs" / "research" / "results" / "simulation_E7_posthoc_diagnostic.json").write_text(
    json.dumps(dict(label="POST-HOC, not pre-registered", rows=rows), indent=2))
