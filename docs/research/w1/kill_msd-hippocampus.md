# Gate A kill test — msd-hippocampus

Protocol: tau=0.5, 8-connectivity, m=10 px @ working res, empty-GT excluded, thresholding sets, lambda grid 200, seeds=[1337].

m justification: Same rule as Kvasir for comparability; smoke test only.

| alpha | n_cal | multi-lesion frac | lambda_pix | R_pix | R_les(lambda_pix) | **G** | R_les test | lambda_les | lambda_union | infl pix | infl les | infl union |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 234 | 0.081 | 0.658 | 0.046 | 0.013 | **0.26** | 0.010 | 0.327 | 0.658 | 1.18 | 0.73 | 1.18 |
| 0.10 | 234 | 0.081 | 0.497 | 0.096 | 0.013 | **0.13** | 0.010 | 0.286 | 0.497 | 1.00 | 0.64 | 1.00 |
| 0.15 | 234 | 0.081 | 0.417 | 0.143 | 0.017 | **0.11** | 0.016 | 0.266 | 0.417 | 0.89 | 0.58 | 0.89 |

## K3 audit

**native** — units 780, excluded-empty 0, train/cal/test 390/234/156, multi-lesion 0.101, mean K 1.11, components 868, <1% area 0.099

size percentiles px: `{'p1': 0.9, 'p5': 1.2, 'p25': 225.7, 'p50': 261.3, 'p75': 308.3, 'p95': 382.3}`

| bin (px) | count |
|---|---|
| [0,10) | 86 |
| [10,50) | 1 |
| [50,200) | 43 |
| [200,1000) | 738 |
| [1000,5000) | 0 |
| [5000,20000) | 0 |
| [20000,100000) | 0 |
| [100000,inf) | 0 |

**256x256** — units 780, excluded-empty 0, train/cal/test 390/234/156, multi-lesion 0.101, mean K 1.11, components 868, <1% area 0.099

size percentiles px: `{'p1': 21.0, 'p5': 26.4, 'p25': 8354.2, 'p50': 9674.0, 'p75': 11430.0, 'p95': 14174.3}`

| bin (px) | count |
|---|---|
| [0,10) | 0 |
| [10,50) | 47 |
| [50,200) | 35 |
| [200,1000) | 4 |
| [1000,5000) | 2 |
| [5000,20000) | 780 |
| [20000,100000) | 0 |
| [100000,inf) | 0 |

**256x256** — units 780, excluded-empty 0, train/cal/test 390/234/156, multi-lesion 0.101, mean K 1.11, components 868, <1% area 0.099

size percentiles px: `{'p1': 21.0, 'p5': 26.4, 'p25': 8354.2, 'p50': 9674.0, 'p75': 11430.0, 'p95': 14174.3}`

| bin (px) | count |
|---|---|
| [0,10) | 0 |
| [10,50) | 47 |
| [50,200) | 35 |
| [200,1000) | 4 |
| [1000,5000) | 2 |
| [5000,20000) | 780 |
| [20000,100000) | 0 |
| [100000,inf) | 0 |
