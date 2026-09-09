# Gate A kill test — kvasir-seg

Protocol: tau=0.5, 8-connectivity, m=10 px @ working res, empty-GT excluded, thresholding sets, lambda grid 200, seeds=[7, 1337, 2024].

m justification: Native-res histogram is bimodal: 143 components <10 px (JPEG mask speckle), zero in [10,200), real polyps from 200 px. At 256x256 the smallest real component is 309 px (p1). m=10 removes only the residual artefact; any m in [10,200) is equivalent. Chosen before the gap test.

| alpha | n_cal | multi-lesion frac | lambda_pix | R_pix | R_les(lambda_pix) | **G** | R_les test | lambda_les | lambda_union | infl pix | infl les | infl union |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 300 | 0.048 ± 0.007 | 0.761 ± 0.027 | 0.046 ± 0.001 | 0.021 ± 0.004 | **0.41 ± 0.079** | 0.016 ± 0.003 | 0.646 ± 0.027 | 0.761 ± 0.027 | 1.58 ± 0.034 | 1.33 ± 0.050 | 1.58 ± 0.034 |
| 0.10 | 300 | 0.048 ± 0.007 | 0.645 ± 0.022 | 0.095 ± 0.000 | 0.046 ± 0.007 | **0.47 ± 0.066** | 0.062 ± 0.017 | 0.548 ± 0.023 | 0.645 ± 0.022 | 1.32 ± 0.033 | 1.10 ± 0.045 | 1.32 ± 0.033 |
| 0.15 | 300 | 0.048 ± 0.007 | 0.563 ± 0.016 | 0.146 ± 0.001 | 0.080 ± 0.008 | **0.54 ± 0.053** | 0.094 ± 0.010 | 0.492 ± 0.011 | 0.563 ± 0.016 | 1.14 ± 0.029 | 0.98 ± 0.026 | 1.14 ± 0.029 |

## K3 audit

**native** — units 1000, excluded-empty 0, train/cal/test 500/300/200, multi-lesion 0.180, mean K 1.21, components 1206, <1% area 0.144

size percentiles px: `{'p1': 1.0, 'p5': 1.1, 'p25': 12139.6, 'p50': 31320.7, 'p75': 61929.9, 'p95': 134453.5}`

| bin (px) | count |
|---|---|
| [0,10) | 143 |
| [10,50) | 0 |
| [50,200) | 0 |
| [200,1000) | 5 |
| [1000,5000) | 45 |
| [5000,20000) | 250 |
| [20000,100000) | 617 |
| [100000,inf) | 146 |

**256x256** — units 1000, excluded-empty 0, train/cal/test 500/300/200, multi-lesion 0.049, mean K 1.06, components 1064, <1% area 0.031

size percentiles px: `{'p1': 308.8, 'p5': 983.5, 'p25': 3344.2, 'p50': 6787.5, 'p75': 13047.8, 'p95': 26199.5}`

| bin (px) | count |
|---|---|
| [0,10) | 1 |
| [10,50) | 0 |
| [50,200) | 6 |
| [200,1000) | 48 |
| [1000,5000) | 344 |
| [5000,20000) | 540 |
| [20000,100000) | 125 |
| [100000,inf) | 0 |

**256x256** — units 1000, excluded-empty 0, train/cal/test 500/300/200, multi-lesion 0.048, mean K 1.06, components 1063, <1% area 0.030

size percentiles px: `{'p1': 327.6, 'p5': 992.3, 'p25': 3347.5, 'p50': 6801.0, 'p75': 13061.5, 'p95': 26203.0}`

| bin (px) | count |
|---|---|
| [0,10) | 0 |
| [10,50) | 0 |
| [50,200) | 6 |
| [200,1000) | 48 |
| [1000,5000) | 344 |
| [5000,20000) | 540 |
| [20000,100000) | 125 |
| [100000,inf) | 0 |
