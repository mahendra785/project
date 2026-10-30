# Running the PUMS revision experiments on Kaggle

These notebooks produced every reported result of the paper: frozen U-Net/ResNet-34 and U-Net/MiT-B2 backbones on Kvasir-SEG, FracAtlas and MSD Task08 liver tumours, followed by calibration, the clustered bootstrap and aggregation. Everything runs on a free Kaggle GPU. The notebooks download the data themselves, so you upload only the notebook file.

## Files

| Session | Notebook file | What it runs | Time |
|---|---|---|---|
| 1 | `pums_session1_2d.ipynb` | Kvasir-SEG + FracAtlas, 2 backbones × 3 seeds | ~4–6 h |
| 2 | `pums_session2_task08_unet-r34.ipynb` | MSD Task08 liver tumours, U-Net/ResNet-34, 3 seeds | ~5–7 h |
| 3 | `pums_session3_task08_unet-mitb2.ipynb` | MSD Task08, U-Net/MiT-B2, 3 seeds | ~6–8 h |

The three sessions are independent and can run at the same time, on one account or on several.

## One-time setup

1. Create an account at **kaggle.com**.
2. Verify your phone number: profile picture → **Settings** → **Phone verification**. Without it, the GPU and Internet options stay locked.

## Running a session (same steps for each file)

1. Click **Create → New Notebook**.
2. Choose **File → Import Notebook** and upload the session's `.ipynb` file. **Do not edit any cell.** The settings are already in the file.
3. In the right-hand **Settings** panel, set:
   - **Accelerator:** GPU T4 x2
   - **Internet:** On
4. Click **Save Version** (top right) → choose **Save & Run All (Commit)** → **Save**.
   - **Do not** press **Run All** in the editor. That runs interactively and stops when you close the browser.
   - If a second, interactive session starts as well, stop it (⋯ → Stop session). Only the **Version** run matters.
5. After about 15 minutes, open **Your Work → Code → the notebook → the version → Logs** and check:
   - Session 1: `kvasir-seg.zip ... verified`, `fracatlas.zip ... verified`, then a line ending `OK` for each smoke test.
   - Sessions 2 and 3: `task08.tar ... verified`, then `msd-hepatic-vessel unet-... OK`.
   - Lines like "Debugger warning" and "retrying without TLS verification" are normal.
6. If all lines read **OK**, you can close the browser or switch off the laptop. The run continues on Kaggle's servers for up to 12 hours, and Kaggle emails you when it finishes.

## When it finishes

1. Open the version → **Output** tab and download `revision_results_<...>.zip` (a few tens of MB).
2. Unzip it into `docs/research/` of this repository (each archive contains `results_v2/`), then run `backend/scripts/experiments/aggregate_v2.py` and `paper/journal/build.py`.

## If something fails

- **Any line says `FAILED`, or the run stops with a red error:** download the log (**Download Logs**) and open an issue in this repository with the log attached. Don't change the code.
- **Session 3 smoke test fails for `unet-mitb2`:** change only `ARCHS = ["unet-mitb2"]` to `ARCHS = ["fpn-effb3"]` in the second cell, then save a new version. This is the fallback backbone named in the pre-registration.
- **"GPU quota exceeded" or "too many sessions":** wait for a running session to finish, or run the file from another account. The free quota is 30 GPU hours per week per account.

## Rules (these keep the results valid for the paper)

- Do not change seeds, epochs, splits or any code. The protocol is pre-registered (Amendment 3, committed before any result).
- Do not re-run a session to "get better numbers". Every completed run is reported.
- Keep the downloaded zips unchanged.
