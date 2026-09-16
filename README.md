# DoseFuse — deformable dose accumulation prototype

CT-to-CT deformable image registration (DIR), dose deformation, physical and EQD2
dose summation, DVH/OAR metrics and DICOM RTDOSE export. A research prototype of the
Velocity / MIM "dose accumulation for reirradiation" workflow, built on SimpleITK + pydicom.

```
CT_A + Dose_A ──► rigid ──► B-spline DIR ──► T: B→A ──► Dose_A(T(x)) on Dose_B grid
                                                            │
Dose_B ─────────────────────────────────────────────────────┤
                                                            ▼
                         voxel-wise BED / EQD2 ──► accumulated dose ──► DVH, D0.1cc, ... ──► RTDOSE
```

## Live demo

https://dosefuse.vercel.app (single demo account; credentials are Vercel env vars `DOSEFUSE_USER`,
`DOSEFUSE_PASSWORD`, `DOSEFUSE_SECRET`). The site is a static viewer (`public/index.html`) talking to a
FastAPI function (`api/index.py`) that runs the same `dosefuse` pipeline on the synthetic phantom.

Vercel functions have 2 vCPUs, so the default and most common settings are precomputed into
`cache/*.npz` (regenerate with `python -c "from dosefuse import session; ..."` see `session.save_cache`);
other settings are registered live (roughly 10-50 s). Do not add a `vercel.json` rewrite for `/api`:
the FastAPI preset already routes every path to the app, and a rewrite replaces the request path.

Deploy: `vercel deploy --prod`. Local API: `.venv/bin/uvicorn api.index:app --port 8000` with
`DOSEFUSE_USER` / `DOSEFUSE_PASSWORD` set.

## Quick start (local, full Streamlit UI)

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/streamlit run app.py          # interactive demo (synthetic phantom is generated on first run)
.venv/bin/python run_demo.py            # headless: runs the pipeline, prints metrics, exports to output/demo
```

Run on real data (two exported Eclipse studies, each folder holding the CT slices, RTDOSE and RTSTRUCT):

```bash
.venv/bin/python run_demo.py --ref /path/to/reirradiation_study --mov /path/to/original_study --fx-ref 5 --fx-mov 30
```

or choose **DICOM folders** in the app sidebar.

## What is implemented (v0.1)

| Step | Module | Notes |
|---|---|---|
| DICOM import: CT series, RTDOSE (DoseGridScaling, GridFrameOffsetVector), RTSTRUCT rasterisation | `dosefuse/dicom_io.py` | LPS patient coordinates preserved |
| Rigid registration (Euler 3D, Mattes MI, multi-resolution) | `dosefuse/registration.py` | |
| Deformable registration: multi-resolution cubic B-spline (LBFGS-B) or symmetric Demons | `dosefuse/registration.py` | control-point spacing is a parameter |
| Deformation vector field + Jacobian determinant (folding / volume-change QA) | `dosefuse/registration.py` | |
| Dose warping onto the reference RTDOSE grid, trilinear | `dosefuse/warp.py` | scalar mapping, no Jacobian correction |
| Physical sum, voxel-wise BED / EQD2 with per-structure α/β map | `dosefuse/accumulate.py` | |
| Cumulative DVH, Dmax/Dmean/D0.03cc/D0.1cc/D1cc/D2cc/D%/V Gy | `dosefuse/dvh.py` | |
| DIR QA: Dice, Hausdorff, landmark TRE, dose-vs-truth (TG-132 style) | `dosefuse/metrics.py` | |
| Export: RTDOSE (physical sum, EQD2 sum, deformed dose), DVF, Jacobian, transform | `dosefuse/pipeline.py` | |
| Synthetic thorax phantom with known ground-truth deformation | `dosefuse/phantom.py` | 60 Gy/30 fx + 30 Gy/5 fx paraspinal recurrence |
| Streamlit UI: fusion (blend / checkerboard), DVF, Jacobian, isodoses, DVH, constraints, export | `app.py` | |

## Conventions

* SimpleITK transforms map **fixed (reference, new CT) → moving (old CT)**. The deformed dose at
  reference voxel *x* is `D_A(T(x))`.
* The accumulation grid is the reference plan's RTDOSE grid, so the new plan's dose is untouched.
* EQD2 per voxel: `BED = D (1 + d/(α/β))`, `d = D / n`, `EQD2 = BED / (1 + 2/(α/β))`.
  α/β is a per-voxel map built from the reference structures (later entries override earlier ones).

## Validation against Velocity (planned)

Export the same two studies from Eclipse, run DoseFuse and Velocity, then compare:
Dice / Hausdorff of propagated contours, landmark TRE, Jacobian statistics, voxel-wise dose
difference and DVH parameters (D0.1cc, D2cc, Dmean) of the accumulated dose. Export
`RTDOSE_sum_EQD2.dcm` back into Eclipse for side-by-side review.

## Limitations of v0.1

* No dose-mapping uncertainty (e.g. TG-132 recommends reporting DIR uncertainty per structure).
* No energy/mass-conserving dose mapping; scalar interpolation only.
* Rasterised structures are slice-matched by nearest slice (no oblique contour handling).
* Not a medical device. Research use only.
