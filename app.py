"""DoseFuse demo UI: CT-to-CT deformable registration, dose deformation, EQD2 accumulation.

    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import SimpleITK as sitk
import streamlit as st

from dosefuse import phantom, pipeline, viz, warp

st.set_page_config(page_title="DoseFuse", page_icon="🎯", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.2rem}
div[data-testid="stMetric"]{background:#1e2430;border-radius:10px;padding:10px 14px}
</style>""", unsafe_allow_html=True)

DEMO_DIR = Path("demo_data")
OUT_DIR = Path("output/app")


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Generating synthetic phantom (two CTs, two plans, structures)...")
def ensure_phantom():
    if not (DEMO_DIR / "phantom_meta.json").exists():
        phantom.generate(DEMO_DIR, log=lambda *_: None)
    return json.loads((DEMO_DIR / "phantom_meta.json").read_text())


@st.cache_resource(show_spinner="Importing DICOM (CT series, RTDOSE, RTSTRUCT)...")
def load_studies(ref_dir, mov_dir, fx_ref, fx_mov):
    ref = pipeline.load_study(ref_dir, "Reference (new CT)", fx_ref)
    mov = pipeline.load_study(mov_dir, "Moving (old CT)", fx_mov)
    return ref, mov


def display_arrays(res: pipeline.Result) -> dict:
    """Everything resampled to the reference CT grid as numpy arrays for display."""
    ct = res.reference.ct
    to_ct = lambda img: viz.to_array(warp.resample_to(img, ct))
    d = {
        "ct_fixed": viz.to_array(ct),
        "ct_rigid": viz.to_array(res.moving_ct_rigid),
        "ct_warped": viz.to_array(res.moving_ct_warped),
        "dose": {
            "Dose A deformed → B": to_ct(res.dose_moving_warped),
            "Dose A rigid-only → B": to_ct(res.dose_moving_rigid),
            "Dose B (reference)": to_ct(res.dose_reference),
            "Physical sum A+B": to_ct(res.dose_sum_physical),
            "EQD2 A (deformed)": to_ct(res.eqd2_moving),
            "EQD2 B": to_ct(res.eqd2_reference),
            "EQD2 sum A+B": to_ct(res.dose_sum_eqd2),
        },
        "structs": {k: viz.to_array(v) for k, v in res.reference.structures.items()},
        "structs_moving": {k: viz.to_array(v) for k, v in res.moving_structures_warped.items()},
        "structs_moving_rigid": {k: viz.to_array(warp.warp_mask(v, res.reg.rigid, ct))
                                 for k, v in res.moving.structures.items()},
        "dvf": viz.to_array(res.reg.dvf),
        "jac": viz.to_array(res.reg.jacobian),
        "spacing": ct.GetSpacing(),
    }
    if "ground_truth_dose" in res.validation:
        d["dose"]["Dose A ground-truth → B"] = to_ct(res.validation["ground_truth_dose"])
    return d


def show(fig):
    st.pyplot(fig, width='stretch')
    plt.close(fig)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎯 DoseFuse")
    st.caption("Deformable dose accumulation prototype v0.1")
    source = st.radio("Data source", ["Demo phantom", "DICOM folders"], horizontal=True)
    gt = None
    if source == "Demo phantom":
        meta = ensure_phantom()
        ref_dir, mov_dir = str(DEMO_DIR / "study_B"), str(DEMO_DIR / "study_A")
        fx_ref, fx_mov = meta["fractions_B"], meta["fractions_A"]
        gt = {"transform": str(DEMO_DIR / meta["ground_truth_transform"]),
              "landmarks_B_mm": meta["landmarks_B_mm"], "landmarks_A_mm": meta["landmarks_A_mm"]}
        st.info(f"Course A: {meta['prescription_A_Gy']} Gy / {fx_mov} fx (old CT)\n\n"
                f"Course B: {meta['prescription_B_Gy']} Gy / {fx_ref} fx (new CT, reference)\n\n"
                "The two CTs are linked by a known deformation, so registration error can be measured.")
    else:
        ref_dir = st.text_input("Reference study folder (new CT + RTDOSE + RTSTRUCT)", "")
        mov_dir = st.text_input("Moving study folder (old CT + RTDOSE + RTSTRUCT)", "")
        c1, c2 = st.columns(2)
        fx_ref = c1.number_input("Fractions (ref)", 1, 60, 5)
        fx_mov = c2.number_input("Fractions (moving)", 1, 60, 30)

    st.subheader("Registration")
    method = st.selectbox("Deformable method", ["bspline", "demons", "rigid"], format_func=lambda m: {
        "bspline": "B-spline (multi-resolution)", "demons": "Symmetric Demons", "rigid": "Rigid only"}[m])
    grid_mm = st.slider("B-spline control-point spacing (mm)", 15, 80, 40, 5)
    iters = st.slider("Deformable iterations (per level)", 5, 100, 20, 5)

    st.subheader("Radiobiology")
    default_ab = st.number_input("Default α/β (Gy)", 0.5, 20.0, 3.0, 0.5)
    ab_text = st.text_area("Per-structure α/β (NAME=value per line)",
                           "SPINAL_CORD=2\nHEART=3\nLUNG_L=3\nLUNG_R=3\nGTV=10\nGTV_RECURRENCE=10", height=150)
    ab = {}
    for line in ab_text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                ab[k.strip()] = float(v)
            except ValueError:
                pass

    run_btn = st.button("▶ Run registration + accumulation", type="primary", width='stretch')

    st.divider()
    st.subheader("Viewer")
    axis = st.radio("Plane", ["axial", "coronal", "sagittal"], horizontal=True)
    show_structs = st.checkbox("Show reference structures", True)
    show_mov_structs = st.checkbox("Show warped moving structures (dashed)", False)

# ----------------------------------------------------------------------------- run
if run_btn:
    if not ref_dir or not mov_dir:
        st.error("Enter both study folders.")
        st.stop()
    try:
        ref, mov = load_studies(ref_dir, mov_dir, int(fx_ref), int(fx_mov))
    except Exception as e:  # noqa: BLE001
        st.error(f"Import failed: {e}")
        st.stop()
    log_box = st.empty()
    lines = []

    def _log(m):
        lines.append(m)
        log_box.code("\n".join(lines))

    with st.spinner("Registering and accumulating..."):
        res = pipeline.run(ref, mov, method, grid_mm, iters, alpha_beta=ab, default_alpha_beta=default_ab,
                           log=_log, ground_truth=gt)
    st.session_state["res"] = res
    st.session_state["disp"] = display_arrays(res)
    log_box.empty()

if "res" not in st.session_state:
    st.title("Deformable dose accumulation")
    st.markdown("""
**Workflow:** import two DICOM studies → rigid + deformable CT-to-CT registration → warp the old RTDOSE
through the deformation onto the new plan's dose grid → voxel-wise BED/EQD2 → sum → DVH / OAR metrics →
export accumulated RTDOSE.

Press **Run** in the sidebar. With the demo phantom the whole pipeline takes a few seconds.
    """)
    st.stop()

res: pipeline.Result = st.session_state["res"]
D = st.session_state["disp"]
spacing = D["spacing"]
nz = viz.n_slices(D["ct_fixed"], axis)
with st.sidebar:
    idx = st.slider("Slice", 0, nz - 1, nz // 2)
structs = D["structs"] if show_structs else None
mov_structs = D["structs_moving"] if show_mov_structs else None

# ----------------------------------------------------------------------------- header metrics
v = res.validation
cols = st.columns(6)
cols[0].metric("Mean displacement", f"{res.dvf_stats['mean_displacement_mm']:.1f} mm")
cols[1].metric("Max displacement", f"{res.dvf_stats['max_displacement_mm']:.1f} mm")
cols[2].metric("Jacobian range", f"{res.dvf_stats['jacobian_min']:.2f}–{res.dvf_stats['jacobian_max']:.2f}")
cols[3].metric("Folding voxels", f"{100*res.dvf_stats['fraction_folding']:.2f} %")
if "tre_dir" in v:
    cols[4].metric("TRE (landmarks)", f"{v['tre_dir']['tre_mean_mm']:.2f} mm",
                   delta=f"{v['tre_dir']['tre_mean_mm']-v['tre_rigid']['tre_mean_mm']:+.2f} vs rigid", delta_color="inverse")
if "SPINAL_CORD" in res.structure_metrics:
    cord = res.structure_metrics["SPINAL_CORD"]["eqd2_sum"]
    cols[5].metric("Cord D0.1cc (EQD2 sum)", f"{cord.get('D0.1cc', 0):.1f} Gy")

tabs = st.tabs(["Fusion", "DVF & Jacobian", "Dose", "DVH & metrics", "DIR QA", "Export", "Log"])

# ----------------------------------------------------------------------------- fusion
with tabs[0]:
    c1, c2, c3 = st.columns([1, 1, 2])
    mode = c1.selectbox("Fusion mode", ["blend", "checker", "fixed", "moving"],
                        format_func=lambda m: {"blend": "Color blend (fixed=green, moving=magenta)",
                                               "checker": "Checkerboard", "fixed": "Fixed CT only",
                                               "moving": "Moving CT only"}[m])
    alpha = c2.slider("Moving opacity", 0.0, 1.0, 0.5, 0.05)
    c3.caption("Left: after rigid registration only. Right: after deformable registration. "
               "Grey = agreement; colored fringes = residual misalignment.")
    a, b = st.columns(2)
    with a:
        show(viz.fig_fusion(D["ct_fixed"], D["ct_rigid"], spacing, axis, idx, mode, alpha, structs,
                            D["structs_moving_rigid"] if show_mov_structs else None, "Rigid only"))
    with b:
        show(viz.fig_fusion(D["ct_fixed"], D["ct_warped"], spacing, axis, idx, mode, alpha, structs,
                            mov_structs, f"After DIR ({method})"))

# ----------------------------------------------------------------------------- dvf
with tabs[1]:
    a, b = st.columns(2)
    with a:
        show(viz.fig_dvf(D["ct_fixed"], D["dvf"], spacing, axis, idx, step=max(2, nz // 24)))
    with b:
        show(viz.fig_jacobian(D["ct_fixed"], D["jac"], spacing, axis, idx))
    st.dataframe(pd.DataFrame([res.dvf_stats]).T.rename(columns={0: "value (inside BODY)"}).style.format("{:.3f}"),
                 width='stretch')
    st.caption("TG-132: inspect the DVF for physical plausibility. det J ≤ 0 means folding (highlighted in green); "
               "values far from 1 flag unrealistic local volume change.")

# ----------------------------------------------------------------------------- dose
with tabs[2]:
    names = list(D["dose"].keys())
    c1, c2, c3 = st.columns([1, 1, 1])
    left = c1.selectbox("Left panel", names, index=0)
    right = c2.selectbox("Right panel", names, index=names.index("EQD2 sum A+B"))
    iso_txt = c3.text_input("Isodose lines (Gy, comma separated)", "10, 20, 30, 45, 60, 80, 100")
    iso = [float(x) for x in iso_txt.replace(";", ",").split(",") if x.strip()]
    shared = st.checkbox("Shared color scale", True)
    dmax_l, dmax_r = float(D["dose"][left].max()), float(D["dose"][right].max())
    dmax = max(dmax_l, dmax_r) if shared else None
    a, b = st.columns(2)
    with a:
        show(viz.fig_dose(D["ct_fixed"], D["dose"][left], spacing, axis, idx, dmax or dmax_l, structs, left, iso))
    with b:
        show(viz.fig_dose(D["ct_fixed"], D["dose"][right], spacing, axis, idx, dmax or dmax_r, structs, right, iso))
    st.caption("All doses are shown on the reference CT. The accumulation grid is the reference RTDOSE grid; "
               "the deformed dose is sampled as D_A(T(x)) with trilinear interpolation (no Jacobian correction).")

# ----------------------------------------------------------------------------- dvh
with tabs[3]:
    snames = list(res.structure_metrics.keys())
    default_sel = [s for s in ["GTV", "GTV_RECURRENCE", "SPINAL_CORD", "HEART", "LUNG_R"] if s in snames] or snames[:3]
    sel = st.multiselect("Structures", snames, default_sel)
    series = st.multiselect("Dose series", ["moving_warped", "reference", "physical_sum", "eqd2_sum"],
                            ["moving_warped", "reference", "eqd2_sum"])
    dash = {"moving_warped": "dot", "reference": "dash", "physical_sum": "dashdot", "eqd2_sum": "solid"}
    label = {"moving_warped": "A deformed", "reference": "B", "physical_sum": "A+B physical", "eqd2_sum": "A+B EQD2"}
    palette = ["#ff7043", "#ab47bc", "#ffee58", "#ef5350", "#4fc3f7", "#66bb6a", "#9e9e9e", "#29b6f6"]
    fig = go.Figure()
    for i, s in enumerate(sel):
        for ser in series:
            c = res.dvh_curves[s][ser]
            fig.add_trace(go.Scatter(x=c["dose"], y=c["volume_pct"], mode="lines", name=f"{s} · {label[ser]}",
                                     line=dict(color=palette[i % len(palette)], dash=dash[ser], width=2)))
    fig.update_layout(xaxis_title="Dose (Gy)", yaxis_title="Volume (%)", height=430, margin=dict(l=40, r=20, t=30, b=40),
                      legend=dict(orientation="h", y=-0.25), template="plotly_dark")
    st.plotly_chart(fig, width='stretch')

    rows = []
    for s in snames:
        for ser in ["moving_warped", "reference", "physical_sum", "eqd2_sum"]:
            m = res.structure_metrics[s][ser]
            rows.append({"structure": s, "series": label[ser], "vol cc": m.get("volume_cc"), "Dmax": m.get("Dmax"),
                         "Dmean": m.get("Dmean"), "D0.03cc": m.get("D0.03cc"), "D0.1cc": m.get("D0.1cc"),
                         "D1cc": m.get("D1cc"), "D2cc": m.get("D2cc"), "D95%": m.get("D95%"), "D2%": m.get("D2%")})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({c: "{:.1f}" for c in df.columns if c not in ("structure", "series")}),
                 width='stretch', height=420)

    st.markdown("**Constraint check (cumulative EQD2)**")
    c1, c2, c3 = st.columns(3)
    c_struct = c1.selectbox("OAR", snames, index=snames.index("SPINAL_CORD") if "SPINAL_CORD" in snames else 0)
    c_metric = c2.selectbox("Metric", ["D0.03cc", "D0.1cc", "D1cc", "D2cc", "Dmax", "Dmean"], index=1)
    c_limit = c3.number_input("Limit (Gy EQD2)", 0.0, 200.0, 60.0, 1.0)
    val = res.structure_metrics[c_struct]["eqd2_sum"].get(c_metric, float("nan"))
    (st.success if val <= c_limit else st.error)(f"{c_struct} {c_metric} = {val:.1f} Gy EQD2 "
                                                  f"({'PASS' if val <= c_limit else 'FAIL'}, limit {c_limit:.0f} Gy)")

# ----------------------------------------------------------------------------- QA
with tabs[4]:
    st.markdown("**Structure agreement after registration** (moving structures warped onto the reference CT vs. "
                "the reference structures). TG-132 tolerance: Dice > 0.8–0.9, mean surface distance < 2–3 mm.")
    if v.get("structures"):
        qa = pd.DataFrame(v["structures"]).T
        st.dataframe(qa.style.format("{:.3f}"), width='stretch')
    if "tre_dir" in v:
        st.markdown("**Landmark TRE vs. known ground-truth deformation** (phantom only)")
        tre = pd.DataFrame({"landmark": range(1, len(v["tre_dir"]["tre_per_point"]) + 1),
                            "rigid (mm)": v["tre_rigid"]["tre_per_point"], "DIR (mm)": v["tre_dir"]["tre_per_point"]})
        st.dataframe(tre.style.format({"rigid (mm)": "{:.2f}", "DIR (mm)": "{:.2f}"}), width='stretch')
        st.markdown("**Deformed dose vs. dose mapped with the true deformation** (inside BODY)")
        st.dataframe(pd.DataFrame(v["dose_vs_truth"]).T.style.format("{:.2f}"), width='stretch')
        a, b = st.columns(2)
        diff_r = D["dose"]["Dose A rigid-only → B"] - D["dose"]["Dose A ground-truth → B"]
        diff_d = D["dose"]["Dose A deformed → B"] - D["dose"]["Dose A ground-truth → B"]
        lim = float(max(np.abs(diff_r).max(), 1e-3))
        for col, arr, t in ((a, diff_r, "Rigid − truth (Gy)"), (b, diff_d, "DIR − truth (Gy)")):
            with col:
                c = viz.window(viz.take_slice(D["ct_fixed"], axis, idx))
                f2, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
                ax.imshow(c, cmap="gray", vmin=0, vmax=1, aspect=viz.aspect(spacing, axis))
                im = ax.imshow(viz.take_slice(arr, axis, idx), cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.7,
                               aspect=viz.aspect(spacing, axis))
                ax.set_title(t, fontsize=10)
                ax.axis("off")
                f2.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
                f2.tight_layout(pad=0.2)
                show(f2)

# ----------------------------------------------------------------------------- export
with tabs[5]:
    st.markdown("Export the deformed dose and the accumulated doses as DICOM RTDOSE (importable into Eclipse / "
                "Velocity / MIM for comparison), plus the DVF, Jacobian and transform.")
    out_dir = st.text_input("Output folder", str(OUT_DIR))
    if st.button("Export"):
        paths = pipeline.export(res, out_dir)
        st.success("Exported:")
        st.json(paths)
        for key in ("rtdose_eqd2_sum", "rtdose_physical_sum", "rtdose_moving_warped"):
            p = Path(paths[key])
            st.download_button(f"Download {p.name}", p.read_bytes(), file_name=p.name, mime="application/dicom")

with tabs[6]:
    st.code("\n".join(res.log))
