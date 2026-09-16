"""A computed demo session: pipeline result + numpy arrays on the reference CT grid, ready to render."""
from __future__ import annotations

import io
import json
import os
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from . import phantom, pipeline, viz, warp

_PHANTOM = {}
_SESSIONS: dict[tuple, "Session"] = {}
_LOCK = threading.Lock()
TIMINGS: list[str] = []

DOSE_SERIES = ["Dose A deformed → B", "Dose A rigid-only → B", "Dose B (reference)", "Physical sum A+B",
               "EQD2 A (deformed)", "EQD2 B", "EQD2 sum A+B", "Dose A ground-truth → B"]


def phantom_studies():
    if "ph" not in _PHANTOM:
        ph = phantom.build()
        m = ph["meta"]
        ref = pipeline.Study("Reference (new CT, course B)", ph["ct_b"], ph["dose_b"], ph["masks_b"], m["fractions_B"])
        mov = pipeline.Study("Moving (old CT, course A)", ph["ct_a"], ph["dose_a"], ph["masks_a"], m["fractions_A"])
        gt = {"transform": ph["T_gt"], "landmarks_B_mm": m["landmarks_B_mm"], "landmarks_A_mm": m["landmarks_A_mm"]}
        _PHANTOM["ph"] = (ref, mov, gt, m)
    return _PHANTOM["ph"]


class Session:
    def __init__(self, method="bspline", grid_mm=40.0, iters=20):
        ref, mov, gt, meta = phantom_studies()
        t0 = time.time()
        self.params = {"method": method, "grid_mm": grid_mm, "iters": iters}
        self.meta = meta
        self.res = pipeline.run(ref, mov, method, grid_mm, iters, log=lambda *_: None, ground_truth=gt)
        self.seconds = time.time() - t0
        ct = ref.ct
        to_ct = lambda img: viz.to_array(warp.resample_to(img, ct))
        r = self.res
        self.ct_fixed = viz.to_array(ct)
        self.ct_rigid = viz.to_array(r.moving_ct_rigid)
        self.ct_warped = viz.to_array(r.moving_ct_warped)
        self.dose = {
            "Dose A deformed → B": to_ct(r.dose_moving_warped),
            "Dose A rigid-only → B": to_ct(r.dose_moving_rigid),
            "Dose B (reference)": to_ct(r.dose_reference),
            "Physical sum A+B": to_ct(r.dose_sum_physical),
            "EQD2 A (deformed)": to_ct(r.eqd2_moving),
            "EQD2 B": to_ct(r.eqd2_reference),
            "EQD2 sum A+B": to_ct(r.dose_sum_eqd2),
            "Dose A ground-truth → B": to_ct(r.validation["ground_truth_dose"]),
        }
        self.structs = {k: viz.to_array(v) for k, v in ref.structures.items()}
        self.structs_moving = {k: viz.to_array(v) for k, v in r.moving_structures_warped.items()}
        self.structs_moving_rigid = {k: viz.to_array(warp.warp_mask(v, r.reg.rigid, ct)) for k, v in mov.structures.items()}
        self.dvf = viz.to_array(r.reg.dvf)
        self.jac = viz.to_array(r.reg.jacobian)
        self.spacing = ct.GetSpacing()

    # ------------------------------------------------------------------ JSON summary
    def summary(self) -> dict:
        if getattr(self, "_summary", None) is not None:
            return self._summary
        r = self.res
        v = r.validation
        curves = {}
        for s, d in r.dvh_curves.items():
            curves[s] = {}
            for ser, c in d.items():
                x, y = np.asarray(c["dose"]), np.asarray(c["volume_pct"])
                step = max(1, len(x) // 250)
                curves[s][ser] = {"dose": x[::step].round(2).tolist(), "volume_pct": y[::step].round(2).tolist()}
        return {
            "params": self.params, "seconds": round(self.seconds, 2), "phantom": self.meta,
            "shape": {"axial": int(self.ct_fixed.shape[0]), "coronal": int(self.ct_fixed.shape[1]),
                      "sagittal": int(self.ct_fixed.shape[2])},
            "dvf_stats": r.dvf_stats,
            "structures": list(self.structs.keys()),
            "dose_series": [k for k in DOSE_SERIES if k in self.dose],
            "series_max": {k: float(v.max()) for k, v in self.dose.items()},
            "structure_metrics": r.structure_metrics,
            "dvh": curves,
            "validation": {
                "structures": v.get("structures", {}),
                "tre_rigid": _tre(v.get("tre_rigid")), "tre_dir": _tre(v.get("tre_dir")),
                "dose_vs_truth": v.get("dose_vs_truth", {}),
            },
            "log": r.log,
        }

    # ------------------------------------------------------------------ rendering
    def render(self, kind: str, axis="axial", idx=None, mode="blend", alpha=0.5, series=None,
               iso=(10, 20, 30, 45, 60, 80, 100), structs=True, mov_structs=False, dmax=None) -> bytes:
        n = viz.n_slices(self.ct_fixed, axis)
        idx = n // 2 if idx is None else int(np.clip(idx, 0, n - 1))
        st = self.structs if structs else None
        sp = self.spacing
        if kind == "fusion_rigid":
            fig = viz.fig_fusion(self.ct_fixed, self.ct_rigid, sp, axis, idx, mode, alpha, st,
                                 self.structs_moving_rigid if mov_structs else None, "Rigid only")
        elif kind == "fusion_dir":
            fig = viz.fig_fusion(self.ct_fixed, self.ct_warped, sp, axis, idx, mode, alpha, st,
                                 self.structs_moving if mov_structs else None, f"After DIR ({self.params['method']})")
        elif kind == "dvf":
            fig = viz.fig_dvf(self.ct_fixed, self.dvf, sp, axis, idx, step=max(2, n // 24))
        elif kind == "jac":
            fig = viz.fig_jacobian(self.ct_fixed, self.jac, sp, axis, idx)
        elif kind == "dose":
            series = series if series in self.dose else "EQD2 sum A+B"
            arr = self.dose[series]
            fig = viz.fig_dose(self.ct_fixed, arr, sp, axis, idx, dmax or float(arr.max()), st, series, list(iso))
        elif kind in ("diff_rigid", "diff_dir"):
            src = "Dose A rigid-only → B" if kind == "diff_rigid" else "Dose A deformed → B"
            diff = self.dose[src] - self.dose["Dose A ground-truth → B"]
            lim = float(max(np.abs(self.dose["Dose A rigid-only → B"] - self.dose["Dose A ground-truth → B"]).max(), 1e-3))
            c = viz.window(viz.take_slice(self.ct_fixed, axis, idx))
            fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
            ax.imshow(c, cmap="gray", vmin=0, vmax=1, aspect=viz.aspect(sp, axis))
            im = ax.imshow(viz.take_slice(diff, axis, idx), cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.7,
                           aspect=viz.aspect(sp, axis))
            ax.set_title("Rigid − truth (Gy)" if kind == "diff_rigid" else "DIR − truth (Gy)", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout(pad=0.2)
        else:
            raise ValueError(kind)
        fig.patch.set_facecolor("#ffffff")
        for ax in fig.axes:
            ax.title.set_color("#1c638a")
            ax.title.set_fontweight("bold")
            ax.tick_params(colors="#303334")
            if ax.yaxis.label:
                ax.yaxis.label.set_color("#303334")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()

    def export_rtdose(self, which: str) -> bytes:
        from . import dicom_io
        import tempfile, os
        if getattr(self, "_exports", None):
            img = self._exports[which]
        else:
            img = {"eqd2": self.res.dose_sum_eqd2, "physical": self.res.dose_sum_physical,
                   "deformed": self.res.dose_moving_warped}[which]
        desc = {"eqd2": ("DoseFuse EQD2 sum", "EFFECTIVE"), "physical": ("DoseFuse physical sum", "PHYSICAL"),
                "deformed": ("DoseFuse deformed dose A", "PHYSICAL")}[which]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "dose.dcm")
            dicom_io.write_rtdose(img, p, series_desc=desc[0], dose_type=desc[1])
            return open(p, "rb").read()


def _tre(t):
    if not t:
        return None
    return {"mean": float(t["tre_mean_mm"]), "max": float(t["tre_max_mm"]),
            "per_point": [float(x) for x in t["tre_per_point"]]}


def get_session(method="bspline", grid_mm=40.0, iters=20) -> Session:
    method = method if method in ("bspline", "demons", "rigid") else "bspline"
    grid_mm = float(np.clip(grid_mm, 15, 80))
    iters = int(np.clip(iters, 5, 100))
    key = (method, grid_mm, iters)
    with _LOCK:
        if key not in _SESSIONS:
            if len(_SESSIONS) > 6:
                _SESSIONS.pop(next(iter(_SESSIONS)))
            t = time.time()
            cached = load_cached(key)
            TIMINGS.append(f"load_cached {key}: {time.time()-t:.2f}s -> {'hit' if cached else 'miss'}")
            if not cached:
                t = time.time()
                cached = Session(method, grid_mm, iters)
                TIMINGS.append(f"compute {key}: {time.time()-t:.2f}s")
            _SESSIONS[key] = cached
        return _SESSIONS[key]


# ----------------------------------------------------------------------------- precomputed cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
DEFAULT_KEY = ("bspline", 40.0, 20)


def _cache_path(key):
    return os.path.join(CACHE_DIR, f"{key[0]}_{key[1]:g}_{key[2]}.npz")


def save_cache(s: "Session"):
    """Serialise everything the API needs (arrays + summary + export volumes) to a compressed npz."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = (s.params["method"], s.params["grid_mm"], s.params["iters"])
    r = s.res
    exports = {"eqd2": r.dose_sum_eqd2, "physical": r.dose_sum_physical, "deformed": r.dose_moving_warped}
    # display arrays are quantised (CT -> int16 HU, dose/DVF/Jacobian -> float16) to keep the bundle small;
    # the exported RTDOSE volumes stay float32.
    q = lambda a: np.clip(np.round(a), -32768, 32767).astype(np.int16)
    payload = {"summary": json.dumps(s.summary(), default=pipeline._jsonable), "spacing": np.array(s.spacing),
               "ct_fixed": q(s.ct_fixed), "ct_rigid": q(s.ct_rigid), "ct_warped": q(s.ct_warped),
               "dvf": s.dvf.astype(np.float16), "jac": s.jac.astype(np.float16),
               "dose_names": np.array(list(s.dose.keys())), "struct_names": np.array(list(s.structs.keys())),
               "structm_names": np.array(list(s.structs_moving.keys())),
               "structr_names": np.array(list(s.structs_moving_rigid.keys()))}
    for i, (k, v) in enumerate(s.dose.items()):
        payload[f"dose_{i}"] = v.astype(np.float16)
    for i, (k, v) in enumerate(s.structs.items()):
        payload[f"struct_{i}"] = v
    for i, (k, v) in enumerate(s.structs_moving.items()):
        payload[f"structm_{i}"] = v
    for i, (k, v) in enumerate(s.structs_moving_rigid.items()):
        payload[f"structr_{i}"] = v
    for k, img in exports.items():
        payload[f"export_{k}"] = sitk.GetArrayFromImage(img)
        payload[f"export_{k}_geom"] = np.array(list(img.GetOrigin()) + list(img.GetSpacing()) + list(img.GetDirection()))
    np.savez_compressed(_cache_path(key), **payload)
    return _cache_path(key)


def load_cached(key):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    s = Session.__new__(Session)
    s._summary = json.loads(str(z["summary"]))
    s.params = s._summary["params"]
    s.meta = s._summary["phantom"]
    s.seconds = s._summary["seconds"]
    s.spacing = tuple(float(v) for v in z["spacing"])
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    s.ct_fixed, s.ct_rigid, s.ct_warped = f32(z["ct_fixed"]), f32(z["ct_rigid"]), f32(z["ct_warped"])
    s.dvf, s.jac = f32(z["dvf"]), f32(z["jac"])
    s.dose = {str(n): f32(z[f"dose_{i}"]) for i, n in enumerate(z["dose_names"])}
    s.structs = {str(n): z[f"struct_{i}"] for i, n in enumerate(z["struct_names"])}
    s.structs_moving = {str(n): z[f"structm_{i}"] for i, n in enumerate(z["structm_names"])}
    s.structs_moving_rigid = {str(n): z[f"structr_{i}"] for i, n in enumerate(z["structr_names"])}
    s._exports = {}
    for k in ("eqd2", "physical", "deformed"):
        g = z[f"export_{k}_geom"]
        img = sitk.GetImageFromArray(z[f"export_{k}"])
        img.SetOrigin(tuple(g[:3])); img.SetSpacing(tuple(g[3:6])); img.SetDirection(tuple(g[6:]))
        s._exports[k] = img
    s.res = None
    return s
