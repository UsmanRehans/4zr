"""Cumulative DVH and standard dose-volume metrics."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def voxel_volume_cc(img: sitk.Image) -> float:
    return float(np.prod(img.GetSpacing())) / 1000.0


def dvh(dose: sitk.Image, mask: sitk.Image, bin_gy=0.1) -> dict:
    """Cumulative DVH. Returns dict(dose=[Gy], volume_pct=[%], volume_cc=[cc], ...)."""
    d = sitk.GetArrayFromImage(dose).astype(np.float64)
    m = sitk.GetArrayFromImage(mask).astype(bool)
    vals = d[m]
    vv = voxel_volume_cc(dose)
    if vals.size == 0:
        return {"dose": np.array([0.0]), "volume_pct": np.array([0.0]), "volume_cc": np.array([0.0]),
                "total_cc": 0.0, "values": vals}
    dmax = float(vals.max())
    edges = np.arange(0, dmax + 2 * bin_gy, bin_gy)
    hist, _ = np.histogram(vals, bins=edges)
    cum = np.cumsum(hist[::-1])[::-1]  # volume receiving >= edge
    cum = np.append(cum, 0)
    return {"dose": edges, "volume_pct": 100.0 * cum / vals.size, "volume_cc": cum * vv,
            "total_cc": vals.size * vv, "values": vals}


def metrics(dose: sitk.Image, mask: sitk.Image, v_levels_gy=(), d_cc=(0.03, 0.1, 1.0, 2.0),
            d_pct=(2, 50, 95, 98)) -> dict:
    d = sitk.GetArrayFromImage(dose).astype(np.float64)
    m = sitk.GetArrayFromImage(mask).astype(bool)
    vals = np.sort(d[m])[::-1]
    vv = voxel_volume_cc(dose)
    out = {"volume_cc": vals.size * vv}
    if vals.size == 0:
        return out
    out["Dmax"] = float(vals[0])
    out["Dmean"] = float(vals.mean())
    out["Dmin"] = float(vals[-1])
    for cc in d_cc:
        n = int(np.ceil(cc / vv))
        if n <= vals.size:
            out[f"D{cc:g}cc"] = float(vals[:n].mean() if n > 1 else vals[0])
    for p in d_pct:
        idx = min(vals.size - 1, int(round(p / 100.0 * vals.size)))
        out[f"D{p}%"] = float(vals[idx])
    for lvl in v_levels_gy:
        out[f"V{lvl:g}Gy_cc"] = float((vals >= lvl).sum() * vv)
        out[f"V{lvl:g}Gy_%"] = float(100.0 * (vals >= lvl).mean())
    return out
