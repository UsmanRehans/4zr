"""Matplotlib rendering helpers for the demo UI (slices, fusion, isodoses, DVF, Jacobian)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import LinearSegmentedColormap

AXES = {"axial": 0, "coronal": 1, "sagittal": 2}
STRUCT_COLORS = {"BODY": "#9e9e9e", "LUNG_L": "#4fc3f7", "LUNG_R": "#4fc3f7", "HEART": "#ef5350",
                 "SPINAL_CORD": "#ffee58", "GTV": "#ff7043", "GTV_RECURRENCE": "#ab47bc"}
ISODOSE_CMAP = LinearSegmentedColormap.from_list("iso", ["#0000ff00", "#0000ff", "#00ffff", "#00ff00",
                                                         "#ffff00", "#ff8000", "#ff0000", "#ffffff"])


def to_array(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)


def take_slice(arr: np.ndarray, axis: str, idx: int) -> np.ndarray:
    a = AXES[axis]
    idx = int(np.clip(idx, 0, arr.shape[a] - 1))
    if a == 0:
        return arr[idx]
    if a == 1:
        return arr[:, idx, :][::-1]
    return arr[:, :, idx][::-1]


def aspect(spacing, axis: str) -> float:
    sx, sy, sz = spacing
    return {"axial": sy / sx, "coronal": sz / sx, "sagittal": sz / sy}[axis]


def window(ct: np.ndarray, level=40, width=400) -> np.ndarray:
    lo, hi = level - width / 2, level + width / 2
    return np.clip((ct - lo) / (hi - lo), 0, 1)


def n_slices(arr: np.ndarray, axis: str) -> int:
    return arr.shape[AXES[axis]]


def fig_fusion(ct_fixed, ct_moving, spacing, axis, idx, mode="blend", alpha=0.5, structures=None,
               moving_structures=None, title=""):
    f = window(take_slice(ct_fixed, axis, idx))
    m = window(take_slice(ct_moving, axis, idx))
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    if mode == "blend":
        rgb = np.stack([f * (1 - alpha) + m * alpha, f * (1 - alpha), f * (1 - alpha) + m * alpha], -1)
        # fixed in green channel, moving in magenta: grey where they agree
        rgb = np.stack([m, f, m], -1) if alpha >= 0.999 else np.stack([m * alpha + f * (1 - alpha), f, m * alpha + f * (1 - alpha)], -1)
        ax.imshow(np.clip(rgb, 0, 1), aspect=aspect(spacing, axis))
    elif mode == "checker":
        n = 8
        yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
        mask = ((yy // (f.shape[0] // n)) + (xx // (f.shape[1] // n))) % 2 == 0
        ax.imshow(np.where(mask, f, m), cmap="gray", vmin=0, vmax=1, aspect=aspect(spacing, axis))
    elif mode == "moving":
        ax.imshow(m, cmap="gray", vmin=0, vmax=1, aspect=aspect(spacing, axis))
    else:
        ax.imshow(f, cmap="gray", vmin=0, vmax=1, aspect=aspect(spacing, axis))
    _draw_structures(ax, structures, axis, idx, "-")
    _draw_structures(ax, moving_structures, axis, idx, "--")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def fig_dose(ct, dose, spacing, axis, idx, dmax=None, structures=None, title="", isodose_levels=None,
             alpha=0.55, cmap="jet"):
    c = window(take_slice(ct, axis, idx))
    d = take_slice(dose, axis, idx)
    dmax = dmax or max(float(np.nanmax(dose)), 1e-3)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    ax.imshow(c, cmap="gray", vmin=0, vmax=1, aspect=aspect(spacing, axis))
    dm = np.ma.masked_less(d, 0.02 * dmax)
    im = ax.imshow(dm, cmap=cmap, vmin=0, vmax=dmax, alpha=alpha, aspect=aspect(spacing, axis))
    if isodose_levels:
        lv = [l for l in isodose_levels if l < d.max()]
        if lv:
            ax.contour(d, levels=sorted(lv), colors="white", linewidths=0.7)
    _draw_structures(ax, structures, axis, idx, "-")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Gy", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.2)
    return fig


def fig_dvf(ct, dvf, spacing, axis, idx, step=4, title="Deformation vector field"):
    c = window(take_slice(ct, axis, idx))
    a = AXES[axis]
    comps = {"axial": (0, 1), "coronal": (0, 2), "sagittal": (1, 2)}[axis]
    u = take_slice(dvf[..., comps[0]], axis, idx)
    v = take_slice(dvf[..., comps[1]], axis, idx)
    if axis != "axial":
        v = -v  # slice was flipped vertically
    mag = np.sqrt(u ** 2 + v ** 2)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    ax.imshow(c, cmap="gray", vmin=0, vmax=1, aspect=aspect(spacing, axis))
    im = ax.imshow(mag, cmap="magma", alpha=0.45, aspect=aspect(spacing, axis))
    yy, xx = np.mgrid[0:u.shape[0]:step, 0:u.shape[1]:step]
    ax.quiver(xx, yy, u[::step, ::step], v[::step, ::step], color="cyan", angles="xy", scale_units="xy",
              scale=spacing[0] * 0.8, width=0.004)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("|u| mm", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.2)
    return fig


def fig_jacobian(ct, jac, spacing, axis, idx, title="Jacobian determinant"):
    c = window(take_slice(ct, axis, idx))
    j = take_slice(jac, axis, idx)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    ax.imshow(c, cmap="gray", vmin=0, vmax=1, aspect=aspect(spacing, axis))
    im = ax.imshow(j, cmap="RdBu_r", vmin=0.5, vmax=1.5, alpha=0.6, aspect=aspect(spacing, axis))
    if (j <= 0).any():
        ax.contour(j <= 0, levels=[0.5], colors="lime", linewidths=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("det J  (<1 compression, >1 expansion)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.2)
    return fig


def _draw_structures(ax, structures, axis, idx, ls):
    if not structures:
        return
    for name, arr in structures.items():
        s = take_slice(arr, axis, idx)
        if s.any():
            ax.contour(s.astype(float), levels=[0.5], colors=[STRUCT_COLORS.get(name, "#ffffff")],
                       linewidths=1.0, linestyles=ls)
