"""Registration validation metrics (TG-132 style): Dice, Hausdorff, TRE, dose difference."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def dice(a: sitk.Image, b: sitk.Image) -> float:
    f = sitk.LabelOverlapMeasuresImageFilter()
    f.Execute(sitk.Cast(a, sitk.sitkUInt8), sitk.Cast(b, sitk.sitkUInt8))
    return float(f.GetDiceCoefficient())


def hausdorff(a: sitk.Image, b: sitk.Image) -> dict:
    f = sitk.HausdorffDistanceImageFilter()
    try:
        f.Execute(sitk.Cast(a, sitk.sitkUInt8), sitk.Cast(b, sitk.sitkUInt8))
        return {"hausdorff_mm": float(f.GetHausdorffDistance()),
                "mean_surface_mm": float(f.GetAverageHausdorffDistance())}
    except RuntimeError:
        return {"hausdorff_mm": float("nan"), "mean_surface_mm": float("nan")}


def centroid_mm(mask: sitk.Image) -> np.ndarray:
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(sitk.Cast(mask, sitk.sitkUInt8))
    return np.array(f.GetCentroid(1)) if f.HasLabel(1) else np.full(3, np.nan)


def target_registration_error(tx: sitk.Transform, fixed_points_mm: np.ndarray,
                              moving_points_mm: np.ndarray) -> dict:
    """TRE: distance between T(fixed landmark) and the corresponding moving landmark."""
    errs = []
    for pf, pm in zip(fixed_points_mm, moving_points_mm):
        mapped = np.array(tx.TransformPoint(tuple(float(v) for v in pf)))
        errs.append(np.linalg.norm(mapped - pm))
    errs = np.array(errs)
    return {"tre_mean_mm": float(errs.mean()), "tre_max_mm": float(errs.max()), "tre_per_point": errs}


def dose_difference(a: sitk.Image, b: sitk.Image, mask: sitk.Image | None = None) -> dict:
    da, db = sitk.GetArrayFromImage(a), sitk.GetArrayFromImage(b)
    diff = da - db
    if mask is not None:
        m = sitk.GetArrayFromImage(mask).astype(bool)
        diff = diff[m]
    return {"mean_diff_gy": float(diff.mean()), "abs_mean_diff_gy": float(np.abs(diff).mean()),
            "max_abs_diff_gy": float(np.abs(diff).max()), "rms_diff_gy": float(np.sqrt((diff**2).mean()))}
