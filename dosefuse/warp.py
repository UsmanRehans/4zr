"""Dose warping and grid resampling."""
from __future__ import annotations

import SimpleITK as sitk


def warp_dose(dose_moving: sitk.Image, tx: sitk.Transform, reference_grid: sitk.Image,
              interpolator=sitk.sitkLinear) -> sitk.Image:
    """Map a dose (defined in the moving CT frame) onto the reference grid via tx (fixed->moving).

    Output voxel x receives D_moving(T(x)). Dose is treated as a scalar field,
    i.e. no Jacobian (energy-conserving) correction is applied - this matches
    the usual clinical DIR dose-mapping convention (Velocity, MIM, SlicerRT).
    """
    return sitk.Resample(dose_moving, reference_grid, tx, interpolator, 0.0, sitk.sitkFloat32)


def resample_to(image: sitk.Image, reference_grid: sitk.Image, interpolator=sitk.sitkLinear,
                default=0.0) -> sitk.Image:
    """Resample an image onto another grid with identity transform (same frame of reference)."""
    return sitk.Resample(image, reference_grid, sitk.Transform(), interpolator, default, sitk.sitkFloat32)


def warp_mask(mask: sitk.Image, tx: sitk.Transform, reference_grid: sitk.Image) -> sitk.Image:
    out = sitk.Resample(mask, reference_grid, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return out


def warp_image(img: sitk.Image, tx: sitk.Transform, reference_grid: sitk.Image, default=-1000.0) -> sitk.Image:
    return sitk.Resample(img, reference_grid, tx, sitk.sitkLinear, default, sitk.sitkFloat32)
