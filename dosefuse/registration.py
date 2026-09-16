"""Rigid + deformable (B-spline / Demons) registration with SimpleITK.

Convention (SimpleITK): the resulting transform T maps points in the FIXED
(reference) image space to the MOVING image space.  Resampling the moving
image (or its dose) with T therefore produces D_moving(T(x)) on the fixed grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk


@dataclass
class RegistrationResult:
    rigid: sitk.Transform
    deformable: sitk.Transform          # composite: rigid + deformable
    dvf: sitk.Image                     # displacement field on the fixed grid (mm)
    jacobian: sitk.Image                # det(J) on the fixed grid
    metric_history: dict = field(default_factory=dict)
    final_metric: float = 0.0


def _mask_body(img: sitk.Image, threshold_hu=-500) -> sitk.Image:
    m = sitk.BinaryThreshold(img, threshold_hu, 4000, 1, 0)
    m = sitk.BinaryMorphologicalClosing(m, [3, 3, 3])
    return sitk.Cast(m, sitk.sitkUInt8)


def register_rigid(fixed: sitk.Image, moving: sitk.Image, iterations=200, log=None) -> tuple[sitk.Transform, list]:
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2, seed=42)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-3,
                                               numberOfIterations=iterations, relaxationFactor=0.6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(init, inPlace=True)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    hist = []
    R.AddCommand(sitk.sitkIterationEvent, lambda: hist.append(R.GetMetricValue()))
    tx = R.Execute(fixed, moving)
    if log:
        log(f"Rigid: {R.GetOptimizerStopConditionDescription()} metric={R.GetMetricValue():.4f}")
    return tx, hist


def register_bspline(fixed: sitk.Image, moving: sitk.Image, initial: sitk.Transform,
                     grid_spacing_mm=40.0, iterations=20, metric="meansquares", sampling=0.03,
                     log=None) -> tuple[sitk.Transform, list]:
    """Multi-resolution (4x/2x/1x) B-spline DIR on top of an initial (rigid) transform, LBFGS-B optimizer.

    `iterations` is per resolution level. Defaults (20 it, 3% random samples) reach the same accuracy on
    the phantom as 100 it / 10% but run ~8x faster, which matters on a single-vCPU cloud function.
    """
    phys = [sz * sp for sz, sp in zip(fixed.GetSize(), fixed.GetSpacing())]
    mesh = [max(1, int(round(p / grid_spacing_mm))) for p in phys]
    bspline = sitk.BSplineTransformInitializer(fixed, mesh, order=3)
    R = sitk.ImageRegistrationMethod()
    if metric == "mattes":
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(sampling, seed=42)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iterations,
                           maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=iterations * 2)
    R.SetMovingInitialTransform(initial)
    R.SetInitialTransform(bspline, inPlace=True)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    hist = []
    R.AddCommand(sitk.sitkIterationEvent, lambda: hist.append(R.GetMetricValue()))
    R.Execute(fixed, moving)
    if log:
        log(f"B-spline ({len(hist)} iterations, mesh {mesh}): {R.GetOptimizerStopConditionDescription().split('.')[0]}")
    composite = sitk.CompositeTransform([initial, bspline])  # fixed -> moving
    return composite, hist


def register_demons(fixed: sitk.Image, moving: sitk.Image, initial: sitk.Transform,
                    iterations=60, sigma=2.0, log=None) -> tuple[sitk.Transform, list]:
    """Symmetric-forces Demons after rigid pre-alignment."""
    moving_r = sitk.Resample(moving, fixed, initial, sitk.sitkLinear, -1000.0)
    f = sitk.Clamp(fixed, sitk.sitkFloat32, -1000, 1000)
    m = sitk.Clamp(moving_r, sitk.sitkFloat32, -1000, 1000)
    d = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    d.SetNumberOfIterations(iterations)
    d.SetStandardDeviations(sigma)
    hist = []
    d.AddCommand(sitk.sitkIterationEvent, lambda: hist.append(d.GetMetric()))
    field_img = d.Execute(f, m)
    disp = sitk.DisplacementFieldTransform(sitk.Cast(field_img, sitk.sitkVectorFloat64))
    if log:
        log(f"Demons: RMS metric={d.GetMetric():.3f}")
    return sitk.CompositeTransform([initial, disp]), hist


def deformation_field(tx: sitk.Transform, reference: sitk.Image) -> sitk.Image:
    """Displacement field (mm, vector float64) of tx sampled on the reference grid."""
    return sitk.TransformToDisplacementField(tx, sitk.sitkVectorFloat64, reference.GetSize(),
                                             reference.GetOrigin(), reference.GetSpacing(),
                                             reference.GetDirection())


def jacobian_determinant(dvf: sitk.Image) -> sitk.Image:
    return sitk.DisplacementFieldJacobianDeterminant(sitk.Cast(dvf, sitk.sitkVectorFloat64))


def register(fixed: sitk.Image, moving: sitk.Image, method="bspline", grid_spacing_mm=40.0,
             deformable_iterations=20, rigid_iterations=200, log=print) -> RegistrationResult:
    """Full pipeline: rigid -> deformable, plus DVF and Jacobian on the fixed grid."""
    rigid, h_rigid = register_rigid(fixed, moving, rigid_iterations, log)
    if method == "bspline":
        tx, h_def = register_bspline(fixed, moving, rigid, grid_spacing_mm, deformable_iterations, log)
    elif method == "demons":
        tx, h_def = register_demons(fixed, moving, rigid, deformable_iterations, log=log)
    elif method == "rigid":
        tx, h_def = rigid, []
    else:
        raise ValueError(method)
    dvf = deformation_field(tx, fixed)
    jac = jacobian_determinant(dvf)
    return RegistrationResult(rigid=rigid, deformable=tx, dvf=dvf, jacobian=jac,
                              metric_history={"rigid": h_rigid, "deformable": h_def},
                              final_metric=float(h_def[-1]) if h_def else float(h_rigid[-1] if h_rigid else 0))


def dvf_stats(dvf: sitk.Image, jac: sitk.Image, mask: sitk.Image | None = None) -> dict:
    v = sitk.GetArrayFromImage(dvf)
    mag = np.linalg.norm(v, axis=-1)
    j = sitk.GetArrayFromImage(jac)
    if mask is not None:
        m = sitk.GetArrayFromImage(mask).astype(bool)
        mag, j = mag[m], j[m]
    return {
        "mean_displacement_mm": float(mag.mean()),
        "max_displacement_mm": float(mag.max()),
        "p95_displacement_mm": float(np.percentile(mag, 95)),
        "jacobian_min": float(j.min()),
        "jacobian_max": float(j.max()),
        "jacobian_mean": float(j.mean()),
        "fraction_folding": float((j <= 0).mean()),
        "fraction_extreme": float(((j < 0.5) | (j > 2.0)).mean()),
    }
