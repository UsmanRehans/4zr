"""Synthetic thorax phantom for demos and validation.

Creates two CT studies linked by a KNOWN ground-truth transform (rigid + smooth
B-spline deformation), a 60 Gy/30 fx "original" dose, a 30 Gy/5 fx
"re-irradiation" dose, and GTV / spinal cord / lung / body structures.
Everything is written as DICOM (CT series, RTDOSE, RTSTRUCT) so the full
import pipeline is exercised.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from . import dicom_io

SIZE = (96, 96, 48)        # x, y, z voxels
SPACING = (2.5, 2.5, 3.0)  # mm


def _grid(size=SIZE, spacing=SPACING):
    origin = (-size[0] * spacing[0] / 2, -size[1] * spacing[1] / 2, -size[2] * spacing[2] / 2)
    ref = sitk.Image(size, sitk.sitkFloat32)
    ref.SetSpacing(spacing)
    ref.SetOrigin(origin)
    z, y, x = np.meshgrid(np.arange(size[2]), np.arange(size[1]), np.arange(size[0]), indexing="ij")
    X = origin[0] + x * spacing[0]
    Y = origin[1] + y * spacing[1]
    Z = origin[2] + z * spacing[2]
    return ref, X, Y, Z


def _img(arr, ref, dtype=np.float32):
    im = sitk.GetImageFromArray(np.ascontiguousarray(arr.astype(dtype)))
    im.CopyInformation(ref)
    return im


def make_ct_a():
    ref, X, Y, Z = _grid()
    hu = np.full(X.shape, -1000.0)
    body = (X / 105) ** 2 + (Y / 80) ** 2 <= 1
    hu[body] = 40
    lung_l = ((X + 48) / 36) ** 2 + ((Y - 5) / 48) ** 2 + ((Z) / 75) ** 2 <= 1
    lung_r = ((X - 48) / 36) ** 2 + ((Y - 5) / 48) ** 2 + ((Z) / 75) ** 2 <= 1
    hu[lung_l | lung_r] = -780
    # vessels to give the lungs texture the registration can lock on to
    rng = np.random.default_rng(1)
    for _ in range(40):
        cx, cy = rng.uniform(-80, 80), rng.uniform(-40, 50)
        r = rng.uniform(2.5, 5)
        vessel = ((X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2) & (lung_l | lung_r)
        hu[vessel] = -200 + rng.uniform(-100, 100)
    heart = ((X + 5) / 30) ** 2 + ((Y + 10) / 28) ** 2 + ((Z + 10) / 32) ** 2 <= 1
    hu[heart] = 55
    spine = (X ** 2 + (Y - 58) ** 2 <= 16 ** 2)
    hu[spine] = 350
    cord = (X ** 2 + (Y - 58) ** 2 <= 5 ** 2)
    hu[cord] = 35
    gtv_c = np.array([52.0, -8.0, 4.0])
    gtv = ((X - gtv_c[0]) ** 2 + (Y - gtv_c[1]) ** 2 + (Z - gtv_c[2]) ** 2) <= 14 ** 2
    hu[gtv] = 45
    hu += rng.normal(0, 12, hu.shape)  # noise
    ct = sitk.SmoothingRecursiveGaussian(_img(hu, ref), 1.0)
    masks = {"BODY": _img(body, ref, np.uint8), "LUNG_L": _img(lung_l, ref, np.uint8),
             "LUNG_R": _img(lung_r, ref, np.uint8), "HEART": _img(heart, ref, np.uint8),
             "SPINAL_CORD": _img(cord, ref, np.uint8), "GTV": _img(gtv, ref, np.uint8)}
    return ct, masks, gtv_c


def ground_truth_transform(ref: sitk.Image, seed=7, amplitude_mm=7.0) -> sitk.Transform:
    """Known transform mapping CT_B (fixed) -> CT_A (moving): rigid + smooth B-spline."""
    rigid = sitk.Euler3DTransform()
    rigid.SetCenter((0.0, 0.0, 0.0))
    rigid.SetRotation(np.deg2rad(2.5), np.deg2rad(-1.5), np.deg2rad(3.0))
    rigid.SetTranslation((6.0, -4.0, 3.5))
    bsp = sitk.BSplineTransformInitializer(ref, [4, 4, 3], order=3)
    rng = np.random.default_rng(seed)
    params = np.array(bsp.GetParameters())
    params = rng.normal(0, amplitude_mm, params.shape)
    # add a systematic "posterior chest-wall expansion" so the deformation is not pure noise
    n = params.size // 3
    params[n:2 * n] += np.linspace(-amplitude_mm, amplitude_mm, n)
    bsp.SetParameters(tuple(params))
    return sitk.CompositeTransform([rigid, bsp])


def _dose_field(ref, center, prescription, r_ptv, sigma_mm, grid_spacing=(3.0, 3.0, 3.0), pad_mm=45):
    """Gaussian-penumbra sphere dose, on its own (coarser, cropped) grid like a TPS export."""
    lo = np.array(center) - r_ptv - pad_mm
    hi = np.array(center) + r_ptv + pad_mm
    size = tuple(int(np.ceil((h - l) / s)) for l, h, s in zip(lo, hi, grid_spacing))
    g = sitk.Image(size, sitk.sitkFloat32)
    g.SetSpacing(grid_spacing)
    g.SetOrigin(tuple(float(v) for v in lo))
    z, y, x = np.meshgrid(np.arange(size[2]), np.arange(size[1]), np.arange(size[0]), indexing="ij")
    X = lo[0] + x * grid_spacing[0]
    Y = lo[1] + y * grid_spacing[1]
    Z = lo[2] + z * grid_spacing[2]
    r = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + ((Z - center[2]) * 1.15) ** 2)
    falloff = np.clip(r - r_ptv, 0, None)
    dose = prescription * 1.03 * np.exp(-(falloff ** 2) / (2 * sigma_mm ** 2))
    dose *= (1 + 0.02 * np.cos(X / 15.0))  # mild inhomogeneity
    # low-dose bath
    dose += 0.12 * prescription * np.exp(-(r ** 2) / (2 * (4 * sigma_mm) ** 2))
    return _img(dose, g)


def build(seed=7, amplitude_mm=7.0) -> dict:
    """Build the phantom in memory. Returns studies A/B (as dicts), ground truth transform and landmarks."""
    ct_a, masks_a, gtv_c_a = make_ct_a()
    ref = ct_a
    T_gt = ground_truth_transform(ref, seed, amplitude_mm)  # B -> A

    # CT_B(x) = CT_A(T_gt(x)); structures likewise
    ct_b = sitk.Resample(ct_a, ref, T_gt, sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
    masks_b = {k: sitk.Resample(m, ref, T_gt, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8) for k, m in masks_a.items()}

    # Dose A: 60 Gy / 30 fx to GTV_A + 8 mm margin, in the A frame
    dose_a = _dose_field(ref, gtv_c_a, 60.0, r_ptv=14 + 8, sigma_mm=9.0)

    # Dose B: 30 Gy / 5 fx re-irradiation of a paraspinal recurrence abutting the cord, in B frame
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(masks_b["GTV"])
    gtv_c_b = np.array(f.GetCentroid(1))
    f.Execute(masks_b["SPINAL_CORD"])
    cord_c_b = np.array(f.GetCentroid(1))
    recurrence_c = cord_c_b + np.array([16.0, -20.0, 4.0])
    dose_b = _dose_field(ref, recurrence_c, 30.0, r_ptv=10 + 5, sigma_mm=6.0)
    _, X, Y, Z = _grid()
    rec = ((X - recurrence_c[0]) ** 2 + (Y - recurrence_c[1]) ** 2 + (Z - recurrence_c[2]) ** 2) <= 10 ** 2
    masks_b["GTV_RECURRENCE"] = _img(rec, ref, np.uint8)

    # landmarks: points in B (fixed) frame; ground-truth partner in A = T_gt(p)
    lm_b = [gtv_c_b, recurrence_c, np.array([0.0, 58.0, -40.0]), np.array([0.0, 58.0, 40.0]),
            np.array([-48.0, 5.0, 0.0]), np.array([48.0, 5.0, 0.0]), np.array([-5.0, -10.0, -10.0])]
    lm_a = [np.array(T_gt.TransformPoint(tuple(float(v) for v in p))) for p in lm_b]
    meta = {"fractions_A": 30, "fractions_B": 5, "prescription_A_Gy": 60, "prescription_B_Gy": 30,
            "landmarks_B_mm": [list(map(float, p)) for p in lm_b],
            "landmarks_A_mm": [list(map(float, p)) for p in lm_a]}
    return {"ct_a": ct_a, "dose_a": dose_a, "masks_a": masks_a, "ct_b": ct_b, "dose_b": dose_b,
            "masks_b": masks_b, "T_gt": T_gt, "meta": meta}


def generate(out_dir: str | Path, seed=7, amplitude_mm=7.0, log=print) -> dict:
    """Build the phantom and write it as two DICOM studies (CT series + RTDOSE + RTSTRUCT)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ph = build(seed, amplitude_mm)
    log("Writing DICOM study A (original course)...")
    study_a, _, frame_a = dicom_io.write_ct_series(ph["ct_a"], out / "study_A" / "CT", series_desc="CT original course")
    dicom_io.write_rtdose(ph["dose_a"], out / "study_A" / "RTDOSE.dcm", study_uid=study_a, frame_uid=frame_a,
                          series_desc="Plan A 60Gy/30fx")
    dicom_io.write_rtstruct(ph["masks_a"], out / "study_A" / "RTSTRUCT.dcm", study_uid=study_a, frame_uid=frame_a)
    log("Writing DICOM study B (re-irradiation course)...")
    study_b, _, frame_b = dicom_io.write_ct_series(ph["ct_b"], out / "study_B" / "CT", series_desc="CT re-irradiation")
    dicom_io.write_rtdose(ph["dose_b"], out / "study_B" / "RTDOSE.dcm", study_uid=study_b, frame_uid=frame_b,
                          series_desc="Plan B 30Gy/5fx")
    dicom_io.write_rtstruct(ph["masks_b"], out / "study_B" / "RTSTRUCT.dcm", study_uid=study_b, frame_uid=frame_b)
    sitk.WriteTransform(ph["T_gt"], str(out / "ground_truth_B_to_A.tfm"))
    meta = dict(ph["meta"], ground_truth_transform="ground_truth_B_to_A.tfm")
    (out / "phantom_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"Phantom written to {out}")
    return meta
