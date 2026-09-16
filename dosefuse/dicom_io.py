"""DICOM import/export for CT series, RTDOSE and RTSTRUCT.

All images are returned as SimpleITK images in the DICOM patient (LPS) coordinate
system, so CT, dose and structures from one study can be overlaid directly.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from matplotlib.path import Path as MplPath
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


# --------------------------------------------------------------------------- CT
def load_ct_series(folder: str | os.PathLike) -> sitk.Image:
    """Load a CT series (folder of .dcm slices) as a float32 HU image."""
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(folder))
    if not ids:
        raise FileNotFoundError(f"No DICOM series found in {folder}")
    files = reader.GetGDCMSeriesFileNames(str(folder), ids[0])
    reader.SetFileNames(files)
    img = reader.Execute()
    return sitk.Cast(img, sitk.sitkFloat32)


def write_ct_series(img: sitk.Image, folder: str | os.PathLike, patient_id="PHANTOM",
                    patient_name="DoseFuse^Phantom", study_uid=None, series_desc="CT"):
    """Write an HU image as a DICOM CT series. Returns (study_uid, series_uid, frame_uid)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    study_uid = study_uid or generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    arr_i16 = np.clip(np.round(arr), -1024, 3071).astype(np.int16)
    sp = img.GetSpacing()
    d = np.array(img.GetDirection()).reshape(3, 3)
    now = datetime.datetime.now()
    for k in range(arr.shape[0]):
        ds = _base_dataset("1.2.840.10008.5.1.4.1.1.2", patient_id, patient_name, study_uid, now)
        ds.Modality = "CT"
        ds.SeriesInstanceUID = series_uid
        ds.SeriesDescription = series_desc
        ds.FrameOfReferenceUID = frame_uid
        ds.ImagePositionPatient = [float(v) for v in img.TransformIndexToPhysicalPoint((0, 0, k))]
        ds.ImageOrientationPatient = [float(v) for v in np.concatenate([d[:, 0], d[:, 1]])]
        ds.PixelSpacing = [float(sp[1]), float(sp[0])]
        ds.SliceThickness = float(sp[2])
        ds.InstanceNumber = k + 1
        ds.Rows, ds.Columns = arr.shape[1], arr.shape[2]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = 0.0
        ds.RescaleSlope = 1.0
        ds.KVP = 120
        ds.PixelData = arr_i16[k].tobytes()
        ds.save_as(folder / f"CT.{k+1:04d}.dcm", write_like_original=False)
    return study_uid, series_uid, frame_uid


# ----------------------------------------------------------------------- RTDOSE
def load_rtdose(path: str | os.PathLike) -> sitk.Image:
    """Load an RTDOSE file as a float32 image in Gy (applies DoseGridScaling)."""
    ds = pydicom.dcmread(str(path))
    if ds.Modality != "RTDOSE":
        raise ValueError(f"{path} is not RTDOSE (Modality={ds.Modality})")
    arr = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)
    if arr.ndim == 2:
        arr = arr[None]
    offsets = np.array([float(v) for v in ds.GridFrameOffsetVector], dtype=float)
    dz = float(np.mean(np.diff(offsets))) if len(offsets) > 1 else float(getattr(ds, "SliceThickness", 1.0))
    iop = np.array([float(v) for v in ds.ImageOrientationPatient])
    row_dir, col_dir = iop[:3], iop[3:]
    z_dir = np.cross(row_dir, col_dir)
    if dz < 0:  # frames go against normal
        z_dir, dz = -z_dir, -dz
    origin = np.array([float(v) for v in ds.ImagePositionPatient]) + z_dir * offsets[0] * 0  # first frame is at IPP
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetOrigin(tuple(origin))
    img.SetSpacing((float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]), dz))
    img.SetDirection(tuple(np.column_stack([row_dir, col_dir, z_dir]).ravel()))
    return img


def write_rtdose(dose: sitk.Image, path: str | os.PathLike, *, patient_id="PHANTOM",
                 patient_name="DoseFuse^Phantom", study_uid=None, frame_uid=None,
                 series_desc="Accumulated dose", dose_summation_type="PLAN", dose_type="PHYSICAL"):
    """Write a Gy image as a DICOM RTDOSE (uint32 pixels + DoseGridScaling)."""
    arr = np.clip(sitk.GetArrayFromImage(dose).astype(np.float64), 0, None)
    scaling = float(arr.max()) / (2**32 - 1) if arr.max() > 0 else 1.0
    scaling = max(scaling, 1e-7)
    pix = np.round(arr / scaling).astype(np.uint32)
    sp = dose.GetSpacing()
    d = np.array(dose.GetDirection()).reshape(3, 3)
    now = datetime.datetime.now()
    ds = _base_dataset("1.2.840.10008.5.1.4.1.1.481.2", patient_id, patient_name, study_uid or generate_uid(), now)
    ds.Modality = "RTDOSE"
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = series_desc
    ds.FrameOfReferenceUID = frame_uid or generate_uid()
    ds.ImagePositionPatient = [float(v) for v in dose.GetOrigin()]
    ds.ImageOrientationPatient = [float(v) for v in np.concatenate([d[:, 0], d[:, 1]])]
    ds.PixelSpacing = [float(sp[1]), float(sp[0])]
    ds.SliceThickness = float(sp[2])
    ds.NumberOfFrames = arr.shape[0]
    ds.FrameIncrementPointer = (0x3004, 0x000C)
    ds.GridFrameOffsetVector = [float(k * sp[2]) for k in range(arr.shape[0])]
    ds.Rows, ds.Columns = arr.shape[1], arr.shape[2]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0
    ds.DoseUnits = "GY"
    ds.DoseType = dose_type
    ds.DoseSummationType = dose_summation_type
    ds.DoseGridScaling = scaling
    ds.PixelData = pix.tobytes()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)
    return str(path)


# --------------------------------------------------------------------- RTSTRUCT
def load_rtstruct(path: str | os.PathLike, reference: sitk.Image) -> dict[str, sitk.Image]:
    """Rasterize every ROI in an RTSTRUCT onto the reference image grid.

    Returns {roi_name: uint8 mask image}. Contours are matched to the nearest
    reference slice, and drawn with even-odd fill so holes work.
    """
    ds = pydicom.dcmread(str(path))
    names = {int(r.ROINumber): str(r.ROIName) for r in ds.StructureSetROISequence}
    size = reference.GetSize()  # x, y, z
    # physical coordinates of all voxel centers of one slice, used for point-in-polygon
    xs = np.arange(size[0])
    ys = np.arange(size[1])
    gx, gy = np.meshgrid(xs, ys)  # (ny, nx)
    masks = {}
    for roi in ds.ROIContourSequence:
        name = names.get(int(roi.ReferencedROINumber), f"ROI{roi.ReferencedROINumber}")
        mask = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
        for c in getattr(roi, "ContourSequence", []):
            if getattr(c, "ContourGeometricType", "CLOSED_PLANAR") not in ("CLOSED_PLANAR", "CLOSEDPLANAR"):
                continue
            pts = np.array(c.ContourData, dtype=float).reshape(-1, 3)
            idx = np.array([reference.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in pts])
            k = int(np.round(np.median(idx[:, 2])))
            if not 0 <= k < size[2]:
                continue
            poly = MplPath(idx[:, :2])
            inside = poly.contains_points(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gy.shape)
            mask[k] ^= inside.astype(np.uint8)  # XOR => even-odd fill for holes
        img = sitk.GetImageFromArray(mask)
        img.CopyInformation(reference)
        masks[name] = img
    return masks


def write_rtstruct(masks: dict[str, sitk.Image], path: str | os.PathLike, *, patient_id="PHANTOM",
                   patient_name="DoseFuse^Phantom", study_uid=None, frame_uid=None,
                   ct_series_uid=None, colors=None):
    """Write masks as a minimal RTSTRUCT (closed planar contours extracted per slice)."""
    colors = colors or {}
    now = datetime.datetime.now()
    ds = _base_dataset("1.2.840.10008.5.1.4.1.1.481.3", patient_id, patient_name, study_uid or generate_uid(), now)
    ds.Modality = "RTSTRUCT"
    ds.SeriesInstanceUID = generate_uid()
    ds.StructureSetLabel = "DoseFuse"
    ds.StructureSetDate = now.strftime("%Y%m%d")
    ds.StructureSetTime = now.strftime("%H%M%S")
    frame_uid = frame_uid or generate_uid()
    ref_frame = Dataset()
    ref_frame.FrameOfReferenceUID = frame_uid
    ds.ReferencedFrameOfReferenceSequence = Sequence([ref_frame])
    ds.StructureSetROISequence = Sequence()
    ds.ROIContourSequence = Sequence()
    ds.RTROIObservationsSequence = Sequence()
    from skimage import measure  # lazy: only needed for writing contours
    palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
    for i, (name, mask) in enumerate(masks.items(), start=1):
        roi = Dataset()
        roi.ROINumber = i
        roi.ReferencedFrameOfReferenceUID = frame_uid
        roi.ROIName = name
        roi.ROIGenerationAlgorithm = "AUTOMATIC"
        ds.StructureSetROISequence.append(roi)
        rc = Dataset()
        rc.ReferencedROINumber = i
        rc.ROIDisplayColor = colors.get(name, palette[(i - 1) % len(palette)])
        rc.ContourSequence = Sequence()
        arr = sitk.GetArrayFromImage(mask)
        for k in range(arr.shape[0]):
            sl = arr[k]
            if not sl.any():
                continue
            for contour in measure.find_contours(np.pad(sl, 1).astype(float), 0.5):
                contour = contour - 1  # undo pad; rows=y, cols=x
                if len(contour) < 3:
                    continue
                pts = [mask.TransformContinuousIndexToPhysicalPoint((float(c[1]), float(c[0]), float(k)))
                       for c in contour[::2]]
                c = Dataset()
                c.ContourGeometricType = "CLOSED_PLANAR"
                c.NumberOfContourPoints = len(pts)
                c.ContourData = [float(v) for p in pts for v in p]
                rc.ContourSequence.append(c)
        ds.ROIContourSequence.append(rc)
        ob = Dataset()
        ob.ObservationNumber = i
        ob.ReferencedROINumber = i
        ob.ROIObservationLabel = name
        ob.RTROIInterpretedType = "ORGAN" if name.lower() not in ("gtv", "ptv", "ctv") else "GTV"
        ob.ROIInterpreter = ""
        ds.RTROIObservationsSequence.append(ob)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)
    return str(path)


# ---------------------------------------------------------------------- helpers
def _base_dataset(sop_class_uid, patient_id, patient_name, study_uid, now):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class_uid
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.1"
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = ""
    ds.PatientSex = "O"
    ds.StudyInstanceUID = study_uid
    ds.StudyID = "1"
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.SeriesNumber = 1
    ds.AccessionNumber = ""
    ds.ReferringPhysicianName = ""
    ds.Manufacturer = "DoseFuse"
    return ds


def find_rt_files(folder: str | os.PathLike) -> dict:
    """Scan a folder: return {'ct': [files], 'rtdose': [files], 'rtstruct': [files]}."""
    out = {"ct": [], "rtdose": [], "rtstruct": []}
    for p in sorted(Path(folder).rglob("*")):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            continue
        m = getattr(ds, "Modality", "")
        if m == "CT":
            out["ct"].append(p)
        elif m == "RTDOSE":
            out["rtdose"].append(p)
        elif m == "RTSTRUCT":
            out["rtstruct"].append(p)
    return out
