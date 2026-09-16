"""End-to-end workflow: load two studies -> register -> warp dose -> accumulate -> DVH."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from . import accumulate as acc
from . import dicom_io, dvh as dvhmod, metrics, registration, warp

DEFAULT_ALPHA_BETA = {"BODY": 3.0, "LUNG_L": 3.0, "LUNG_R": 3.0, "HEART": 3.0, "SPINAL_CORD": 2.0,
                      "GTV": 10.0, "GTV_RECURRENCE": 10.0, "PTV": 10.0, "CTV": 10.0}


@dataclass
class Study:
    name: str
    ct: sitk.Image
    dose: sitk.Image
    structures: dict[str, sitk.Image]
    fractions: int
    folder: str = ""


def load_study(folder: str | Path, name: str, fractions: int) -> Study:
    files = dicom_io.find_rt_files(folder)
    if not files["ct"]:
        raise FileNotFoundError(f"No CT slices under {folder}")
    ct = dicom_io.load_ct_series(files["ct"][0].parent)
    if not files["rtdose"]:
        raise FileNotFoundError(f"No RTDOSE under {folder}")
    dose = dicom_io.load_rtdose(files["rtdose"][0])
    structures = dicom_io.load_rtstruct(files["rtstruct"][0], ct) if files["rtstruct"] else {}
    return Study(name, ct, dose, structures, fractions, str(folder))


@dataclass
class Result:
    reference: Study
    moving: Study
    reg: registration.RegistrationResult
    moving_ct_rigid: sitk.Image          # moving CT after rigid only, on reference CT grid
    moving_ct_warped: sitk.Image         # moving CT after DIR, on reference CT grid
    moving_structures_warped: dict[str, sitk.Image]
    dose_grid: sitk.Image                # accumulation grid (= reference RTDOSE grid)
    dose_moving_warped: sitk.Image       # moving dose mapped onto the accumulation grid
    dose_moving_rigid: sitk.Image        # moving dose mapped rigidly only (for comparison)
    dose_reference: sitk.Image           # reference dose on the accumulation grid
    dose_sum_physical: sitk.Image
    eqd2_moving: sitk.Image
    eqd2_reference: sitk.Image
    dose_sum_eqd2: sitk.Image
    alpha_beta: sitk.Image
    dvf_stats: dict
    structure_metrics: dict = field(default_factory=dict)
    dvh_curves: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


def run(reference: Study, moving: Study, method="bspline", grid_spacing_mm=40.0,
        deformable_iterations=20, rigid_iterations=200, alpha_beta: dict | None = None,
        default_alpha_beta=3.0, log=print, ground_truth: dict | None = None) -> Result:
    """Deform `moving` (older course) onto `reference` (newer course) and accumulate dose."""
    messages = []

    def _log(m):
        messages.append(m)
        log(m)

    _log(f"Registering {moving.name} -> {reference.name} ({method})")
    reg = registration.register(reference.ct, moving.ct, method, grid_spacing_mm,
                                deformable_iterations, rigid_iterations, _log)
    ct_rigid = warp.warp_image(moving.ct, reg.rigid, reference.ct)
    ct_warped = warp.warp_image(moving.ct, reg.deformable, reference.ct)
    body = reference.structures.get("BODY")
    stats = registration.dvf_stats(reg.dvf, reg.jacobian, body)
    _log(f"DVF: mean {stats['mean_displacement_mm']:.1f} mm, max {stats['max_displacement_mm']:.1f} mm, "
         f"Jacobian [{stats['jacobian_min']:.2f}, {stats['jacobian_max']:.2f}], folding {100*stats['fraction_folding']:.2f}%")

    # accumulation grid = reference RTDOSE grid (TPS dose geometry preserved)
    grid = reference.dose
    dose_ref = warp.resample_to(reference.dose, grid)
    dose_mov_w = warp.warp_dose(moving.dose, reg.deformable, grid)
    dose_mov_r = warp.warp_dose(moving.dose, reg.rigid, grid)
    _log("Warped moving RTDOSE onto reference dose grid")

    # structures on the dose grid: reference structures resampled, moving structures warped
    struct_grid = {k: warp.resample_to(m, grid, sitk.sitkNearestNeighbor) for k, m in reference.structures.items()}
    struct_grid = {k: sitk.Cast(v, sitk.sitkUInt8) for k, v in struct_grid.items()}
    mov_struct_w = {k: warp.warp_mask(m, reg.deformable, reference.ct) for k, m in moving.structures.items()}

    ab = dict(DEFAULT_ALPHA_BETA)
    ab.update(alpha_beta or {})
    ab_map = acc.alpha_beta_map(grid, struct_grid, ab, default_alpha_beta)
    eq_mov = acc.eqd2(dose_mov_w, moving.fractions, ab_map)
    eq_ref = acc.eqd2(dose_ref, reference.fractions, ab_map)
    phys = acc.physical_sum(dose_mov_w, dose_ref)
    eq_sum = acc.physical_sum(eq_mov, eq_ref)
    _log("Computed physical sum and EQD2 sum")

    res = Result(reference, moving, reg, ct_rigid, ct_warped, mov_struct_w, grid, dose_mov_w, dose_mov_r,
                 dose_ref, phys, eq_mov, eq_ref, eq_sum, ab_map, stats, log=messages)

    # DVH + metrics on reference structures
    for name, m in struct_grid.items():
        res.structure_metrics[name] = {
            "moving_warped": dvhmod.metrics(dose_mov_w, m),
            "reference": dvhmod.metrics(dose_ref, m),
            "physical_sum": dvhmod.metrics(phys, m),
            "eqd2_sum": dvhmod.metrics(eq_sum, m),
        }
        res.dvh_curves[name] = {
            "moving_warped": dvhmod.dvh(dose_mov_w, m),
            "reference": dvhmod.dvh(dose_ref, m),
            "physical_sum": dvhmod.dvh(phys, m),
            "eqd2_sum": dvhmod.dvh(eq_sum, m),
        }

    # structure agreement: warped moving structures vs reference structures (DIR QA, TG-132)
    val = {}
    for name in reference.structures:
        if name in mov_struct_w:
            ref_m, mov_m = reference.structures[name], mov_struct_w[name]
            rig_m = warp.warp_mask(moving.structures[name], reg.rigid, reference.ct)
            val[name] = {"dice_rigid": metrics.dice(ref_m, rig_m), "dice_dir": metrics.dice(ref_m, mov_m),
                         **{k + "_dir": v for k, v in metrics.hausdorff(ref_m, mov_m).items()}}
    res.validation["structures"] = val
    if ground_truth:
        gt_tx = ground_truth["transform"]
        if isinstance(gt_tx, str):
            gt_tx = sitk.ReadTransform(gt_tx)
        lm_b, lm_a = np.array(ground_truth["landmarks_B_mm"]), np.array(ground_truth["landmarks_A_mm"])
        res.validation["tre_rigid"] = metrics.target_registration_error(reg.rigid, lm_b, lm_a)
        res.validation["tre_dir"] = metrics.target_registration_error(reg.deformable, lm_b, lm_a)
        gt_dose = warp.warp_dose(moving.dose, gt_tx, grid)
        res.validation["dose_vs_truth"] = {
            "rigid": metrics.dose_difference(dose_mov_r, gt_dose, struct_grid.get("BODY")),
            "dir": metrics.dose_difference(dose_mov_w, gt_dose, struct_grid.get("BODY")),
        }
        res.validation["ground_truth_dose"] = gt_dose
        _log(f"TRE vs ground truth: rigid {res.validation['tre_rigid']['tre_mean_mm']:.2f} mm -> "
             f"DIR {res.validation['tre_dir']['tre_mean_mm']:.2f} mm")
    return res


def export(res: Result, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "rtdose_physical_sum": dicom_io.write_rtdose(res.dose_sum_physical, out / "RTDOSE_sum_physical.dcm",
                                                     series_desc="DoseFuse physical sum", dose_type="PHYSICAL"),
        "rtdose_eqd2_sum": dicom_io.write_rtdose(res.dose_sum_eqd2, out / "RTDOSE_sum_EQD2.dcm",
                                                 series_desc="DoseFuse EQD2 sum", dose_type="EFFECTIVE"),
        "rtdose_moving_warped": dicom_io.write_rtdose(res.dose_moving_warped, out / "RTDOSE_A_deformed_to_B.dcm",
                                                      series_desc="DoseFuse deformed dose A"),
    }
    sitk.WriteImage(res.reg.dvf, str(out / "dvf.mha"))
    sitk.WriteImage(res.reg.jacobian, str(out / "jacobian.mha"))
    sitk.WriteTransform(res.reg.deformable, str(out / "transform_B_to_A.tfm"))
    paths.update({"dvf": str(out / "dvf.mha"), "jacobian": str(out / "jacobian.mha"),
                  "transform": str(out / "transform_B_to_A.tfm")})
    summary = {"dvf_stats": res.dvf_stats, "structure_metrics": res.structure_metrics,
               "validation": {k: v for k, v in res.validation.items() if k != "ground_truth_dose"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    paths["summary"] = str(out / "summary.json")
    return paths


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
