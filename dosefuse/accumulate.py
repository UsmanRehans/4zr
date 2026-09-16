"""Dose summation: physical and radiobiological (BED / EQD2) accumulation."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def physical_sum(*doses: sitk.Image) -> sitk.Image:
    out = doses[0]
    for d in doses[1:]:
        out = out + d
    return out


def alpha_beta_map(reference_grid: sitk.Image, structure_masks: dict[str, sitk.Image],
                   alpha_beta: dict[str, float], default=3.0) -> sitk.Image:
    """Per-voxel alpha/beta image. Later entries in `alpha_beta` overwrite earlier ones."""
    ab = np.full(sitk.GetArrayFromImage(reference_grid).shape, default, dtype=np.float32)
    for name, value in alpha_beta.items():
        if name in structure_masks:
            m = sitk.GetArrayFromImage(structure_masks[name]).astype(bool)
            ab[m] = value
    img = sitk.GetImageFromArray(ab)
    img.CopyInformation(reference_grid)
    return img


def bed(dose: sitk.Image, n_fractions: int, alpha_beta: sitk.Image | float) -> sitk.Image:
    """BED = D (1 + d / (a/b)) with d = D / n per voxel."""
    D = sitk.GetArrayFromImage(dose).astype(np.float64)
    ab = sitk.GetArrayFromImage(alpha_beta) if isinstance(alpha_beta, sitk.Image) else float(alpha_beta)
    d = D / max(n_fractions, 1)
    out = D * (1.0 + d / ab)
    img = sitk.GetImageFromArray(out.astype(np.float32))
    img.CopyInformation(dose)
    return img


def eqd2_from_bed(bed_img: sitk.Image, alpha_beta: sitk.Image | float) -> sitk.Image:
    B = sitk.GetArrayFromImage(bed_img).astype(np.float64)
    ab = sitk.GetArrayFromImage(alpha_beta) if isinstance(alpha_beta, sitk.Image) else float(alpha_beta)
    out = B / (1.0 + 2.0 / ab)
    img = sitk.GetImageFromArray(out.astype(np.float32))
    img.CopyInformation(bed_img)
    return img


def eqd2(dose: sitk.Image, n_fractions: int, alpha_beta: sitk.Image | float) -> sitk.Image:
    return eqd2_from_bed(bed(dose, n_fractions, alpha_beta), alpha_beta)


def accumulate(doses_and_fractions: list[tuple[sitk.Image, int]], alpha_beta: sitk.Image | float) -> dict:
    """Return physical sum and EQD2 sum (all doses must be on the same grid)."""
    phys = physical_sum(*[d for d, _ in doses_and_fractions])
    eq = physical_sum(*[eqd2(d, n, alpha_beta) for d, n in doses_and_fractions])
    return {"physical": phys, "eqd2": eq}
