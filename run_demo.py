"""CLI: generate phantom (if needed), run DIR + dose accumulation, export results.

    python run_demo.py                 # phantom demo
    python run_demo.py --ref study_B --mov study_A --fx-ref 5 --fx-mov 30 --out output/case1
"""
import argparse
import json
import time
from pathlib import Path

from dosefuse import phantom, pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_data", help="phantom folder (generated if missing)")
    ap.add_argument("--ref", help="reference (newer) study folder: CT + RTDOSE + RTSTRUCT")
    ap.add_argument("--mov", help="moving (older) study folder")
    ap.add_argument("--fx-ref", type=int, default=5)
    ap.add_argument("--fx-mov", type=int, default=30)
    ap.add_argument("--method", default="bspline", choices=["bspline", "demons", "rigid"])
    ap.add_argument("--grid", type=float, default=40.0, help="B-spline control point spacing (mm)")
    ap.add_argument("--iters", type=int, default=20, help="deformable iterations per level")
    ap.add_argument("--out", default="output/demo")
    a = ap.parse_args()

    gt = None
    if not (a.ref and a.mov):
        data = Path(a.data)
        if not (data / "phantom_meta.json").exists():
            phantom.generate(data)
        meta = json.loads((data / "phantom_meta.json").read_text())
        a.ref, a.mov = data / "study_B", data / "study_A"
        a.fx_ref, a.fx_mov = meta["fractions_B"], meta["fractions_A"]
        gt = {"transform": str(data / meta["ground_truth_transform"]),
              "landmarks_B_mm": meta["landmarks_B_mm"], "landmarks_A_mm": meta["landmarks_A_mm"]}

    t0 = time.time()
    ref = pipeline.load_study(a.ref, "Reference (B)", a.fx_ref)
    mov = pipeline.load_study(a.mov, "Moving (A)", a.fx_mov)
    print(f"Loaded: CT {ref.ct.GetSize()} dose {ref.dose.GetSize()} structures {list(ref.structures)}")
    res = pipeline.run(ref, mov, a.method, a.grid, a.iters, ground_truth=gt)
    paths = pipeline.export(res, a.out)
    print(f"\nDone in {time.time()-t0:.1f}s. Exports:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("\nCumulative EQD2 metrics (Gy):")
    for s, m in res.structure_metrics.items():
        e = m["eqd2_sum"]
        print(f"  {s:16s} Dmax {e.get('Dmax',0):6.1f}  Dmean {e.get('Dmean',0):6.1f}  D0.1cc {e.get('D0.1cc',0):6.1f}  D2cc {e.get('D2cc',0):6.1f}")
    if "structures" in res.validation:
        print("\nDIR QA (Dice rigid -> DIR):")
        for s, v in res.validation["structures"].items():
            print(f"  {s:16s} {v['dice_rigid']:.3f} -> {v['dice_dir']:.3f}   HD {v['hausdorff_mm_dir']:.1f} mm")


if __name__ == "__main__":
    main()
