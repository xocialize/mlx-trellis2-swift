#!/usr/bin/env python3
"""Convert the Pixal3D CUDA goldens (AB-T-0485 P1, .npz) into one .safetensors per .npz.

MLX-Swift reads .npy/.safetensors but not .npz, so the `pixal3dparity` gate reads this mirror.
Keys keep the npz array names. float64 -> float32 and int64 -> int32 (MLX has no f64 on GPU);
fp16 stays fp16. The source folder is left untouched.

    python -I convert_goldens.py SRC_DIR DST_DIR
    # SRC_DIR = /Volumes/DEV_VOL1/Evidence/pixal3d-oracle-2026-11-14
    # DST_DIR = /Volumes/DEV_VOL1/Evidence/pixal3d-oracle-2026-11-14-st
"""
import json
import os
import shutil
import sys

import numpy as np
from safetensors.numpy import save_file


def convert_run(src_run: str, dst_run: str) -> int:
    gsrc = os.path.join(src_run, "golden")
    gdst = os.path.join(dst_run, "golden")
    os.makedirs(gdst, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(gsrc)):
        if name == "manifest.json":
            shutil.copy2(os.path.join(gsrc, name), os.path.join(gdst, name))
            continue
        if not name.endswith(".npz"):
            continue
        out = os.path.join(gdst, name[:-4] + ".safetensors")
        if os.path.exists(out):
            n += 1
            continue
        with np.load(os.path.join(gsrc, name)) as z:
            tensors = {}
            for k in z.files:
                a = np.asarray(z[k])
                if a.dtype == np.float64:
                    a = a.astype(np.float32)
                elif a.dtype == np.int64:
                    a = a.astype(np.int32)
                elif a.dtype == np.bool_:
                    a = a.astype(np.uint8)
                if a.ndim == 0:
                    a = a.reshape(1)
                tensors[k] = np.ascontiguousarray(a)
        save_file(tensors, out)
        n += 1
    for extra in ("info.json",):
        p = os.path.join(src_run, extra)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst_run, extra))
    return n


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    summary = {}
    for run in sorted(os.listdir(src)):
        if not run.startswith("golden_") or not os.path.isdir(os.path.join(src, run)):
            continue
        summary[run] = convert_run(os.path.join(src, run), os.path.join(dst, run))
        print(run, summary[run], "files", flush=True)
    with open(os.path.join(dst, "CONVERTED.json"), "w") as f:
        json.dump({"source": src, "runs": summary,
                   "tool": "mlx-trellis2-swift/Tools/pixal3d/convert_goldens.py"}, f, indent=1)


if __name__ == "__main__":
    main()
