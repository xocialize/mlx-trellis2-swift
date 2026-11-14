#!/usr/bin/env python3
"""Convert Pixal3D (TencentARC/Pixal3D @ b0cb2e1b) flows and the NAF upsampler to the
mlx-trellis2-swift module-key layout (AB-T-0485 P2).

Flows: fp32 on disk -> bf16 storage (like xocialize/trellis2-mlx). Key changes:
  blocks.N.cross_attn.cross_attn_block.*  ->  blocks.N.cross_attn.*   (the stock names)
  blocks.N.cross_attn.proj_linear.*       kept (read by ModulatedTransformerCrossBlock)
  rope_phases (complex64, SS only)        dropped (SSGridPhases computes it)
NAF: naf_release.pth (valeoai/NAF release asset) -> naf.safetensors, fp32, keys unchanged.

    python -I convert_weights.py PIXAL3D_SRC NAF_SRC OUT_DIR [--mv]
"""
import hashlib
import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

FLOWS = {
    "ss_flow_img_dit_1_3B_64_bf16": "pixal3d_struct_flow",
    "slat_flow_img2shape_dit_1_3B_512_bf16": "pixal3d_shape_flow_512",
    "slat_flow_img2shape_dit_1_3B_1024_bf16": "pixal3d_shape_flow_1024",
    "slat_flow_imgshape2tex_dit_1_3B_1024_bf16": "pixal3d_tex_flow_1024",
}
RENAME = re.compile(r"^(blocks\.\d+\.cross_attn\.)cross_attn_block\.(.*)$")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_flow(src: str, dst: str) -> dict:
    sd = load_file(src)
    out, dropped, renamed = {}, [], 0
    for k, v in sd.items():
        if k == "rope_phases":
            dropped.append(k)
            continue
        m = RENAME.match(k)
        if m:
            k = m.group(1) + m.group(2)
            renamed += 1
        if v.is_floating_point():
            v = v.to(torch.bfloat16)
        out[k] = v.contiguous()
    save_file(out, dst, metadata={"format": "pt", "source": os.path.basename(src),
                                  "converted_by": "mlx-trellis2-swift/Tools/pixal3d/convert_weights.py"})
    proj = sorted(k for k in out if ".proj_linear." in k)
    return {"tensors": len(out), "renamed": renamed, "dropped": dropped,
            "proj_linear_tensors": len(proj),
            "proj_shape": list(out[proj[0]].shape) if proj else None}


def main() -> None:
    src, naf_src, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    mv = "--mv" in sys.argv
    os.makedirs(outdir, exist_ok=True)
    report_path = os.path.join(outdir, "SOURCE.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {
        "pixal3d": {"repo": "TencentARC/Pixal3D", "revision": "b0cb2e1b"},
        "naf": {"repo": "valeoai/NAF", "commit": "37f2dfc",
                "asset": "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"},
        "files": {},
    }
    for stem, name in FLOWS.items():
        for suffix in (["", "_mv"] if mv else [""]):
            s = os.path.join(src, "ckpts", f"{stem}{suffix}.safetensors")
            d = os.path.join(outdir, f"{name}{suffix}.safetensors")
            if not os.path.exists(s):
                print("missing", s, flush=True)
                continue
            if os.path.exists(d) and os.path.basename(d) in report["files"]:
                print("exists", d, flush=True)
                continue
            info = convert_flow(s, d)
            info["source_sha256"] = sha256(s)
            info["sha256"] = sha256(d)
            info["bytes"] = os.path.getsize(d)
            report["files"][os.path.basename(d)] = info
            print(os.path.basename(d), info, flush=True)
            json.dump(report, open(report_path, "w"), indent=1)

    naf_pth = os.path.join(naf_src, "naf_release.pth")
    naf_out = os.path.join(outdir, "naf.safetensors")
    if not os.path.exists(naf_out):
        sd = torch.load(naf_pth, map_location="cpu", weights_only=True)
        save_file({k: v.float().contiguous() for k, v in sd.items()}, naf_out,
                  metadata={"format": "pt", "source": "naf_release.pth"})
        report["files"]["naf.safetensors"] = {"tensors": len(sd), "source_sha256": sha256(naf_pth),
                                              "sha256": sha256(naf_out),
                                              "params": int(sum(v.numel() for v in sd.values()))}
    # licence and notice files travel with the weights (AB-D-0229)
    for f, dstname in (("LICENSE", "LICENSE"), ("NOTICE", "NOTICE")):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(outdir, dstname))
    p = os.path.join(naf_src, "LICENSE")
    if os.path.exists(p):
        shutil.copy2(p, os.path.join(outdir, "NAF_LICENSE"))
    json.dump(report, open(report_path, "w"), indent=1)


if __name__ == "__main__":
    main()
