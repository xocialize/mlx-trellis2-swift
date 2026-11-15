# Third-party licenses

`mlx-trellis2-swift` vendors the following third-party code in-tree (previously consumed as the
`../mlx-swift-mesh` and `../SwiftXatlas` sibling packages). Each is redistributed under its MIT
license, reproduced in full below.

| Vendored path | Origin | License | Copyright |
|---|---|---|---|
| `Sources/MLXMesh/` | mlx-swift-mesh | MIT | © 2026 Hiroaki Yamane |
| `Sources/Cxatlas/xatlas.cpp`, `Sources/Cxatlas/include/xatlas.h` | [jpcy/xatlas](https://github.com/jpcy/xatlas) | MIT | © 2018–2020 Jonathan Young |
| `Sources/Xatlas/`, `Sources/Cxatlas/xatlas_shim.{cpp,h}`, `module.modulemap` | SwiftXatlas (thin Swift/C++ wrapper over xatlas) | MIT (per xatlas) | wrapper code |
| `Sources/Cfdg/fdg.cpp` | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) `o-voxel/src/convert/flexible_dual_grid.cpp` | MIT | © 2025 Jianfeng XIANG / Microsoft |
| `Sources/Cfdg/eigen/Eigen/` | [Eigen](https://gitlab.com/libeigen/eigen) @ `21e4582d` (upstream o-voxel submodule pin) | MPL-2.0 (`COPYING.MPL2` in-tree) | Eigen contributors |

The vendored files are byte-for-byte copies (the two symlinked xatlas sources were dereferenced into
real files); no functional modifications were made during vendoring.

Cfdg deviations from upstream (documented, no algorithm changes): the torch tensor entry point is
replaced by a plain-buffer C ABI (`fdg_shim.h`); two GNU `0.0d` float-suffix literals were changed
to standard `0.0` (clang rejects the extension); the decode-direction mesh-extraction helpers
(`quad_to_2tri`, `face_from_dual_vertices`) are omitted — the Swift port already has its own
bit-exact dual-grid→mesh path (`DualGridMesh.swift`). Parity: `swift run fdgparity` — voxel set,
intersection flags, and dual vertices bit-exact vs the Python oracle at 64³/256³.

---

## xatlas — the atlas UV parameterizer (`Sources/Cxatlas/xatlas.cpp`, `xatlas.h`)

```
MIT License

Copyright (c) 2018-2020 Jonathan Young

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## mlx-swift-mesh — the MLX mesh ops (`Sources/MLXMesh/`)

```
MIT License

Copyright (c) 2026 Hiroaki Yamane

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Pixal3D and NAF — ported code (branch `pixal3d`, AB-T-0485; not released)

These are Swift translations, not vendored copies. The converted weights carry their own notices
(Tencent's `LICENSE` and `NOTICE` from `TencentARC/Pixal3D` @ `b0cb2e1b`, and NAF's Apache-2.0 text,
next to the weight files; AB-D-0229).

| Ported path | Origin | License | Copyright |
|---|---|---|---|
| `Sources/TRELLIS2/Pixal3D.swift`, `Pixal3DPipeline.swift`, the `proj_linear` term in `TransformerBlock.swift` | [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) @ `f7cf384` (`image_conditioned_proj.py`, `proj_attention.py`, `pixal3d_image_to_3d.py`) | MIT | © 2026 Tencent |
| `Sources/TRELLIS2/NAF.swift` | [valeoai/NAF](https://github.com/valeoai/NAF) @ `37f2dfc` (`src/model/naf.py`, `src/layers/`) | Apache-2.0 | valeo.ai |

NAF's axial RoPE (`src/layers/rope.py`) carries Meta's DINOv3 header; the port computes the same
formula. NATTEN (the CUDA kernel under NAF upstream) is not used: the neighbourhood attention is
written in MLX.

### Pixal3D — MIT

```
MIT License

Copyright (c) 2026 Tencent.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### NAF — Apache License 2.0

The full text is in the NAF repository's `LICENSE` and ships as `NAF_LICENSE` with the converted
weights. Changes in the port: translated to Swift/MLX; NATTEN replaced by an MLX gather and softmax;
the image encoder's input-downsample branch (unused by Pixal3D) is not ported.

