# Third-party licenses

`mlx-trellis2-swift` vendors the following third-party code in-tree (previously consumed as the
`../mlx-swift-mesh` and `../SwiftXatlas` sibling packages). Each is redistributed under its MIT
license, reproduced in full below.

| Vendored path | Origin | License | Copyright |
|---|---|---|---|
| `Sources/MLXMesh/` | mlx-swift-mesh | MIT | © 2026 Hiroaki Yamane |
| `Sources/Cxatlas/xatlas.cpp`, `Sources/Cxatlas/include/xatlas.h` | [jpcy/xatlas](https://github.com/jpcy/xatlas) | MIT | © 2018–2020 Jonathan Young |
| `Sources/Xatlas/`, `Sources/Cxatlas/xatlas_shim.{cpp,h}`, `module.modulemap` | SwiftXatlas (thin Swift/C++ wrapper over xatlas) | MIT (per xatlas) | wrapper code |

The vendored files are byte-for-byte copies (the two symlinked xatlas sources were dereferenced into
real files); no functional modifications were made during vendoring.

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
