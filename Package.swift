// swift-tools-version: 5.12
import PackageDescription

// mlx-trellis2-swift — native Swift-MLX port of Microsoft TRELLIS.2 (image → textured GLB).
//
// TRELLIS2  : the parity-validated neural core (pure MLX + mlx-swift-mesh; NO engine import).
//             Every op gated cosine vs the PyTorch oracle. This is the Stage-1 Path-B port.
// Trellis2Kit: the MLXToolKit `ModelPackage` conformer (capability imageTo3D) — manifest,
//             two-layer license gate, WeightSourcing auto-materialization, load()/run() over
//             TRELLIS2's Trellis2Pipeline → UV-textured GLB. Depends on MLXToolKit (offline
//             contract) + swift-huggingface (first-run weight materialization).
let package = Package(
    name: "mlx-trellis2-swift",
    platforms: [.macOS("26.0")],   // MLXToolKit engine-contract floor
    products: [
        .library(name: "TRELLIS2", targets: ["TRELLIS2"]),
        .library(name: "Trellis2Kit", targets: ["Trellis2Kit"]),
    ],
    dependencies: [
        // Neural runtime — sibling path checkouts (spike layout). The sibling mlx-swift is 0.31.3;
        // mlx-engine-swift declares `mlx-swift from: 0.31.5`, but a `path` dependency overrides the
        // version constraint (SPM uses the on-disk package for that identity), so the graph resolves
        // on 0.31.3. See the SW6 report for the exact conflict + resolution.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.5"),  // 0.31.6 de-risk-verified (M5 NAX/TF32 within cosine gates; no workaround needed)
        // (mlx-swift-mesh + SwiftXatlas are now VENDORED in-tree as the MLXMesh/Xatlas/Cxatlas targets
        //  below — self-contained, no external path deps; see THIRD_PARTY_LICENSES.md.)
        // MLXEngine contract + coordinator + conformance harness (URL-consumed like the sibling
        // -swift wrappers). imageTo3D stable since 1.12.0; 0.30.0 carries the split QuantFootprint,
        // MAT gate (MaterializationConformance) and CAN gate (CancellationConformance).
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.31.0"),
        // First-run auto-materialization (MAT): HubClient snapshot download into the ModelStore layout.
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
    ],
    targets: [
        .target(
            name: "TRELLIS2",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                "MLXMesh",   // vendored (was the mlx-swift-mesh package product)
            ],
            swiftSettings: [.interoperabilityMode(.Cxx)]  // MLXMesh consumes the vendored xatlas C++ interop
        ),
        // The MLXToolKit ModelPackage conformer (capability imageTo3D). Offline-buildable against
        // MLXToolKit alone; HuggingFace is only touched at load()-time materialization.
        .target(
            name: "Trellis2Kit",
            dependencies: [
                "TRELLIS2",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        // --- parity / dev CLIs (GPU stream; run via `swift run`, never `swift test`) ---
        .executableTarget(name: "parity", dependencies: [
            "TRELLIS2", .product(name: "MLX", package: "mlx-swift"),
        ], swiftSettings: [.interoperabilityMode(.Cxx)]),
        .executableTarget(name: "dinoparity", dependencies: [
            "TRELLIS2", .product(name: "MLX", package: "mlx-swift"),
        ], swiftSettings: [.interoperabilityMode(.Cxx)]),
        .executableTarget(name: "meshbake", dependencies: [
            "TRELLIS2", .product(name: "MLX", package: "mlx-swift"),
        ], swiftSettings: [.interoperabilityMode(.Cxx)]),
        .executableTarget(name: "scaletest", dependencies: [
            "TRELLIS2",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXRandom", package: "mlx-swift"),
        ], swiftSettings: [.interoperabilityMode(.Cxx)]),
        .executableTarget(name: "generate", dependencies: [
            "TRELLIS2", .product(name: "MLX", package: "mlx-swift"),
        ], swiftSettings: [.interoperabilityMode(.Cxx)]),
        // --- Stage-2 packaging CLIs ---
        // Born-clean conformance gate lane (C0–C13 static + MAT-1..5 + CAN-1..3). Uses
        // MLXServeConformance; NO MLX kernels, NO weights — safe as a plain `swift run` (the CLI lane
        // the skill prescribes so nothing runs under `swift test`, whose metallib is unreliable).
        .executableTarget(name: "trellis2-gate", dependencies: [
            "Trellis2Kit",
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
        ], swiftSettings: [.interoperabilityMode(.Cxx)]),
        // Weight consolidation tool: assembles the verified per-component safetensors + a
        // normalization.json into the module-key layout of the consolidated repo (xocialize/
        // trellis2-mlx) locally, so load() resolves them from an explicit weightsRootOverride now
        // (and from the ModelStore once the repo is published). No key remap — the published keys
        // ARE the Swift module keys.
        .executableTarget(name: "trellis2-consolidate", dependencies: [
            "TRELLIS2", .product(name: "MLX", package: "mlx-swift"),
        ], swiftSettings: [.interoperabilityMode(.Cxx)]),
        // Engine-driven GPU e2e CLI (Stage-2 step 7): drives the package THROUGH MLXServeEngine
        // (register → prepare → run → decode Mesh → GLB) on a real image. Real weights, GPU stream.
        .executableTarget(name: "trellis2-run-engine", dependencies: [
            "Trellis2Kit",
            .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            .product(name: "MLX", package: "mlx-swift"),
        ], swiftSettings: [.interoperabilityMode(.Cxx)]),
        // --- VENDORED mesh + xatlas (were ../mlx-swift-mesh + ../SwiftXatlas path deps; folded
        //     in-tree so the package is self-contained and we own maintenance). ---
        // xatlas C++ (jpcy/xatlas, MIT — see THIRD_PARTY_LICENSES.md): the atlas UV parameterizer,
        // built from the vendored xatlas.cpp + a thin C shim (symlinks dereferenced into real files).
        // LIVE DEPENDENCY — do NOT strip as dead weight: it's the UV-unwrap backend the texture bake
        // actually calls (MeshBake.run → mesh.uvUnwrap() → MLXMesh/UVUnwrap.swift → Xatlas.Atlas()).
        // It's fast ONLY because MeshBake unwraps the DC-remeshed + simplified mesh; RAW xatlas on the
        // unconditioned dual-grid mesh is pathologically slow (~2.2h) — the reason it was first
        // abandoned. Conditioned-unwrap is the reference-fidelity path (mirrors cumesh.uv_unwrap).
        .target(
            name: "Cxatlas",
            path: "Sources/Cxatlas",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("include"),
                .define("NDEBUG"),   // silence xatlas internal XA_DEBUG_ASSERTs on consumer error paths
            ]
        ),
        // Swift C++-interop wrapper over Cxatlas.
        .target(
            name: "Xatlas",
            dependencies: ["Cxatlas"],
            path: "Sources/Xatlas",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        // MLX mesh ops (BVH, DC-remesh, simplify, UV unwrap) — consumes Xatlas.
        .target(
            name: "MLXMesh",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                "Xatlas",
            ],
            path: "Sources/MLXMesh",
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
    ],
    cxxLanguageStandard: .cxx14
)
