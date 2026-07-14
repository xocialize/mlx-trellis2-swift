// swift-tools-version: 5.12
import PackageDescription

let package = Package(
    name: "mlx-trellis2-swift",
    platforms: [.macOS(.v14), .iOS(.v16), .visionOS(.v1)],
    products: [
        .library(name: "TRELLIS2", targets: ["TRELLIS2"]),
    ],
    dependencies: [
        .package(path: "../mlx-swift"),
        .package(path: "../mlx-swift-mesh"),
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
                .product(name: "MLXMesh", package: "mlx-swift-mesh"),
            ],
            swiftSettings: [.interoperabilityMode(.Cxx)]  // MLXMesh consumes SwiftXatlas C++ interop
        ),
        .executableTarget(
            name: "parity",
            dependencies: [
                "TRELLIS2",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        .executableTarget(
            name: "dinoparity",
            dependencies: [
                "TRELLIS2",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        .executableTarget(
            name: "meshbake",
            dependencies: [
                "TRELLIS2",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        .executableTarget(
            name: "scaletest",
            dependencies: [
                "TRELLIS2",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
    ]
)
