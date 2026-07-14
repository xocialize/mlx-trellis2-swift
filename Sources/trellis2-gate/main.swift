// trellis2-gate — the born-clean offline conformance lane for Trellis2Package.
//
// Runs the C0–C13 static gate + the MAT-1..5 gate (MaterializationConformance) + the CAN-1..3 gate
// (CancellationConformance) with NO MLX kernels and NO weights, so it is a plain `swift run` (the
// CLI lane the mlx-swift-integration skill prescribes — nothing runs under `swift test`, whose
// metallib is unreliable). Exits nonzero if any check fails.
//
//   swift run -c release trellis2-gate

import Foundation
import MLXToolKit
import MLXServeConformance
import Trellis2Kit

var failures = 0
func check(_ name: String, _ passed: Bool, _ note: String = "") {
    print("\(passed ? "✅" : "❌") \(name)\(note.isEmpty ? "" : " — \(note)")")
    if !passed { failures += 1 }
}
func section(_ s: String) { print("\n=== \(s) ===") }

let manifest = Trellis2Package.manifest

// ---------------------------------------------------------------------------------------------
section("C0–C13 static conformance")

// C0 — contract version declared and matches the engine's current contract.
check("C0 contract version",
      manifest.contractVersion == ContractVersion.current,
      "declares \(manifest.contractVersion), engine current \(ContractVersion.current)")

// C1 — exactly one capability surface, and it's imageTo3D; capabilities derived from surfaces.
let caps = manifest.capabilities
check("C1 capability surface", caps == [.imageTo3D],
      "surfaces=\(manifest.surfaces.map(\.name)) capabilities=\(caps.map(\.rawValue))")

// C2/C11 — the surface descriptor is well-formed (hand-tuned summary, canonical params, modes).
if let surf = manifest.surfaces.first {
    let ok = surf.capability == .imageTo3D
        && !surf.summary.isEmpty
        && surf.parameters.contains { $0.name == "image" && $0.required }
        && surf.supportedModes.contains(ImageTo3DContract.res512)
    check("C2/C11 descriptor well-formed", ok,
          "\(surf.parameters.count) params, modes=\(surf.supportedModes.map(\.rawValue))")
} else {
    check("C2/C11 descriptor well-formed", false, "no surface")
}

// C7 — weight license present and admitted by the default .permissiveOnly gate.
let gate = LicensePolicy.permissiveOnly.evaluate(manifest.license)
check("C7 weight license (permissiveOnly)", gate.isAdmitted,
      "weight=\(manifest.license.weightLicense) → \(gate)")

// C8 — port-code license is the second, independently-gated layer.
check("C8 port-code license", LicensePolicy.permissiveOnly.admits(manifest.license.portCodeLicense),
      "portCode=\(manifest.license.portCodeLicense)")

// C10 — cost-to-run declared: footprints (split, nonzero), backend, chip floor, OS floor.
let reqs = manifest.requirements
let fp = reqs.footprints.first
let footprintOK = fp != nil && fp!.residentBytes > 0 && fp!.peakActivationBytes > 0
check("C10 footprint split declared", footprintOK,
      fp.map { "resident \(String(format: "%.1f", Double($0.residentBytes)/1e9)) GB + "
             + "peakActivation \(String(format: "%.1f", Double($0.peakActivationBytes)/1e9)) GB" } ?? "none")
check("C10 required backends", reqs.requiredBackends.contains(.metalGPU),
      "\(reqs.requiredBackends.map(\.rawValue))")
check("C10 chip floor", reqs.chipFloor != nil, "\(reqs.chipFloor.map { "\($0)" } ?? "nil")")
check("C10 OS floor", reqs.os.minMacOS != nil, "minMacOS=\(reqs.os.minMacOS.map { "\($0)" } ?? "nil")")

// C6 — specialty declared (model-level selection metadata).
check("C6 specialty", manifest.specialties.contains { $0.specialty == Specialty(rawValue: "3d-generation") },
      "\(manifest.specialties.map { $0.specialty.rawValue })")

// Provenance tier (process gate) — tier 3 (multi-component pipeline).
check("provenance tier", manifest.provenance.tier == 3 && manifest.provenance.sourceRepo == "microsoft/TRELLIS.2-4B",
      "\(manifest.provenance.sourceRepo) tier \(manifest.provenance.tier)")

// C13 — inversion of control: registration builds via PackageRegistration.of, engine constructs.
let registration = Trellis2Package.registration
check("C13 registration (IoC)", registration.manifest.capabilities == [.imageTo3D],
      "PackageRegistration.of(Trellis2Package.self)")

// ---------------------------------------------------------------------------------------------
section("MAT-1..5 (MaterializationConformance, offline)")

func stagedSnapshotDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appending(path: "trellis2-mat-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    for f in Trellis2Configuration.probeFiles {
        FileManager.default.createFile(atPath: dir.appending(path: f).path, contents: Data("probe".utf8))
    }
    return dir
}

do {
    let fresh = Trellis2Configuration()
    let dir = try stagedSnapshotDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let satisfied = Trellis2Configuration(weightsRootOverride: dir)
    let report = MaterializationConformance.check(freshConfiguration: fresh,
                                                  satisfiedConfiguration: satisfied)
    print(report.summary)
    check("MAT gate (MAT-1..5)", report.passed)
} catch {
    check("MAT gate (MAT-1..5)", false, "\(error)")
}

// ---------------------------------------------------------------------------------------------
section("CAN-1..3 (CancellationConformance, offline)")

let package = Trellis2Package(configuration: Trellis2Configuration())
let canReport = await CancellationConformance.checkRun(
    package: package,
    request: ImageTo3DRequest(image: Image(format: .png, data: Data())))
print(canReport.summary)
check("CAN-1/2 pre-cancelled run()", canReport.passed)

check("CAN-3 long-run implied", CancellationConformance.longRunImplied(by: manifest),
      "imageTo3D is a long-run capability")
let cadence = CancellationConformance.checkCadence(
    manifest: manifest,
    posture: .cadence([
        // FlowEulerSampler bails per Euler step across ALL diffusion stages (SS + shape + tex).
        .init(phase: .denoise, unit: .step),
        // The sparse VAE decoders bail per conv block; the pipeline early-outs at every stage seam,
        // and run() rethrows the CancellationError unchanged after generate() returns.
        .init(phase: .decode, unit: .layer),
    ]))
print(cadence.summary)
check("CAN-3 checkpoint cadence", cadence.passed)

// ---------------------------------------------------------------------------------------------
section("Result")
if failures == 0 {
    print("✅ ALL GATES PASSED (C0–C13 static + MAT-1..5 + CAN-1..3)")
} else {
    print("❌ \(failures) gate check(s) FAILED")
    exit(1)
}
