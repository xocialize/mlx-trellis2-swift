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
import TRELLIS2   // Pixal3D camera maths (host-side Double, no MLX kernels)

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

/// A minimal VALID safetensors byte stream (8-byte LE header length + JSON header) carrying the
/// canonical original key — so `snapshotPresent`'s content guard (which reads struct_flow's header)
/// treats the staged snapshot as genuine, not a renamed-key stale one. Other files stay stubs (the
/// guard only inspects struct_flow).
func safetensorsStub(withKey key: String) -> Data {
    let headerJSON = Data("{\"\(key)\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"__metadata__\":{}}".utf8)
    var out = Data()
    withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) { out.append(contentsOf: $0) }
    out.append(headerJSON)
    out.append(Data(count: 4))   // the one f32 tensor's bytes
    return out
}

func stagedSnapshotDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appending(path: "trellis2-mat-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    for f in Trellis2Configuration.probeFiles {
        let contents = f == "struct_flow.safetensors"
            ? safetensorsStub(withKey: Trellis2Configuration.canonicalKey)
            : Data("probe".utf8)
        FileManager.default.createFile(atPath: dir.appending(path: f).path, contents: contents)
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
// Pixal3D lane (AB-T-0485 P3): two sources; satisfied only when both snapshots are present.
do {
    let stockDir = try stagedSnapshotDir()
    let pixDir = FileManager.default.temporaryDirectory.appending(path: "pixal3d-mat-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: pixDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: stockDir); try? FileManager.default.removeItem(at: pixDir) }
    for f in Trellis2Configuration.pixal3dProbeFiles(multiView: false) {
        FileManager.default.createFile(atPath: pixDir.appending(path: f).path,
            contents: f == "pixal3d_struct_flow.safetensors" ? safetensorsStub(withKey: Trellis2Configuration.pixal3dCanonicalKey) : Data("probe".utf8))
    }
    let fresh = Trellis2Configuration.pixal3dLane(ImageTo3DContract.res1024)
    let half = Trellis2Configuration.pixal3dLane(ImageTo3DContract.res1024, weightsRootOverride: stockDir)
    let both = Trellis2Configuration.pixal3dLane(ImageTo3DContract.res1024, weightsRootOverride: stockDir, pixal3dWeightsDir: pixDir)
    let report = MaterializationConformance.check(freshConfiguration: fresh, satisfiedConfiguration: both)
    print(report.summary)
    check("MAT gate, Pixal3D lane (MAT-1..5)", report.passed)
    check("Pixal3D lane with only the stock snapshot reports the Pixal3D source missing",
          half.missingWeightSources(storeRoot: nil).map(\.repo) == [Trellis2Configuration.pixal3dRepo])
    check("Pixal3D lane prewarms its own files, not the stock SS/shape flows",
          both.prewarmPaths.contains { $0.lastPathComponent == "pixal3d_shape_flow_1024.safetensors" }
          && !both.prewarmPaths.contains { $0.lastPathComponent == "struct_flow.safetensors" })
} catch {
    check("MAT gate, Pixal3D lane", false, "\(error)")
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
section("FIT (FootprintConformance) — manifest + one lane per tier (AB-A-0262)")

let fitManifest = FootprintConformance.check(manifest: manifest)
print(fitManifest.summary)
check("FIT manifest", fitManifest.passed)
var laneTotals: [UInt64] = []
for tier in [ImageTo3DContract.res512, ImageTo3DContract.res1024, ImageTo3DContract.res1536] {
    let lane = Trellis2Configuration.lane(tier)
    let fit = FootprintConformance.check(manifest: manifest, configuration: lane)
    print(fit.summary)
    let total = (lane.residentBytesHint ?? 0) + (lane.peakActivationBytesHint ?? 0)
    laneTotals.append(total)
    check("FIT lane \(tier.rawValue)", fit.passed,
          String(format: "resident %.0f + activation %.0f = %.0f GB, HR token bound %@",
                 Double(lane.residentBytesHint ?? 0) / 1e9, Double(lane.peakActivationBytesHint ?? 0) / 1e9,
                 Double(total) / 1e9, lane.hrTokenBound.map(String.init) ?? "n/a"))
}
check("lane footprints rise with the tier", laneTotals == laneTotals.sorted() && Set(laneTotals).count == 3)
let worst = manifest.requirements.footprints.map { $0.residentBytes + $0.peakActivationBytes }.max() ?? 0
check("manifest footprint ≥ the stock res1536 lane (largest-footprint fallback never under-reserves)",
      worst >= (laneTotals.last ?? 0))
// A lane refuses a request above its tier before weights (run() checks it ahead of notLoaded).
do {
    let lanePkg = Trellis2Package(configuration: .lane(ImageTo3DContract.res1024))
    _ = try await lanePkg.run(ImageTo3DRequest(image: Image(format: .png, data: Data()),
                                               mode: ImageTo3DContract.res1536))
    check("res1024 lane refuses a res1536 request", false, "run() did not throw")
} catch let PackageError.unsupportedRequestFeature(msg) {
    check("res1024 lane refuses a res1536 request", true, msg)
} catch {
    check("res1024 lane refuses a res1536 request", false, "threw \(error)")
}
let small = Trellis2Configuration.lane(ImageTo3DContract.res512)
check("res512 lane at the default maxAtlasSize (1024) = 34 GB, v0.11.0's charge",
      (small.residentBytesHint ?? 0) + (small.peakActivationBytesHint ?? 0) == 34_000_000_000)
for (atlas, total) in [(2048, 38_000_000_000 as UInt64), (4096, 48_000_000_000)] {
    let big = Trellis2Configuration.lane(ImageTo3DContract.res512, maxAtlasSize: atlas)
    let fit = FootprintConformance.check(manifest: manifest, configuration: big)
    check("res512 lane at maxAtlasSize \(atlas) = \(total / 1_000_000_000) GB", fit.passed
          && (big.residentBytesHint ?? 0) + (big.peakActivationBytesHint ?? 0) == total)
    let cascade = Trellis2Configuration.lane(ImageTo3DContract.res1536, maxAtlasSize: atlas)
    check("res1536 lane charge does not move with maxAtlasSize \(atlas)",
          cascade.peakActivationBytesHint == Trellis2Configuration.lane(ImageTo3DContract.res1536).peakActivationBytesHint)
}
do {
    _ = try Trellis2RequestOptions(metaData: ["atlasSize": .int(2048)], configuration: small)
    check("maxAtlasSize 1024 refuses atlasSize 2048", false, "accepted")
} catch PackageError.unsupportedRequestFeature {
    check("maxAtlasSize 1024 refuses atlasSize 2048", true)
} catch { check("maxAtlasSize 1024 refuses atlasSize 2048", false, "\(error)") }
let roaming = Trellis2Configuration(defaultMode: ImageTo3DContract.res512)
check("no tierCeiling = the pre-0.11 behaviour (charged for defaultMode)",
      roaming.residentBytesHint == Trellis2Configuration.lane(ImageTo3DContract.res512).residentBytesHint)

// Pixal3D lanes (AB-T-0485 P3): one per tier, measured footprints, a backer of the same capability.
var pixTotals: [UInt64] = []
for tier in [ImageTo3DContract.res1024, ImageTo3DContract.res1536] {
    let lane = Trellis2Configuration.pixal3dLane(tier)
    let fit = FootprintConformance.check(manifest: manifest, configuration: lane)
    let total = (lane.residentBytesHint ?? 0) + (lane.peakActivationBytesHint ?? 0)
    pixTotals.append(total)
    check("FIT Pixal3D lane \(tier.rawValue)", fit.passed && lane.isPixal3DLane,
          String(format: "resident %.0f + activation %.0f = %.0f GB, HR token bound %@",
                 Double(lane.residentBytesHint ?? 0) / 1e9, Double(lane.peakActivationBytesHint ?? 0) / 1e9,
                 Double(total) / 1e9, lane.hrTokenBound.map(String.init) ?? "n/a"))
}
do {
    var lean = Trellis2Configuration.pixal3dLane(ImageTo3DContract.res1024); lean.flowWeightsBF16 = true
    let fit = FootprintConformance.check(manifest: manifest, configuration: lean)
    let total = (lean.residentBytesHint ?? 0) + (lean.peakActivationBytesHint ?? 0)
    check("FIT Pixal3D lane res1024, bf16-resident flows (11 GB less resident, same activation)", fit.passed
          && lean.residentBytesHint == Trellis2Footprint.pixal3dResidentCascadeBF16 && total < pixTotals[0],
          String(format: "%.1f GB", Double(total) / 1e9))
    var stockBF16 = Trellis2Configuration.lane(ImageTo3DContract.res1024); stockBF16.flowWeightsBF16 = true
    check("a stock lane with bf16 flows keeps its fp32 charge (unmeasured)",
          stockBF16.residentBytesHint == Trellis2Configuration.lane(ImageTo3DContract.res1024).residentBytesHint)
}
check("Pixal3D lane footprints rise with the tier", pixTotals == pixTotals.sorted() && Set(pixTotals).count == 2)
check("manifest footprint ≥ every lane, stock and Pixal3D", (laneTotals + pixTotals).allSatisfy { $0 <= worst }
      && worst == max(laneTotals.last ?? 0, pixTotals.last ?? 0))
check("a Pixal3D lane declares both weight sources; a stock lane only the snapshot",
      Trellis2Configuration.pixal3dLane(ImageTo3DContract.res1024).weightSources.map(\.repo)
        == [Trellis2Configuration.consolidatedRepo, Trellis2Configuration.pixal3dRepo]
      && Trellis2Configuration.lane(ImageTo3DContract.res1024).weightSources.count == 1)
// P3 res512 decision: no Pixal3D res512 path — a Pixal3D request at res512 is refused before weights.
do {
    let stockLane = Trellis2Package(configuration: .lane(ImageTo3DContract.res512))
    _ = try await stockLane.run(ImageTo3DRequest(image: Image(format: .png, data: Data()),
                                                 mode: ImageTo3DContract.res512, metaData: ["pixal3d": .bool(true)]))
    check("Pixal3D at res512 is refused (res512 stays stock)", false, "run() did not throw")
} catch let PackageError.unsupportedRequestFeature(msg) {
    check("Pixal3D at res512 is refused (res512 stays stock)", true, msg)
} catch {
    check("Pixal3D at res512 is refused (res512 stays stock)", false, "threw \(error)")
}

// ---------------------------------------------------------------------------------------------
section("Per-request options from metaData (AB-A-0263)")

let base = Trellis2Configuration(maxAtlasSize: 4096, seed: 3, texture: true, decimateFaces: 300_000, yUpOutput: false)
do {
    let none = try Trellis2RequestOptions(metaData: [:], configuration: base)
    check("no keys = the configuration's job", none == Trellis2RequestOptions(configuration: base)
          && none.seed == 3 && none.steps == 12 && none.texture && none.decimateFaces == 300_000
          && none.atlasSize == 1024 && !none.yUp && !none.pbrMaps)
    let nulls = try Trellis2RequestOptions(metaData: ["seed": .null, "yUp": .null], configuration: base)
    check("null keys fall back", nulls == none)
    let all = try Trellis2RequestOptions(metaData: [
        "seed": .string("18446744073709551615"), "steps": .int(10), "texture": .bool(false),
        "decimateFaces": .double(50_000), "atlasSize": .int(4096), "yUp": .bool(true), "pbrMaps": .bool(true),
    ], configuration: base)
    check("every key overrides", all.seed == UInt64.max && all.steps == 10 && !all.texture
          && all.decimateFaces == 50_000 && all.atlasSize == 4096 && all.yUp && all.pbrMaps)
} catch {
    check("options parse", false, "\(error)")
}
for (key, bad) in [("atlasSize", MetaValue.int(3000)), ("seed", .int(-1)), ("steps", .int(0)),
                   ("yUp", .string("true")), ("decimateFaces", .double(1.5)), ("texture", .int(1))] {
    do {
        _ = try Trellis2RequestOptions(metaData: [key: bad], configuration: base)
        check("refuses \(key) = \(bad)", false, "accepted")
    } catch PackageError.unsupportedRequestFeature {
        check("refuses \(key) = \(bad)", true)
    } catch {
        check("refuses \(key) = \(bad)", false, "threw \(error)")
    }
}

// ---------------------------------------------------------------------------------------------
section("Pixal3D request options (AB-T-0485)")
do {
    let none = try Trellis2RequestOptions(metaData: [:], configuration: base)
    check("Pixal3D off by default; fov 20°, frame front, no cameras",
          !none.pixal3d && none.pixal3dFovDegrees == 20 && none.pixal3dFrame == "front" && none.pixal3dCameras == nil)
    let front: [MetaValue] = [1, 0, 0, 0, 0, 0, -1, -2.8356, 0, 1, 0, 0, 0, 0, 0, 1].map { .double($0) }
    let side: [MetaValue] = [0, 0, 1, 2.8356, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1].map { .double($0) }
    let on = try Trellis2RequestOptions(metaData: [
        "pixal3d": .bool(true), "pixal3dFovDegrees": .int(30), "pixal3dFrame": .string("world"),
        "pixal3dCameras": .array([.array(front), .array(side)]),
    ], configuration: base)
    check("every Pixal3D key overrides", on.pixal3d && on.pixal3dFovDegrees == 30 && on.pixal3dFrame == "world"
          && on.pixal3dCameras?.count == 2 && on.pixal3dCameras?[1][3] == 2.8356)
    let only = try Trellis2RequestOptions(metaData: [:], configuration: Trellis2Configuration(pixal3dOnly: true))
    check("a pixal3dOnly registration defaults requests to Pixal3D", only.pixal3d)
    // camera maths: distance closed form, view 0 snaps to the canonical front, front→world is rigid
    let cam = Pixal3DCamera()
    check("fov 20° → distance 2.8356 (0.5/tan(fov/2))", abs(cam.distance - 2.835641) < 1e-5)
    let tilted = Mat4([[1, 0, 0, 0], [0, 0.8, -0.6, -2.0], [0, 0.6, 0.8, 1.5], [0, 0, 0, 1]])
    let calc = Pixal3DCamera.calcMats([Pixal3DCamera(c2w: tilted), Pixal3DCamera(c2w: tilted)])
    let F = Mat4.front(distance: tilted.translationNorm)
    check("calc_mat of view 0 is the canonical front F(|t0|)", zip(calc[0].m, F.m).allSatisfy { abs($0 - $1) < 1e-9 })
    let W = Pixal3DCamera.frontToWorld([Pixal3DCamera(c2w: tilted)])
    check("front→world maps F back onto C0", zip((W * F).m, tilted.m).allSatisfy { abs($0 - $1) < 1e-9 })
    // turnaround preset: front = F(d), left = upstream's azimuth-90 example, d = 1.1 · 0.5/tan(10°) = 3.1192
    let rig = Pixal3DCamera.turnaround([.front, .left, .back, .right])
    let d = rig[0].c2w!.translationNorm
    let exL = Mat4([[0, 0, 1, d], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    check("turnaround: front is F(d), left is upstream's azimuth-90 matrix, d = 3.1192 (pad 1.1, 20°)",
          abs(d - 3.11921) < 1e-4 && zip(rig[0].c2w!.m, Mat4.front(distance: d).m).allSatisfy { abs($0 - $1) < 1e-9 }
          && zip(rig[1].c2w!.m, exL.m).allSatisfy { abs($0 - $1) < 1e-9 })
    let typed = try Pixal3DPackage.turnaroundCameras(["front", "back"], fovXDegrees: 20)
    check("turnaroundCameras gives valid contract cameras", typed.count == 2 && typed.allSatisfy { $0.shapeProblem == nil })
} catch {
    check("Pixal3D options parse", false, "\(error)")
}
for (key, bad) in [("pixal3d", MetaValue.string("yes")), ("pixal3dFovDegrees", .int(0)), ("pixal3dFovDegrees", .double(150)),
                   ("pixal3dFrame", .string("up")), ("pixal3dCameras", .array([.array([.int(1), .int(0)])])),
                   ("pixal3dCameras", .array([]))] {
    do {
        _ = try Trellis2RequestOptions(metaData: [key: bad], configuration: base)
        check("refuses \(key) = \(bad)", false, "accepted")
    } catch PackageError.unsupportedRequestFeature {
        check("refuses \(key) = \(bad)", true)
    } catch {
        check("refuses \(key) = \(bad)", false, "threw \(error)")
    }
}

// ---------------------------------------------------------------------------------------------
section("Pixal3DPackage — its own lane (contract 1.64.0 cameras, AB-A-0284 / AB-D-0230)")
let pm = Pixal3DPackage.manifest
check("P3D C0 contract version", pm.contractVersion == ContractVersion.current, "\(pm.contractVersion)")
check("P3D C1 one imageTo3D surface", pm.capabilities == [.imageTo3D], "\(pm.surfaces.map(\.name))")
check("P3D C7/C8 licences admitted", LicensePolicy.permissiveOnly.evaluate(pm.license).isAdmitted
      && LicensePolicy.permissiveOnly.admits(pm.license.portCodeLicense))
check("P3D provenance is xocialize/pixal3d-mlx (distinct derived id from the stock package)",
      pm.provenance.sourceRepo == Trellis2Configuration.pixal3dRepo && pm.provenance.sourceRepo != manifest.provenance.sourceRepo)
if let surf = pm.surfaces.first {
    check("P3D surface advertises only the tiers Pixal3D honors",
          surf.supportedModes == [ImageTo3DContract.res1024, ImageTo3DContract.res1536], "\(surf.supportedModes.map(\.rawValue))")
    let ctl = surf.imageTo3DControls
    check("P3D controls declared and coherent (declarationProblems.isEmpty)",
          ctl == Pixal3DPackage.controls && (ctl?.declarationProblems.isEmpty ?? false),
          "\(ctl.map { $0.features.map(\.rawValue).sorted() } ?? [])")
    check("P3D surface advertises cameras and frame", surf.parameters.contains { $0.name == "cameras" }
          && surf.parameters.contains { $0.name == "frame" }, "\(surf.parameters.map(\.name))")
}
check("stock surface declares no camera controls", manifest.surfaces.first?.imageTo3DControls == nil)
let pfp = pm.requirements.footprints.first
check("P3D manifest footprint = the Pixal3D res1536 lane",
      pfp.map { $0.residentBytes + $0.peakActivationBytes } == pixTotals.last)
check("P3D registration (IoC)", Pixal3DPackage.registration.manifest.capabilities == [.imageTo3D])
check("P3D lane declares both weight sources",
      Trellis2Configuration.pixal3dLane(ImageTo3DContract.res1024).weightSources.count == 2)
// typed-field refusals (what the engine pre-flight does; the packages check the same before weights)
func refusalOf(_ pkg: any ModelPackage, _ req: ImageTo3DRequest) async -> String? {
    do { _ = try await pkg.run(req); return nil }
    catch let PackageError.unsupportedRequestFeature(m) { return m }
    catch { return "other: \(error)" }
}
let img = Image(format: .png, data: Data())
let stockPkg = Trellis2Package(configuration: .lane(ImageTo3DContract.res1024))
let pixPkg = Pixal3DPackage(configuration: .pixal3dLane(ImageTo3DContract.res1024))
let r1 = await refusalOf(stockPkg, ImageTo3DRequest(image: img, mode: ImageTo3DContract.res1024,
                                                    cameras: [ImageTo3DCamera(fovXDegrees: 20)]))
check("stock lane refuses typed cameras", r1?.contains("fovXDegrees") == true, r1 ?? "accepted")
let r2 = await refusalOf(pixPkg, ImageTo3DRequest(image: img, mode: ImageTo3DContract.res1024,
                                                  cameras: [ImageTo3DCamera(fovXDegrees: 150)]))
check("Pixal3D refuses a fov outside its declared range", r2?.contains("150") == true, r2 ?? "accepted")
let r3 = await refusalOf(pixPkg, ImageTo3DRequest(image: img, mode: ImageTo3DContract.res512))
check("Pixal3D refuses res512", r3?.contains("res512") == true, r3 ?? "accepted")
let r4 = await refusalOf(pixPkg, ImageTo3DRequest(image: img, mode: ImageTo3DContract.res1024, frame: .world))
check("Pixal3D refuses frame world without poses (contract shape)", r4 != nil && !(r4!.hasPrefix("other")), r4 ?? "accepted")
let r5 = await refusalOf(pixPkg, ImageTo3DRequest(image: img, mode: ImageTo3DContract.res1024,
                                                  cameras: [ImageTo3DCamera(fovXDegrees: 20)]))
check("Pixal3D accepts a single-view fov (then needs weights: notLoaded)", r5?.contains("notLoaded") == true
      || r5?.hasPrefix("other") == true, r5 ?? "ran")

// ---------------------------------------------------------------------------------------------
section("Result")
if failures == 0 {
    print("✅ ALL GATES PASSED (C0–C13 static + MAT-1..5 + CAN-1..3 + FIT lanes + request options + Pixal3D options + Pixal3DPackage)")
} else {
    print("❌ \(failures) gate check(s) FAILED")
    exit(1)
}
