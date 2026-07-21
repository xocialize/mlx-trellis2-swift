import Foundation
import MLX

/// FlowEulerGuidanceIntervalSampler — matches trellis2. Deterministic:
///   t_seq = linspace(1,0,steps+1), warped t' = r·t/(1+(r-1)·t) (rescale_t=r);
///   per step: CFG velocity (inside guidance interval) + optional CFG-rescale,
///   Euler update x -= (t - t_prev)·v. Timestep fed to model is 1000·t.
/// Shared core drives both the dense SS stage and the sparse SLat stages.
public enum FlowEulerSampler {
    static func predToXstart(_ x: MLXArray, _ t: Float, _ v: MLXArray, sigmaMin: Float) -> MLXArray {
        (1 - sigmaMin) * x - (sigmaMin + (1 - sigmaMin) * t) * v
    }
    static func xstartToPred(_ x: MLXArray, _ t: Float, _ x0: MLXArray, sigmaMin: Float) -> MLXArray {
        ((1 - sigmaMin) * x - x0) / (sigmaMin + (1 - sigmaMin) * t)
    }
    /// Biased std over all elements (batch=1) — matches both torch.std (unbiased factor
    /// cancels in the pos/cfg ratio) and SparseTensor.std (sqrt(E[x²]-E[x]²)).
    static func stdAll(_ x: MLXArray) -> MLXArray {
        let f = x.reshaped([-1]); let m = f.mean()
        return MLX.sqrt(((f - m) * (f - m)).mean())
    }

    /// Core Euler+CFG loop. `predict(x, t1000, useCond)` returns the model velocity for
    /// the current sample using cond (useCond=true) or neg_cond (false).
    /// `negEvery` — CFG caching: recompute vNeg only on every negEvery-th CFG step and
    /// reuse the cached one between (vPos + the CFG combination stay per-step). 1 = off
    /// (upstream-exact). vNeg drifts slowly across adjacent t, so reuse approximates far
    /// more gently than dropping guidance outright at the schedule edges.
    static func runLoop(
        noise: MLXArray, steps: Int, guidanceStrength: Float, guidanceRescale: Float,
        guidanceInterval: (Float, Float), rescaleT: Float, sigmaMin: Float,
        negEvery: Int = 1,
        predict: (MLXArray, MLXArray, Bool) -> MLXArray
    ) -> MLXArray {
        // Compute the schedule in Double to match numpy's float64 (the interval boundary
        // — e.g. t=0.6 at rescale_t=3, steps=12 — is a knife-edge that float32 can flip,
        // swinging CFG 7.5×↔1× for that step).
        let rT = Double(rescaleT), g0 = Double(guidanceInterval.0), g1 = Double(guidanceInterval.1)
        var tSeq = [Double]()
        for i in 0...steps {
            let lt = 1.0 - Double(i) / Double(steps)                  // linspace(1,0,steps+1)
            tSeq.append(rT * lt / (1 + (rT - 1) * lt))
        }
        var x = noise
        var cachedVNeg: MLXArray? = nil
        var cfgStep = 0
        for i in 0..<steps {
            let tD = tSeq[i], tPrevD = tSeq[i + 1]
            let t = Float(tD), tPrev = Float(tPrevD)
            let t1000 = MLXArray([1000.0 * t])
            let inInterval = (tD >= g0 && tD <= g1)
            let s = inInterval ? guidanceStrength : 1.0
            var v: MLXArray
            if s == 1.0 {
                v = predict(x, t1000, true)
            } else if s == 0.0 {
                v = predict(x, t1000, false)
            } else {
                let vPos = predict(x, t1000, true)
                let vNeg: MLXArray
                if negEvery > 1, let cached = cachedVNeg, cfgStep % negEvery != 0 {
                    vNeg = cached                    // CFG cache hit: reuse last real vNeg
                } else {
                    vNeg = predict(x, t1000, false)
                    cachedVNeg = vNeg
                }
                cfgStep += 1
                v = s * vPos + (1 - s) * vNeg
                if guidanceRescale > 0 {
                    let x0Pos = predToXstart(x, t, vPos, sigmaMin: sigmaMin)
                    let x0Cfg = predToXstart(x, t, v, sigmaMin: sigmaMin)
                    let x0Res = x0Cfg * (stdAll(x0Pos) / stdAll(x0Cfg))
                    let x0 = guidanceRescale * x0Res + (1 - guidanceRescale) * x0Cfg
                    v = xstartToPred(x, t, x0, sigmaMin: sigmaMin)
                }
            }
            x = x - (t - tPrev) * v
            MLX.eval(x)
        }
        return x
    }

    /// Dense sparse-structure stage. `model(x, t1000, cond, phases) -> v`.
    public static func sampleSS(
        model: SparseStructureFlowModel, noise: MLXArray, cond: MLXArray, negCond: MLXArray, phases: MLXArray,
        steps: Int = 12, guidanceStrength: Float = 7.5, guidanceRescale: Float = 0.7,
        guidanceInterval: (Float, Float) = (0.6, 1.0), rescaleT: Float = 5.0, sigmaMin: Float = 1e-5
    ) -> MLXArray {
        runLoop(noise: noise, steps: steps, guidanceStrength: guidanceStrength, guidanceRescale: guidanceRescale,
                guidanceInterval: guidanceInterval, rescaleT: rescaleT, sigmaMin: sigmaMin) { x, t1000, useCond in
            model(x, t: t1000, cond: useCond ? cond : negCond, phases: phases)
        }
    }

    /// Sparse SLat stage (shape or tex). Operates on voxel feats at fixed `coords`.
    /// `concatCond` (feats, same coords) is the tex stage's shape conditioning.
    /// Returns the sampled feats [T, C].
    public static func sampleSLat(
        model: SLatFlowModel, noiseFeats: MLXArray, coords: MLXArray, cond: MLXArray, negCond: MLXArray,
        concatCond: MLXArray? = nil, steps: Int = 12, guidanceStrength: Float = 7.5, guidanceRescale: Float = 0.5,
        guidanceInterval: (Float, Float) = (0.6, 1.0), rescaleT: Float = 3.0, sigmaMin: Float = 1e-5,
        negEvery: Int = 1
    ) -> MLXArray {
        runLoop(noise: noiseFeats, steps: steps, guidanceStrength: guidanceStrength, guidanceRescale: guidanceRescale,
                guidanceInterval: guidanceInterval, rescaleT: rescaleT, sigmaMin: sigmaMin, negEvery: negEvery) { x, t1000, useCond in
            model(SparseTensor(feats: x, coords: coords), t: t1000, cond: useCond ? cond : negCond, concatCond: concatCond).feats
        }
    }
}
