/* HeartRisk 3D - in-browser inference engine.
 * Runs the exported model bundle (artifacts/model_bundle.json) exactly like the Python pipeline:
 *   standardise -> linear / gradient-boosted trees raw score -> Platt calibration
 *   -> Mondrian conformal set -> bootstrap band -> attribution -> OOD check.
 * Works in the browser (window.HeartEngine) and in Node (module.exports) for the parity test.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.HeartEngine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const LABELS = {
    CAD: ["No CAD", "CAD"],
    LAD: ["Normal", "Stenotic"],
    LCX: ["Normal", "Stenotic"],
    RCA: ["Normal", "Stenotic"],
  };

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  function percentile(arr, q) {
    const a = arr.slice().sort((x, y) => x - y);
    if (!a.length) return NaN;
    const pos = (a.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return a[lo] + (a[hi] - a[lo]) * (pos - lo);
  }

  // ---- raw model scores -------------------------------------------------
  function standardise(m, x) {
    const z = new Array(x.length);
    for (let i = 0; i < x.length; i++) z[i] = (x[i] - m.mean[i]) / m.scale[i];
    return z;
  }

  function treeValue(t, z) {
    let n = 0;
    while (t.l[n] !== -1) n = Math.fround(z[t.f[n]]) <= t.t[n] ? t.l[n] : t.r[n];
    return t.v[n];
  }

  function rawFromZ(m, z) {
    if (m.kind === "linear") {
      let s = m.intercept;
      for (let i = 0; i < z.length; i++) s += m.coef[i] * z[i];
      return s;
    }
    let s = 0;
    for (const t of m.trees) s += treeValue(t, z);
    return m.init + m.lr * s;
  }

  function rawScore(m, x) {
    return rawFromZ(m, standardise(m, x));
  }

  // Attribution in log-odds relative to the training mean (exact SHAP for the linear model,
  // occlusion-to-mean for gradient-boosted trees).
  function attribution(m, x) {
    const z = standardise(m, x);
    if (m.kind === "linear") return z.map((zi, i) => m.coef[i] * zi);
    const base = rawFromZ(m, z);
    return z.map((_, i) => {
      const zz = z.slice();
      zz[i] = 0;
      return base - rawFromZ(m, zz);
    });
  }

  // ---- engine -----------------------------------------------------------
  function Engine(bundle) {
    this.b = bundle;
    this.meta = {};
    bundle.features.forEach((f) => (this.meta[f.name] = f));
  }

  Engine.prototype.setKey = function (available) {
    const on = this.b.optional_modalities.filter((m) => available && available[m]);
    return on.length ? on.join("+") : "bedside";
  };

  Engine.prototype.vector = function (cfg, values) {
    const imputed = [];
    const x = cfg.features.map((name) => {
      const v = values[name];
      if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) {
        imputed.push(name);
        return this.meta[name].median;
      }
      return Number(v);
    });
    return { x, imputed };
  };

  Engine.prototype.ood = function (values) {
    const o = this.b.ood;
    const outOfRange = [];
    for (const [name, [lo, hi]] of Object.entries(o.ranges)) {
      const v = Number(values[name]);
      if (!Number.isNaN(v) && (v < lo || v > hi)) outOfRange.push({ name, value: v, range: [lo, hi] });
    }
    const z = o.features.map((n, i) => (Number(values[n] ?? this.meta[n].median) - o.mean[i]) / o.scale[i] - o.location[i]);
    let d2 = 0;
    for (let i = 0; i < z.length; i++) for (let j = 0; j < z.length; j++) d2 += z[i] * o.precision[i][j] * z[j];
    const distance = Math.sqrt(Math.max(d2, 0));
    return { outOfRange, distance, threshold: o.threshold, isOOD: distance > o.threshold || outOfRange.length > 0 };
  };

  Engine.prototype.predictTarget = function (cfg, tgt, x) {
    const t = cfg.targets[tgt];
    const [a, b] = t.platt;
    const p = sigmoid(a * rawScore(t.model, x) + b);
    const boots = t.boot.map((m) => sigmoid(a * rawScore(m, x) + b));
    const all = boots.concat([p]);
    const q = t.conformal;
    // Mondrian (class-conditional) set, plus the most probable label so the set can never
    // exclude the label the model itself prefers. Adding labels only enlarges the set,
    // so the coverage guarantee still holds.
    const in1 = 1 - p <= q.q1 || p >= 0.5;
    const in0 = p <= q.q0 || p < 0.5;
    const set = [];
    if (in0) set.push(LABELS[tgt][0]);
    if (in1) set.push(LABELS[tgt][1]);
    const contrib = attribution(t.model, x);
    return {
      p,
      band: [percentile(all, 0.1), percentile(all, 0.9)],
      set,
      setKind: in0 && in1 ? "both" : in1 ? "positive" : in0 ? "negative" : "empty",
      family: t.family,
      contrib,
    };
  };

  Engine.prototype.predict = function (values, available) {
    const key = this.setKey(available);
    const cfg = this.b.configs[key];
    const { x, imputed } = this.vector(cfg, values);
    const out = { set: key, modalities: cfg.modalities, imputed, targets: {} };
    for (const tgt of this.b.targets) {
      const r = this.predictTarget(cfg, tgt, x);
      const drivers = cfg.features.map((name, i) => ({
        name,
        label: this.meta[name].label,
        modality: this.meta[name].modality,
        value: x[i],
        contrib: r.contrib[i],
      }));
      drivers.sort((u, v) => Math.abs(v.contrib) - Math.abs(u.contrib));
      const byMod = {};
      let absTotal = 0;
      for (const d of drivers) {
        byMod[d.modality] = byMod[d.modality] || { signed: 0, abs: 0 };
        byMod[d.modality].signed += d.contrib;
        byMod[d.modality].abs += Math.abs(d.contrib);
        absTotal += Math.abs(d.contrib);
      }
      for (const k in byMod) byMod[k].share = absTotal ? byMod[k].abs / absTotal : 0;
      delete r.contrib;
      out.targets[tgt] = Object.assign(r, { drivers, modalityContribution: byMod });
    }
    const T = out.targets;
    out.expectedVessels = T.LAD.p + T.LCX.p + T.RCA.p;
    const maxV = Math.max(T.LAD.p, T.LCX.p, T.RCA.p);
    out.coherence = {
      ok: !(maxV > T.CAD.p + 0.25 || (T.CAD.p > 0.8 && maxV < 0.3)),
      note: "CAD in this dataset means at least one of LAD/LCX/RCA has >=50% stenosis.",
    };
    out.ood = this.ood(values);
    out.decision = this.decision(out);
    return out;
  };

  Engine.prototype.decision = function (out) {
    const c = out.targets.CAD;
    if (out.ood.isOOD)
      return { level: "check", title: "Check the inputs", text: "Some values are outside what the model was trained on, so the estimate is less reliable." };
    if (c.setKind === "positive")
      return { level: "high", title: "CAD likely", text: "At 90% confidence the model's prediction set contains only CAD. Consider cardiology review / angiography as per clinical judgement." };
    if (c.setKind === "negative")
      return { level: "low", title: "CAD unlikely", text: "At 90% confidence the prediction set contains only 'No CAD'. This does not rule out disease; use clinical judgement." };
    if (c.setKind === "both")
      return { level: "unsure", title: "Uncertain: refer", text: "The model cannot separate CAD from no CAD with 90% confidence for this patient. Refer to a cardiologist." };
    return { level: "check", title: "Atypical pattern", text: "No label reaches 90% confidence (empty set). Re-check the inputs." };
  };

  // What-if: model sensitivity to modifiable inputs moved to a healthy reference value.
  Engine.prototype.whatIf = function (values, available) {
    const base = this.predict(values, available);
    const rows = [];
    const tg = this.b.whatif_targets;
    const h = Number(values.Length) / 100;
    for (const [name, target] of Object.entries(tg)) {
      if (target === null || target === undefined) continue;
      const cur = Number(values[name]);
      const healthy = name === "HDL" ? cur < target : cur > target;
      if (!healthy) continue;
      const v2 = Object.assign({}, values, { [name]: target });
      if (name === "BMI" && h > 0) v2.Weight = Math.round(target * h * h * 10) / 10;
      const r = this.predict(v2, available);
      const delta = {};
      for (const t of this.b.targets) delta[t] = r.targets[t].p - base.targets[t].p;
      rows.push({ name, label: this.meta[name].label, from: cur, to: target, delta });
    }
    rows.sort((u, v) => u.delta.CAD - v.delta.CAD);
    return { base, rows };
  };

  return { Engine, LABELS, sigmoid };
});
