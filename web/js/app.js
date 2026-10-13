/* HeartRisk 3D - application controller: form, prediction service, 3D + dashboard rendering. */
(function () {
  "use strict";
  const B = window.MODEL_BUNDLE;
  const A = window.ANATOMY;
  const eng = new HeartEngine.Engine(B);
  const META = {};
  B.features.forEach((f) => (META[f.name] = f));
  const VESSELS = ["LAD", "LCX", "RCA"];
  const TNAME = { CAD: "Coronary artery disease", LAD: "LAD stenosis", LCX: "LCX stenosis", RCA: "RCA stenosis" };
  const MODCOL = { history: "#7aa2f7", symptoms: "#c792ea", ecg: "#f78c6c", labs: "#c3e88d", echo: "#4fd1c5" };
  const QUICK = ["Age", "Sex", "Weight", "Length", "BMI", "DM", "HTN", "Current Smoker", "FH", "Typical Chest Pain", "Atypical",
    "Nonanginal", "Dyspnea", "BP", "Q Wave", "St Elevation", "St Depression", "Tinversion", "FBS", "LDL", "HDL", "TG",
    "EF-TTE", "Region RWMA", "VHD"];
  const ORDINAL_LABELS = {
    "Function Class": ["0", "I", "II", "III"],
    "Region RWMA": ["0 (none)", "1", "2", "3", "4"],
    VHD: ["None", "Mild", "Moderate", "Severe"],
  };
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const pct = (p) => (p > 0.99 ? ">99%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`);
  const pts = (d) => (Math.abs(d) < 0.0005 ? "0.0 pts" : `${d > 0 ? "+" : ""}${(d * 100).toFixed(1)} pts`);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (v) => (Math.abs(v) >= 100 ? Math.round(v) : Math.round(v * 10) / 10);

  const state = {
    values: {},
    available: { ecg: true, labs: true, echo: true },
    fields: "quick",
    whyTarget: "CAD",
    selected: null,
    scenario: null,
    last: null,
    mode: "browser",
  };

  // ------------------------------------------------------------------ prediction service
  const Service = {
    async init() {
      if (location.protocol.startsWith("http")) {
        try {
          const r = await fetch("api/health", { cache: "no-store" });
          if (r.ok) state.mode = "api";
        } catch (e) { /* static hosting: browser engine */ }
      }
      $("#modeBadge").textContent = state.mode === "api" ? "engine: Python API (scikit-learn + SHAP)" : "engine: in-browser (same model bundle)";
      $("#privacyBadge").textContent = state.mode === "api" ? "Data stays on this device (localhost API)" : "Data stays in this browser";
    },
    async predict(values, available) {
      if (state.mode === "api") {
        try {
          const r = await fetch("api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values, available }) });
          if (r.ok) return r.json();
        } catch (e) { /* fall back */ }
      }
      return eng.predict(values, available);
    },
    async whatIf(values, available) {
      if (state.mode === "api") {
        try {
          const r = await fetch("api/whatif", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values, available }) });
          if (r.ok) return r.json();
        } catch (e) { /* fall back */ }
      }
      return eng.whatIf(values, available);
    },
  };

  // ------------------------------------------------------------------ form
  function buildForm() {
    const form = $("#form");
    form.innerHTML = "";
    for (const mod of B.modalities) {
      const fs = document.createElement("fieldset");
      fs.dataset.mod = mod;
      fs.innerHTML = `<legend>${esc(B.modality_label[mod])}</legend>`;
      for (const f of B.features.filter((x) => x.modality === mod)) {
        if (f.name === "BBB_RBBB") continue;
        const row = document.createElement("div");
        row.className = "row";
        row.dataset.name = f.group === "BBB" ? "BBB" : f.name;
        if (f.group === "BBB") {
          row.innerHTML = `<label for="f_BBB">Bundle branch block</label>
            <select id="f_BBB"><option value="N">None</option><option value="LBBB">LBBB</option><option value="RBBB">RBBB</option></select>`;
        } else if (f.kind === "binary") {
          row.innerHTML = `<label for="f_${f.name}">${esc(f.label)}</label>
            <label class="switch"><input type="checkbox" id="f_${f.name}" data-f="${esc(f.name)}"><span></span></label>`;
        } else if (f.kind === "ordinal") {
          const opts = (ORDINAL_LABELS[f.name] || []).map((l, i) => `<option value="${i}">${esc(l)}</option>`).join("");
          row.innerHTML = `<label for="f_${f.name}">${esc(f.label)}</label><select id="f_${f.name}" data-f="${esc(f.name)}">${opts}</select>`;
        } else {
          const nr = f.normal ? `normal ${f.normal[0]}-${f.normal[1]}` : `data ${fmt(f.min)}-${fmt(f.max)}`;
          row.innerHTML = `<label for="f_${f.name}">${esc(f.label)}<small>${esc(f.unit)} ${f.unit ? "· " : ""}${nr}</small></label>
            <input type="number" id="f_${f.name}" data-f="${esc(f.name)}" step="${f.step}" ${f.name === "BMI" ? "readonly title='Calculated from weight and height'" : ""}>`;
        }
        fs.appendChild(row);
      }
      form.appendChild(fs);
    }
    form.addEventListener("input", onFormChange);
    form.addEventListener("change", onFormChange);
    applyFieldView();
    applyAvailability();
  }

  function writeForm() {
    for (const f of B.features) {
      if (f.group === "BBB") continue;
      const el = $(`#f_${CSS.escape(f.name)}`);
      if (!el) continue;
      const v = state.values[f.name];
      if (el.type === "checkbox") el.checked = Number(v) === 1;
      else el.value = f.kind === "numeric" ? fmt(v) : String(Math.round(v));
    }
    $("#f_BBB").value = state.values.BBB_LBBB ? "LBBB" : state.values.BBB_RBBB ? "RBBB" : "N";
    markRanges();
  }

  function readForm() {
    for (const el of $$("[data-f]", $("#form"))) {
      const n = el.dataset.f;
      state.values[n] = el.type === "checkbox" ? (el.checked ? 1 : 0) : el.value === "" ? null : Number(el.value);
    }
    const bbb = $("#f_BBB").value;
    state.values.BBB_LBBB = bbb === "LBBB" ? 1 : 0;
    state.values.BBB_RBBB = bbb === "RBBB" ? 1 : 0;
    const h = Number(state.values.Length) / 100, w = Number(state.values.Weight);
    if (h > 0 && w > 0) {
      state.values.BMI = w / (h * h);
      $("#f_BMI").value = fmt(state.values.BMI);
    }
  }

  function markRanges() {
    for (const f of B.features.filter((x) => x.kind === "numeric")) {
      const el = $(`#f_${CSS.escape(f.name)}`);
      if (!el) continue;
      const v = Number(state.values[f.name]);
      el.classList.toggle("abn", !!f.normal && (v < f.normal[0] || v > f.normal[1]));
      const r = B.ood.ranges[f.name];
      el.classList.toggle("oor", !!r && (v < r[0] || v > r[1]));
    }
  }

  function applyFieldView() {
    $$(".row", $("#form")).forEach((row) => {
      const n = row.dataset.name;
      row.classList.toggle("hidden", state.fields === "quick" && !QUICK.includes(n));
    });
    $$("fieldset", $("#form")).forEach((fs) => {
      fs.style.display = $$(".row:not(.hidden)", fs).length ? "" : "none";
    });
  }

  function applyAvailability() {
    $$("fieldset", $("#form")).forEach((fs) => {
      const m = fs.dataset.mod;
      fs.classList.toggle("off", B.optional_modalities.includes(m) && !state.available[m]);
    });
    const key = eng.setKey(state.available);
    $("#availHint").textContent = key === "ecg+labs+echo"
      ? "History and symptoms are always used. Turn off tests that were not done; a model trained without them is used."
      : `Using the model trained on: ${B.configs[key].modalities.map((m) => B.modality_label[m]).join(", ")}.`;
  }

  let timer = null;
  function onFormChange() {
    readForm();
    markRanges();
    clearTimeout(timer);
    timer = setTimeout(run, 120);
  }

  // ------------------------------------------------------------------ main update
  async function run() {
    const out = await Service.predict(state.values, state.available);
    state.last = out;
    if (!state.scenario) paint3D(out);
    renderSummary(out);
    renderVessels(out);
    renderWhy(out);
    if ($("#tab-whatif").classList.contains("on")) renderWhatIf();
  }

  function paint3D(out) {
    const risk = {};
    VESSELS.forEach((v) => (risk[v] = out.targets[v].p));
    heart.setRisk(risk);
    Bullseye.render($("#bullseye"), risk, A.bullseye, riskColorHex);
  }

  // ------------------------------------------------------------------ summary
  function bandBar(t) {
    const c = riskColorHex(t.p);
    return `<div class="bar" title="probability ${pct(t.p)}, bootstrap band ${pct(t.band[0])}-${pct(t.band[1])}">
      <div class="fill" style="width:${t.p * 100}%;background:${c}"></div>
      <div class="bandmark" style="left:${t.band[0] * 100}%;width:${Math.max(0.8, (t.band[1] - t.band[0]) * 100)}%"></div></div>`;
  }
  function chips(t) {
    return t.set.length
      ? t.set.map((s) => `<span class="chip ${/Normal|No CAD/.test(s) ? "neg" : "pos"}">${esc(s)}</span>`).join("")
      : `<span class="chip">empty set</span>`;
  }

  function renderSummary(out) {
    const c = out.targets.CAD, d = out.decision;
    const used = out.modalities.map((m) => B.modality_label[m]).join(" · ");
    let warn = "";
    if (out.ood.isOOD) {
      const list = out.ood.outOfRange.map((o) => `${esc(META[o.name].label)} = ${fmt(o.value)} (training range ${fmt(o.range[0])}-${fmt(o.range[1])})`).join("; ");
      warn += `<div class="warnbox"><b>Unusual input.</b> ${list || "The combination of values is far from the training patients"} (Mahalanobis ${out.ood.distance.toFixed(1)} vs ${out.ood.threshold.toFixed(1)}). Treat the estimate with caution.</div>`;
    }
    if (out.imputed && out.imputed.length) warn += `<div class="warnbox">Missing values filled with the training median: ${out.imputed.map((n) => esc(META[n].label)).join(", ")}.</div>`;
    if (!out.coherence.ok) warn += `<div class="warnbox">Vessel and overall CAD estimates disagree for this patient. ${esc(out.coherence.note)}</div>`;
    $("#tab-summary").innerHTML = `
      <div class="card decision ${d.level}"><b>${esc(d.title)}</b><div class="small" style="margin-top:4px">${esc(d.text)}</div></div>
      ${warn}
      <div class="card">
        <div class="muted small">Probability of coronary artery disease (>=50% stenosis in any major vessel)</div>
        <div class="big"><span class="num" style="color:${riskColorHex(c.p)}">${pct(c.p)}</span>
          <span class="band">range ${pct(c.band[0])} - ${pct(c.band[1])}</span></div>
        ${bandBar(c)}
        <div class="small muted" style="margin-top:6px">90% conformal prediction set:</div><div class="chips">${chips(c)}</div>
      </div>
      <div class="card">
        <div class="kv">
          <span>Expected diseased vessels</span><span><b>${out.expectedVessels.toFixed(1)}</b> of 3 (sum of vessel probabilities)</span>
          ${VESSELS.map((v) => `<span>${v}</span><span><b style="color:${riskColorHex(out.targets[v].p)}">${pct(out.targets[v].p)}</b> · set: ${out.targets[v].set.join(" / ") || "empty"}</span>`).join("")}
          <span>Inputs used</span><span>${esc(used)}</span>
        </div>
      </div>
      <div class="card">
        <div class="small muted" style="margin-bottom:6px">Patient-friendly audio summary and printable report</div>
        <button class="btn" type="button" id="btnReport">Download report (PDF)</button>
        <button class="btn" type="button" data-say="en">Play in English</button>
        <button class="btn" type="button" data-say="hi">हिंदी में सुनें</button>
        <span class="small muted" id="sayNote"></span>
      </div>
      <p class="small muted">Band = 10th-90th percentile across 10 bootstrap refits. Conformal sets are calibrated on out-of-fold predictions (class-conditional, alpha = 0.10). This is decision support only and does not replace angiography or a clinician.</p>`;
  }

  // ------------------------------------------------------------------ vessels
  function renderVessels(out) {
    $("#tab-vessels").innerHTML = VESSELS.map((v) => {
      const t = out.targets[v], a = A.vessels[v];
      const top = t.drivers.slice(0, 4).map((d) => `<span class="${d.contrib > 0 ? "pos" : "neg"}">${d.contrib > 0 ? "▲" : "▼"} ${esc(d.label)}</span>`).join(" · ");
      return `<div class="card vrow ${state.selected === v ? "on" : ""}" data-v="${v}">
        <div style="display:flex;justify-content:space-between;gap:8px"><b>${esc(a.name)}</b><b style="color:${riskColorHex(t.p)}">${pct(t.p)}</b></div>
        ${bandBar(t)}
        <div class="chips">${chips(t)}<span class="chip">model: ${t.family === "linear" || t.family === "logreg" ? "logistic regression" : "gradient boosting"}</span></div>
        <div class="small" style="margin-top:6px">${esc(a.course)}</div>
        <div class="small muted">Typical territory: ${esc(a.territory)}</div>
        <div class="small" style="margin-top:6px">Main drivers: ${top}</div>
      </div>`;
    }).join("") + `<p class="small muted">Predictions are per vessel (stenosis >=50%). The dataset does not record where along the vessel a lesion is, so the 3D view colours the whole vessel.</p>`;
    $$(".vrow").forEach((el) => el.addEventListener("click", () => heart.select(el.dataset.v)));
  }

  // ------------------------------------------------------------------ why
  function renderWhy(out) {
    const tgt = state.whyTarget, t = out.targets[tgt];
    const mc = t.modalityContribution;
    const mods = Object.keys(mc).sort((a, b) => mc[b].share - mc[a].share);
    const stack = mods.map((m) => `<span style="width:${mc[m].share * 100}%;background:${MODCOL[m]}" title="${esc(B.modality_label[m])} ${pct(mc[m].share)}"></span>`).join("");
    const leg = mods.map((m) => `<span><i style="background:${MODCOL[m]}"></i>${esc(B.modality_label[m])} ${pct(mc[m].share)} <span class="${mc[m].signed > 0 ? "pos" : "neg"}">${mc[m].signed > 0 ? "↑" : "↓"}</span></span>`).join("");
    const drv = t.drivers.slice(0, 12);
    const maxAbs = Math.max(0.05, ...drv.map((d) => Math.abs(d.contrib)));
    const rows = drv.map((d) => {
      const f = META[d.name];
      const w = (Math.abs(d.contrib) / maxAbs) * 50;
      const val = f.kind === "binary" ? (d.value ? "yes" : "no") : f.kind === "ordinal" ? (ORDINAL_LABELS[f.name] || [])[Math.round(d.value)] || fmt(d.value) : `${fmt(d.value)} ${f.unit}`;
      const abn = f.normal && (d.value < f.normal[0] || d.value > f.normal[1]) ? ` <span class="pos" title="outside normal ${f.normal[0]}-${f.normal[1]}">(abnormal)</span>` : "";
      return `<div class="drv"><span>${esc(f.label)}: <b>${esc(val)}</b>${abn}</span>
        <div class="dbar"><i style="${d.contrib > 0 ? `left:50%;width:${w}%;background:var(--crimson)` : `right:50%;width:${w}%;background:var(--teal)`}"></i></div></div>`;
    }).join("");
    const method = t.attribution_method || (t.family === "logreg" ? "exact linear SHAP (log-odds vs the average training patient)" : "occlusion to the training mean (log-odds)");
    $("#tab-why").innerHTML = `
      <div class="seltarget">${["CAD", ...VESSELS].map((k) => `<button type="button" data-why="${k}" class="${k === tgt ? "on" : ""}">${k}</button>`).join("")}</div>
      <div class="card"><b>${esc(TNAME[tgt])}: ${pct(t.p)}</b>
        <h3>Which test type drove this prediction</h3>
        <div class="stack">${stack}</div><div class="mlegend">${leg}</div>
        <p class="small muted">Share of total attribution by modality. Arrow = net direction (↑ raises risk, ↓ lowers it).</p>
      </div>
      <div class="card"><h3 style="margin-top:0">Top factors for this patient</h3>
        <div class="small muted" style="display:flex;justify-content:space-between;margin-bottom:4px"><span></span><span style="width:140px;display:flex;justify-content:space-between"><span class="neg">lowers</span><span class="pos">raises</span></span></div>
        ${rows}
        <p class="small muted">Method: ${esc(method)}. Contributions are associations learned from 242 training patients, not causes.</p>
      </div>`;
    $$("[data-why]").forEach((b) => b.addEventListener("click", () => { state.whyTarget = b.dataset.why; renderWhy(state.last); }));
  }

  // ------------------------------------------------------------------ what-if
  const SLIDERS = [
    { name: "BP", min: 90, max: 200, step: 1 },
    { name: "BMI", min: 17, max: 40, step: 0.1 },
    { name: "FBS", min: 60, max: 300, step: 1 },
    { name: "LDL", min: 40, max: 230, step: 1 },
    { name: "HDL", min: 20, max: 90, step: 1 },
    { name: "TG", min: 40, max: 600, step: 1 },
  ];
  async function renderWhatIf() {
    const res = await Service.whatIf(state.values, state.available);
    const sc = state.scenario || {};
    const sl = SLIDERS.filter((s) => s.name === "BMI" || B.configs[eng.setKey(state.available)].features.includes(s.name)).map((s) => {
      const v = sc[s.name] !== undefined ? sc[s.name] : state.values[s.name];
      return `<div class="slider"><label for="w_${s.name}">${esc(META[s.name].label)}</label>
        <input type="range" id="w_${s.name}" data-w="${s.name}" min="${s.min}" max="${s.max}" step="${s.step}" value="${v}">
        <span id="wv_${s.name}">${fmt(v)}</span></div>`;
    }).join("");
    const smoke = sc["Current Smoker"] !== undefined ? sc["Current Smoker"] : state.values["Current Smoker"];
    const table = res.rows.length
      ? `<table><tr><th>Change (to reference)</th>${["CAD", ...VESSELS].map((t) => `<th class="num">${t}</th>`).join("")}</tr>${res.rows.map((r) => `<tr><td>${esc(r.label)} ${fmt(r.from)} → ${fmt(r.to)}</td>${["CAD", ...VESSELS].map((t) => `<td class="num ${r.delta[t] > 0.005 ? "pos" : r.delta[t] < -0.005 ? "neg" : ""}">${pts(r.delta[t])}</td>`).join("")}</tr>`).join("")}</table>`
      : `<p class="small muted">All modifiable inputs are already at or better than the reference values.</p>`;
    const paradox = res.rows.filter((r) => r.delta.CAD > 0.005).map((r) => esc(r.label));
    const paradoxNote = paradox.length
      ? `<p class="small" style="color:#ffd99a;margin:8px 0 0">Counter-intuitive: moving <b>${paradox.join(", ")}</b> to the healthy value raises the model's CAD estimate. This reflects confounding in observational data (for example the "obesity paradox", or treated patients having better lab values), and is exactly why this view must not be read as treatment advice.</p>`
      : "";
    $("#tab-whatif").innerHTML = `
      <div class="warnbox">What-if shows how the <b>model's output</b> responds when an input changes. It is model sensitivity learned from observational data, <b>not</b> the effect of treating the patient. Age, sex, history and test findings are locked.</div>
      <div class="card"><h3 style="margin-top:0">Move modifiable inputs to a healthy reference</h3>${table}${paradoxNote}</div>
      <div class="card"><h3 style="margin-top:0">Try a scenario</h3>${sl}
        <div class="slider"><label for="w_smoke">Current smoker</label><label class="switch"><input type="checkbox" id="w_smoke" ${Number(smoke) === 1 ? "checked" : ""}><span></span></label><span></span></div>
        <div id="scenarioOut" class="small" style="margin-top:8px"></div>
        <button class="btn" type="button" id="wReset" style="margin-top:8px">Reset scenario</button>
      </div>`;
    $$("[data-w]").forEach((el) => el.addEventListener("input", onScenario));
    $("#w_smoke").addEventListener("change", onScenario);
    $("#wReset").addEventListener("click", () => { state.scenario = null; paint3D(state.last); renderWhatIf(); });
    if (state.scenario) onScenario();
  }

  let stimer = null;
  function onScenario() {
    const sc = {};
    $$("[data-w]").forEach((el) => { sc[el.dataset.w] = Number(el.value); $(`#wv_${el.dataset.w}`).textContent = fmt(Number(el.value)); });
    sc["Current Smoker"] = $("#w_smoke").checked ? 1 : 0;
    const h = Number(state.values.Length) / 100;
    if (h > 0) sc.Weight = Math.round(sc.BMI * h * h * 10) / 10;
    state.scenario = sc;
    clearTimeout(stimer);
    stimer = setTimeout(async () => {
      const v2 = Object.assign({}, state.values, sc);
      const r = await Service.predict(v2, state.available);
      paint3D(r);
      const base = state.last;
      $("#scenarioOut").innerHTML = `<table><tr><th></th><th class="num">Baseline</th><th class="num">Scenario</th><th class="num">Change</th></tr>${["CAD", ...VESSELS].map((t) => {
        const d = r.targets[t].p - base.targets[t].p;
        return `<tr><td>${t}</td><td class="num">${pct(base.targets[t].p)}</td><td class="num">${pct(r.targets[t].p)}</td><td class="num ${d > 0.005 ? "pos" : d < -0.005 ? "neg" : ""}">${pts(d)}</td></tr>`;
      }).join("")}</table><p class="muted">The 3D heart now shows the scenario. Reset to return to the patient.</p>`;
    }, 120);
  }

  // ------------------------------------------------------------------ model tab (static)
  function ci(m) { return `${m.value.toFixed(3)} <span class="muted">[${m.ci95[0].toFixed(2)}-${m.ci95[1].toFixed(2)}]</span>`; }
  function renderModel() {
    const M = B.metrics, R = B.reports;
    const FULL = "ecg+labs+echo";
    const tgts = ["CAD", ...VESSELS];
    const test = `<table><tr><th>Target</th><th class="num">ROC-AUC</th><th class="num">Accuracy</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th></tr>
      ${tgts.map((t) => { const m = M.test_full[t]; return `<tr><td>${t}</td><td class="num">${ci(m.roc_auc)}</td><td class="num">${ci(m.accuracy)}</td><td class="num">${ci(m.precision)}</td><td class="num">${ci(m.recall)}</td><td class="num">${ci(m.f1)}</td></tr>`; }).join("")}</table>`;
    const cv = `<table><tr><th>Target</th><th class="num">CV ROC-AUC (mean ± sd)</th><th class="num">Brier (OOF)</th></tr>${tgts.map((t) => `<tr><td>${t}</td><td class="num">${M.ablation[FULL][t].cv_auc_mean.toFixed(3)} ± ${M.ablation[FULL][t].cv_auc_std.toFixed(3)}</td><td class="num">${R.calibration[t].brier_oof.toFixed(3)}</td></tr>`).join("")}</table>`;
    const fams = { logreg: "Elastic-net LR", gbm: "Gradient boosting", rf: "Random forest", hgb_monotone: "HistGB (monotone)" };
    const bench = `<table><tr><th>Family</th>${tgts.map((t) => `<th class="num">${t}</th>`).join("")}</tr>${Object.keys(fams).map((f) => `<tr><td>${fams[f]}</td>${tgts.map((t) => { const r = M.benchmark[t].families[f]; const ch = M.benchmark[t].chosen === f; return `<td class="num" style="${ch ? "color:#b8f3ec;font-weight:700" : ""}">${r.auc_mean.toFixed(3)}${ch ? " ✓" : ""}</td>`; }).join("")}</tr>`).join("")}</table>
      <p class="small muted">Repeated 5×5 CV ROC-AUC on the training split. ✓ = deployed: the simplest model within one standard error of the best (1-SE rule).</p>`;
    const order = ["bedside", "ecg", "labs", "echo", "ecg+labs", "ecg+echo", "labs+echo", FULL];
    const setName = (k) => (k === "bedside" ? "History + symptoms" : "+ " + k.split("+").map((m) => B.modality_label[m]).join(" + "));
    const abl = `<table><tr><th>Inputs</th>${tgts.map((t) => `<th class="num">${t}</th>`).join("")}</tr>${order.map((k) => `<tr ${k === FULL ? 'class="chosen"' : ""}><td>${esc(setName(k))}</td>${tgts.map((t) => `<td class="num">${M.ablation[k][t].cv_auc_mean.toFixed(3)}</td>`).join("")}</tr>`).join("")}</table>
      <p class="small muted">CV ROC-AUC when only some test types are available. This is what the "Tests available" switch uses.</p>`;
    const conf = `<table><tr><th>Target</th><th class="num">Coverage (target 90%)</th><th class="num">Single label</th><th class="num">Uncertain (both)</th></tr>${tgts.map((t) => { const c = M.conformal[FULL][t]; return `<tr><td>${t}</td><td class="num">${pct(c.test_coverage)}</td><td class="num">${pct(c.share_single)}</td><td class="num">${pct(c.share_both)}</td></tr>`; }).join("")}</table>
      <p class="small muted">Measured on the 61 held-out patients; with n = 61 the sampling error of a coverage estimate is about ±7 points.</p>`;
    const cols = { CAD: "#e6eff0", LAD: "#f78c6c", LCX: "#c792ea", RCA: "#4fd1c5" };
    const roc = Charts.line(tgts.map((t) => ({ x: R.curves[t].test.roc.fpr, y: R.curves[t].test.roc.tpr, color: cols[t], label: t })), { xlab: "False positive rate", ylab: "True positive rate", diag: true, title: "ROC on held-out test" });
    const cal = Charts.line(tgts.map((t) => ({ x: R.calibration[t].oof.map((b) => b.pred), y: R.calibration[t].oof.map((b) => b.obs), color: cols[t], label: t, dots: true })), { xlab: "Predicted probability", ylab: "Observed frequency", diag: true, title: "Calibration" });
    const dc = R.decision_curve.CAD;
    const dca = Charts.line([
      { x: dc.map((r) => r.threshold), y: dc.map((r) => r.model), color: "#4fd1c5", label: "Model" },
      { x: dc.map((r) => r.threshold), y: dc.map((r) => r.treat_all), color: "#f0a202", label: "Refer all", dash: "4 3" },
      { x: dc.map((r) => r.threshold), y: dc.map((r) => r.treat_none), color: "#8aa3a8", label: "Refer none", dash: "2 3" },
    ], { xlab: "Threshold probability", ylab: "Net benefit", xmin: 0.05, xmax: 0.8, ymin: -0.1, ymax: 0.8, title: "Decision curve" });
    const sub = `<table><tr><th>Subgroup (OOF)</th>${tgts.map((t) => `<th class="num">${t}</th>`).join("")}</tr>${["male", "female", "age<60", "age>=60"].map((g) => `<tr><td>${g} (n=${(R.subgroups.CAD[g] || {}).n || "-"})</td>${tgts.map((t) => `<td class="num">${R.subgroups[t][g] ? R.subgroups[t][g].auc.toFixed(3) : "-"}</td>`).join("")}</tr>`).join("")}</table>`;
    $("#tab-model").innerHTML = `
      <div class="card"><h3 style="margin-top:0">Held-out test (n = ${B.n_test}), 95% bootstrap CI</h3>${test}<p class="small muted">Threshold 0.5 on calibrated probabilities.</p></div>
      <div class="card"><h3 style="margin-top:0">Cross-validation (n = ${B.n_train})</h3>${cv}</div>
      <div class="card"><h3 style="margin-top:0">Model comparison</h3>${bench}</div>
      <div class="card"><h3 style="margin-top:0">Multimodal ablation</h3>${abl}</div>
      <div class="card"><h3 style="margin-top:0">ROC (test)</h3>${roc}</div>
      <div class="card"><h3 style="margin-top:0">Calibration (out-of-fold)</h3>${cal}</div>
      <div class="card"><h3 style="margin-top:0">Decision curve, CAD (out-of-fold)</h3>${dca}<p class="small muted">Net benefit above "refer all" means the model adds value at that referral threshold.</p></div>
      <div class="card"><h3 style="margin-top:0">Conformal coverage</h3>${conf}</div>
      <div class="card"><h3 style="margin-top:0">Subgroups (CV ROC-AUC)</h3>${sub}<p class="small muted">Small groups; shown for transparency, not as proof of fairness.</p></div>
      <div class="card small"><b>Leakage control.</b> LAD, LCX, RCA and Cath are never model inputs (checked by an automated test). Scaling is fitted inside each CV fold. <b>Region RWMA</b> is a 0-4 score with no location, so it is not mapped onto the heart. Data: ${B.n_train + B.n_test} patients, single centre; external validation is still needed.</div>`;
  }

  // ------------------------------------------------------------------ audio summary
  function speak(lang) {
    const out = state.last;
    if (!out || !("speechSynthesis" in window)) { $("#sayNote").textContent = "Speech is not supported in this browser."; return; }
    const p = (t) => Math.round(out.targets[t].p * 100);
    const unsure = out.decision.level === "unsure";
    const voices = speechSynthesis.getVoices();
    const low = (x) => (x.lang || "").toLowerCase().replace("_", "-");
    const hiVoice = voices.find((x) => low(x).startsWith("hi"));
    const enVoice = voices.find((x) => low(x) === "en-in") || voices.find((x) => low(x).startsWith("en"));
    let text, voice, note;
    if (lang === "hi" && hiVoice) {
      text = `कोरोनरी आर्टरी डिज़ीज़ की अनुमानित संभावना ${p("CAD")} प्रतिशत है। एल ए डी नस में ${p("LAD")} प्रतिशत, एल सी एक्स में ${p("LCX")} प्रतिशत, और आर सी ए में ${p("RCA")} प्रतिशत। ${unsure ? "मॉडल पक्का नहीं है, कृपया हृदय रोग विशेषज्ञ से मिलें।" : ""} यह केवल निर्णय में सहायता के लिए है, यह निदान नहीं है।`;
      voice = hiVoice;
      note = `voice: ${hiVoice.name}`;
    } else if (lang === "hi") {
      // no Hindi voice on this device: romanised Hindi read by an English (India) voice
      text = `Coronary artery disease ki anumaanit sambhaavna ${p("CAD")} pratishat hai. L A D mein ${p("LAD")} pratishat, L C X mein ${p("LCX")} pratishat, aur R C A mein ${p("RCA")} pratishat. ${unsure ? "Model pakka nahin hai, kripya cardiologist se milen." : ""} Yeh keval nirnay mein sahaayata ke liye hai, nidaan nahin.`;
      voice = enVoice;
      note = "No Hindi voice on this device: romanised Hindi used. (Install Hindi speech in OS settings for native audio.)";
    } else {
      text = `The estimated chance of coronary artery disease is ${p("CAD")} percent. For the three main arteries: L A D ${p("LAD")} percent, L C X ${p("LCX")} percent, and R C A ${p("RCA")} percent. ${out.decision.title}. This is decision support only, not a diagnosis.`;
      voice = enVoice;
      note = enVoice ? `voice: ${enVoice.name}` : "";
    }
    const u = new SpeechSynthesisUtterance(text);
    if (voice) { u.voice = voice; u.lang = voice.lang; } else u.lang = lang === "hi" ? "hi-IN" : "en-IN";
    u.rate = 0.95;
    $("#sayNote").textContent = note;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  }
  if ("speechSynthesis" in window) speechSynthesis.getVoices(); // warm the voice list (loads async)

  // ------------------------------------------------------------------ printable report
  function report() {
    const out = state.last;
    if (!out) return;
    const img = heart.snapshot();
    const tg = ["CAD", ...VESSELS];
    const rows = tg.map((t) => {
      const r = out.targets[t];
      const top = r.drivers.slice(0, 4).map((d) => `${esc(d.label)} (${d.contrib > 0 ? "raises" : "lowers"})`).join(", ");
      return `<tr><td>${esc(TNAME[t])}</td><td>${pct(r.p)}</td><td>${pct(r.band[0])} - ${pct(r.band[1])}</td><td>${esc(r.set.join(" / ") || "empty")}</td><td>${top}</td></tr>`;
    }).join("");
    const keyVals = QUICK.filter((n) => out.modalities.includes(META[n].modality)).map((n) => {
      const f = META[n], v = state.values[n];
      const val = f.kind === "binary" ? (Number(v) ? "yes" : "no") : f.kind === "ordinal" ? (ORDINAL_LABELS[n] || [])[Math.round(v)] || v : `${fmt(Number(v))} ${f.unit}`;
      return `<tr><td>${esc(f.label)}</td><td>${esc(val)}</td></tr>`;
    }).join("");
    const w = window.open("", "_blank");
    if (!w) { alert("Please allow pop-ups to open the report."); return; }
    w.document.write(`<!doctype html><html><head><meta charset="utf-8"><title>HeartRisk 3D report</title>
      <meta name="color-scheme" content="light"><style>html,body{background:#fff}body{font:12px/1.4 Segoe UI,Arial,sans-serif;color:#1d2a2c;margin:24px}h1{font-size:20px;margin:0;color:#0e4d52}
      h2{font-size:14px;color:#0e4d52;border-bottom:2px solid #d9821e;margin:14px 0 6px}table{border-collapse:collapse;width:100%}
      td,th{border:1px solid #c9dcda;padding:3px 6px;text-align:left;vertical-align:top}th{background:#eaf4f3}
      .warn{background:#fcefdc;border-left:4px solid #d9821e;padding:6px 10px;margin:8px 0}.flex{display:flex;gap:14px}
      .flex img{width:42%;max-height:360px;object-fit:contain;border-radius:8px;background:#0c171b}.dec{font-size:16px;font-weight:700}</style></head><body>
      <h1>HeartRisk 3D - decision-support report</h1>
      <div>${new Date().toLocaleString()} · model ${esc(B.version)} · inputs: ${esc(out.modalities.map((m) => B.modality_label[m]).join(", "))}</div>
      <div class="warn"><b>Decision support / education only.</b> Not a diagnosis and not a substitute for angiography or clinical assessment.</div>
      <div class="flex"><img src="${img}" alt="3D heart"><div><div class="dec">${esc(out.decision.title)}</div><p>${esc(out.decision.text)}</p>
      <p>Expected diseased vessels: <b>${out.expectedVessels.toFixed(1)}</b> of 3</p>${out.ood.isOOD ? '<p class="warn">Unusual input values: interpret with caution.</p>' : ""}</div></div>
      <h2>Predictions</h2><table><tr><th>Target</th><th>Probability</th><th>Band (10-90%)</th><th>90% conformal set</th><th>Main factors</th></tr>${rows}</table>
      <h2>Key inputs</h2><table>${keyVals}</table>
      <p style="color:#5b6b6e;margin-top:12px">Model: trained on the Z-Alizadeh Sani extension dataset (303 patients, single centre, CC BY 4.0). Factors are statistical associations, not causes.</p>
      <script>setTimeout(()=>print(),400)<\/script></body></html>`);
    w.document.close();
  }

  // ------------------------------------------------------------------ guided tour
  const TOUR = [
    { sel: ".input", text: "1/4 · A real held-out patient is loaded. Every input on the left updates the prediction instantly. Try another patient from the dropdown." },
    { sel: ".viewer", text: "2/4 · Each coronary artery is coloured by its predicted stenosis probability. Drag to rotate, scroll to zoom, click a vessel for details." },
    { sel: ".avail", text: "3/4 · No echo or lab results? Untick them: a model trained without those tests is used and the uncertainty is shown honestly." },
    { sel: ".tabs", text: "4/4 · Why explains the prediction, What-if shows model sensitivity, Model shows the validation. Decision support only." },
  ];
  function tour(i) {
    $$(".tour-on").forEach((e) => e.classList.remove("tour-on"));
    $("#tourCard")?.remove();
    if (i >= TOUR.length) { try { localStorage.setItem("hr3d_tour", "1"); } catch (e) { /* ignore */ } return; }
    const el = $(TOUR[i].sel);
    el.classList.add("tour-on");
    const r = el.getBoundingClientRect();
    const card = document.createElement("div");
    card.id = "tourCard";
    card.className = "tourcard";
    card.innerHTML = `<p>${esc(TOUR[i].text)}</p><div><button class="btn" type="button" data-t="skip">Skip</button> <button class="btn primary" type="button" data-t="next">${i === TOUR.length - 1 ? "Done" : "Next"}</button></div>`;
    document.body.appendChild(card);
    const W = window.innerWidth || document.documentElement.clientWidth || 1280;
    const H = window.innerHeight || document.documentElement.clientHeight || 720;
    const top = Math.max(12, Math.min(H - card.offsetHeight - 12, r.top + 12));
    const left = r.right + 12 + 300 < W ? r.right + 12 : r.left + 12;
    card.style.top = `${top}px`;
    card.style.left = `${Math.max(12, Math.min(left, W - 320))}px`;
    card.querySelector('[data-t="next"]').addEventListener("click", () => tour(i + 1));
    card.querySelector('[data-t="skip"]').addEventListener("click", () => tour(TOUR.length));
  }

  // ------------------------------------------------------------------ wiring
  let heart;
  async function start() {
    heart = new HeartView($("#heart3d"), A, {
      onSelect: (k) => {
        state.selected = k;
        if (k) { $$(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === "vessels")); $$(".tab").forEach((t) => t.classList.toggle("on", t.id === "tab-vessels")); }
        if (state.last) renderVessels(state.last);
      },
    });
    $("#ver").textContent = `model ${B.version} · data ${B.data_sha256_12}`;
    const sel = $("#preset");
    sel.innerHTML = B.presets.map((p, i) => `<option value="${i}">${esc(p.label)}</option>`).join("");
    sel.addEventListener("change", () => loadPreset(Number(sel.value)));
    $$(".avail input").forEach((cb) => cb.addEventListener("change", () => { state.available[cb.dataset.mod] = cb.checked; applyAvailability(); run(); }));
    $$(".seg button").forEach((b) => b.addEventListener("click", () => { state.fields = b.dataset.fields; $$(".seg button").forEach((x) => x.classList.toggle("on", x === b)); applyFieldView(); }));
    $$(".tabs button").forEach((b) => b.addEventListener("click", () => {
      $$(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
      $$(".tab").forEach((t) => t.classList.toggle("on", t.id === `tab-${b.dataset.tab}`));
      if (b.dataset.tab === "whatif") renderWhatIf();
      else if (state.scenario) { state.scenario = null; paint3D(state.last); }
    }));
    $$("[data-view]").forEach((b) => b.addEventListener("click", () => heart.view(b.dataset.view)));
    $("#tTerr").addEventListener("change", (e) => heart.setOptions({ territories: e.target.checked }));
    $("#tFlow").addEventListener("change", (e) => heart.setOptions({ flow: e.target.checked }));
    document.addEventListener("click", (e) => {
      const b = e.target.closest("[data-say]");
      if (b) speak(b.dataset.say);
      if (e.target.closest("#btnReport")) report();
    });
    $("#btnTour").addEventListener("click", () => tour(0));
    buildForm();
    renderModel();
    await Service.init();
    // deep links for demos/screenshots: ?preset=1&off=labs,echo&tab=why&target=LAD&view=lateral
    const q = new URLSearchParams(location.search);
    (q.get("off") || "").split(",").filter(Boolean).forEach((m) => {
      state.available[m] = false;
      const cb = $(`.avail input[data-mod="${m}"]`);
      if (cb) cb.checked = false;
    });
    applyAvailability();
    if (q.get("target")) state.whyTarget = q.get("target");
    const pi = q.has("preset") ? Number(q.get("preset")) : 2;
    sel.value = String(pi);
    loadPreset(pi);
    if (q.get("tab")) $(`.tabs button[data-tab="${q.get("tab")}"]`)?.click();
    if (q.get("view")) heart.view(q.get("view"));
    let seen = "1";
    try { seen = localStorage.getItem("hr3d_tour") || ""; } catch (e) { /* storage blocked */ }
    if (!seen && !q.toString()) setTimeout(() => tour(0), 900);
  }

  function loadPreset(i) {
    state.values = Object.assign({}, B.presets[i].values);
    state.scenario = null;
    writeForm();
    run();
  }

  window.addEventListener("DOMContentLoaded", start);
})();
