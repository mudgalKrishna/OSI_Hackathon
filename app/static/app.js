/* NitroTrace single-page dashboard. Plain JS, no build step. */
"use strict";

const state = { record: null, result: null, stale: false, cases: [], sources: {} };
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x) => (x === null || x === undefined || Number.isNaN(x)) ? "—" :
  Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(3) : Number(x.toPrecision(4)).toString();

const UNITS = {
  amount: ["nmol", "µmol", "mmol", "pmol", "mol"],
  mass: ["µg", "ng", "mg", "g"],
  concentration_molar: ["µM", "nM", "mM", "M"],
  concentration_mass: ["µg/L", "mg/L", "ng/mL", "g/L"],
};
const COVERAGE_TEXT = {
  quantified: "quantified", unknown: "unknown", not_applicable: "declared n/a", specified_synthetic: "synthetic-specified",
  ambiguous_overlap: "ambiguous overlap", partial_species: "partial species", partial_feeds: "a feed unbounded", scenario_only: "what-if only",
};

function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 2200); }

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let detail = r.statusText;
    try { const j = await r.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail, null, 1); } catch { /* ignore */ }
    throw new Error(detail);
  }
  return r;
}

/* ------------------------------------------------------------------ tabs */
function showTab(name) {
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

/* --------------------------------------------------------------- loading */
async function init() {
  try {
    const [cases, src] = await Promise.all([api("/cases").then((r) => r.json()), api("/sources").then((r) => r.json())]);
    state.cases = cases; state.sources = src.sources;
    renderStart();
  } catch (e) { toast("Could not reach the server: " + e.message); }
  loadBenchmark();
}

function renderStart() {
  const demo = ["synthetic_gas_budget_01", "synthetic_gas_budget_02", "synthetic_gas_budget_03"];
  const blurbs = {
    synthetic_gas_budget_01: "200 nmol observed; 50 mL/min for 120 min with ≤1 ppm delivered impurity → ≈245 nmol budget.",
    synthetic_gas_budget_02: "Same experiment, impurity bound tightened to 0.01 ppm → ≈2.45 nmol.",
    synthetic_gas_budget_03: "Impurity measurement removed → the gas source becomes unknown.",
  };
  $("#demoCards").innerHTML = demo.map((id, i) => {
    const c = state.cases.find((x) => x.case_id === id);
    return c ? `<button class="card" data-load="${id}"><span class="t">Step ${i + 1}. ${esc(c.title.replace(/^Demo step \d — /, ""))}</span>
      <span class="d">${esc(blurbs[id])}</span></button>` : "";
  }).join("");
  $("#publishedCards").innerHTML = state.cases.filter((c) => c.group === "published").map((c) =>
    `<button class="card" data-load="${c.case_id}"><span class="t">${esc(c.title)}</span>
     <span class="d"><span class="tag ${c.record_type === "published" ? "published" : ""}">${esc(c.record_type.replace("_", " "))}</span></span></button>`).join("");
  $("#syntheticSelect").innerHTML = state.cases.filter((c) => c.group === "synthetic").map((c) =>
    `<option value="${c.case_id}">${esc(c.title)}</option>`).join("");
  $$("[data-load]").forEach((b) => b.addEventListener("click", () => loadCase(b.dataset.load)));
}

async function loadCase(id) {
  const rec = await api(`/cases/${id}`).then((r) => r.json());
  setRecord(rec);
  await runAudit();
  showTab("audit");
}

function setRecord(rec) {
  // Optional collections may be omitted in imported JSON.
  for (const [k, v] of [["gas_sources", []], ["controls", []], ["declared_coverage", {}], ["assumptions", []],
    ["sources", []], ["isotope", {}], ["context", {}]]) if (rec[k] == null) rec[k] = v;
  state.record = rec; state.result = null; state.stale = false;
  $("#caseLabel").textContent = `${rec.title}  ·  ${rec.record_type}`;
  $("#runBtn").disabled = false;
  renderEditor(); renderAudit(); renderControls(); renderReport();
}

$("#loadSynthetic").addEventListener("click", () => loadCase($("#syntheticSelect").value));
$("#newRecord").addEventListener("click", () => { setRecord(blankRecord()); showTab("experiment"); });
$("#importJson").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try { const j = JSON.parse(await f.text()); setRecord(j.record || j); showTab("experiment"); toast("Record imported"); }
  catch (err) { toast("Invalid JSON: " + err.message); }
  e.target.value = "";
});
$("#runBtn").addEventListener("click", async () => { await runAudit(); showTab("audit"); });

function blankRecord() {
  return {
    case_id: "user_record_" + Date.now().toString(36), title: "New experiment record", record_type: "user", description: "",
    observed_product: { species: "NH3", basis: "amount", unit: "nmol", lower: null, upper: null, bound_type: "stated_interval",
      nondetect: false, volume_mL: null, dilution_factor: 1, interval: { start_min: 0, end_min: 120 }, n_replicates: null,
      evidence: { provenance: "manual", confirmed: false } },
    gas_sources: [newGas()], controls: [], declared_coverage: {}, coverage_model: null,
    isotope: { performed: false, quantitative: false, time_series: false, labeled_impurities_quantified: false },
    context: { flowing_gas: true }, assumptions: [], sources: [],
  };
}
function newGas() {
  return { id: "gas_" + Math.random().toString(36).slice(2, 6), label: "N2 feed gas",
    impurity: null, flow: { value: 50, unit: "mL/min", volume_basis: "actual", temperature_K: 298.15, pressure_Pa: 101325,
      evidence: { provenance: "manual", confirmed: false } },
    exposure: null, eta_upper: 1, eta_lower: 0, eta_basis: "conservative_upper_bound_1", evidence: { provenance: "manual", confirmed: false } };
}
function newImpurity() {
  return { species: "NOx", nitrogen_atoms: 1, unit: "ppm", lower: 0, upper: null, bound_type: "hard_limit",
    measured_where: "delivered_to_cell", covers_all_reactive_nitrogen: false, evidence: { provenance: "manual", confirmed: false } };
}
function newControl() {
  return { id: "ctl_" + Math.random().toString(36).slice(2, 6), label: "Argon background", kind: "system_blank",
    covers: ["electrolyte", "components"],
    measurement: { species: "NH3", basis: "amount", unit: "nmol", lower: null, upper: null, bound_type: "stated_interval",
      nondetect: false, interval: { start_min: 0, end_min: 120 }, evidence: { provenance: "manual", confirmed: false } },
    conditions_match: "unknown", include_as_scenario: false, time_scaling: "none", contains_nafion: false };
}

/* ---------------------------------------------------------------- editor */
function getPath(obj, path) { return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj); }
function setPath(obj, path, value) {
  const keys = path.split("."); let o = obj;
  keys.slice(0, -1).forEach((k) => { if (o[k] == null) o[k] = {}; o = o[k]; });
  o[keys.at(-1)] = value;
}

function field(path, label, type = "text", opts = {}) {
  const v = getPath(state.record, path);
  const id = "f_" + path.replace(/\W/g, "_");
  if (type === "check") {
    return `<div class="field check ${opts.cls || ""}"><input type="checkbox" id="${id}" data-path="${path}" data-type="check" ${v ? "checked" : ""}>
      <label for="${id}">${esc(label)}</label></div>`;
  }
  let input;
  if (type === "select") {
    const options = opts.options.map((o) => Array.isArray(o) ? o : [o, o]);
    input = `<select id="${id}" data-path="${path}" data-type="${opts.valueType || "text"}">${options.map(([val, txt]) =>
      `<option value="${esc(val)}" ${String(v ?? "") === String(val) ? "selected" : ""}>${esc(txt)}</option>`).join("")}</select>`;
  } else if (type === "textarea") {
    input = `<textarea id="${id}" data-path="${path}" data-type="text" rows="${opts.rows || 2}">${esc(v ?? "")}</textarea>`;
  } else {
    input = `<input id="${id}" data-path="${path}" data-type="${type}" type="${type === "number" ? "number" : "text"}" step="any"
      value="${esc(v ?? "")}" placeholder="${esc(opts.placeholder || (type === "number" ? "unknown" : ""))}">`;
  }
  return `<div class="field ${opts.cls || ""}"><label for="${id}">${esc(label)}</label>${input}</div>`;
}

function evidenceBlock(path) {
  const ev = getPath(state.record, path) || {};
  const src = ev.source_id && state.sources[ev.source_id];
  const link = src ? `<a href="${esc(ev.url || src.open_full_text || src.url)}" target="_blank" rel="noopener">[${esc(ev.source_id)}]</a>` : "";
  return `<details class="evidence"><summary><span class="conf ${ev.confirmed ? "yes" : "no"}">${ev.confirmed ? "confirmed" : "unconfirmed"}</span>
    <span>${esc(ev.provenance || "manual")}</span> ${link} <span class="muted">${esc(ev.location || "")}</span></summary>
    ${ev.quote ? `<blockquote>“${esc(ev.quote)}”</blockquote>` : ""}
    ${ev.note ? `<p class="muted" style="font-size:.84rem">${esc(ev.note)}</p>` : ""}
    <div class="grid" style="margin-top:.5rem">
      ${field(path + ".source_id", "Source id", "select", { options: [["", "—"], ...Object.keys(state.sources)] })}
      ${field(path + ".location", "Page / table / figure")}
      ${field(path + ".provenance", "Provenance", "select", { options: ["reported", "manual", "derived", "assumed", "synthetic"] })}
      ${field(path + ".confirmed", "Human-confirmed", "check")}
      ${field(path + ".quote", "Supporting passage", "textarea", { cls: "wide" })}
      ${field(path + ".note", "Note", "textarea", { cls: "wide" })}
    </div></details>`;
}

function measurementFields(p, title) {
  const m = getPath(state.record, p);
  const basis = m.basis || "";
  const units = UNITS[basis] || [m.unit];
  const conc = basis.startsWith("concentration");
  return `<div class="grid">
    ${field(p + ".species", "Species", "select", { options: [["", "— not declared —"], "NH3", "NH4+", "N"] })}
    ${field(p + ".basis", "Reporting basis", "select", { options: [["", "— not declared —"], "amount", "mass", "concentration_molar", "concentration_mass"] })}
    ${field(p + ".unit", "Unit", "select", { options: units.includes(m.unit) ? units : [m.unit, ...units] })}
    ${field(p + ".lower", m.nondetect ? "Lower (0 for nondetect)" : "Lower bound", "number")}
    ${field(p + ".upper", m.nondetect ? "Detection limit" : "Upper bound", "number")}
    ${field(p + ".bound_type", "Bound type", "select", { options: ["stated_interval", "hard_limit", "detection_limit", "assumed_range", "point"] })}
    ${field(p + ".nondetect", "Nondetect", "check")}
    ${conc ? field(p + ".volume_mL", "Volume (mL)", "number") : ""}
    ${conc ? field(p + ".dilution_factor", "Dilution factor", "number") : ""}
    ${field(p + ".interval.start_min", "Interval start (min)", "number")}
    ${field(p + ".interval.end_min", "Interval end (min)", "number")}
    ${title === "product" ? field(p + ".n_replicates", "Independent replicates", "number") : ""}
  </div>${evidenceBlock(p + ".evidence")}`;
}

function renderEditor() {
  const r = state.record;
  $("#tab-experiment [data-empty]").style.display = r ? "none" : "";
  if (!r) { $("#editor").innerHTML = ""; return; }
  const op = r.observed_product;
  const gas = r.gas_sources.map((g, i) => {
    const p = `gas_sources.${i}`;
    const imp = g.impurity
      ? `<div class="grid">
          ${field(p + ".impurity.species", "Impurity species")}
          ${field(p + ".impurity.nitrogen_atoms", "N atoms / molecule", "number")}
          ${field(p + ".impurity.unit", "Unit", "select", { options: ["ppm", "ppb", "mole_fraction", "mol_percent"] })}
          ${field(p + ".impurity.lower", "Lower bound", "number")}
          ${field(p + ".impurity.upper", "Upper bound", "number")}
          ${field(p + ".impurity.measured_where", "Measured where", "select", { options: [["delivered_to_cell", "delivered to cell"], ["upstream_of_purifier", "upstream of purifier"], ["supplier_specification", "supplier specification"], ["unknown", "unknown"]] })}
          ${field(p + ".impurity.covers_all_reactive_nitrogen", "Bound covers all reactive-N species", "check", { cls: "wide" })}
         </div>${evidenceBlock(p + ".impurity.evidence")}
         <button class="small" data-act="rm-imp" data-i="${i}" style="margin-top:.5rem">Remove impurity measurement</button>`
      : `<p class="missing">No impurity measurement — this gas source is unknown, not zero.</p>
         <button class="small" data-act="add-imp" data-i="${i}">Add impurity measurement</button>`;
    const flow = g.flow
      ? `<div class="grid">
          ${field(p + ".flow.value", "Flow", "number")}
          ${field(p + ".flow.unit", "Flow unit", "select", { options: ["mL/min", "L/min", "sccm", "slpm"] })}
          ${field(p + ".flow.volume_basis", "Volume basis", "select", { options: ["actual", "standard"] })}
          ${field(p + ".flow.temperature_K", "Gas T (K)", "number")}
          ${field(p + ".flow.pressure_Pa", "Gas P (Pa)", "number")}
          ${g.flow.volume_basis === "standard" ? field(p + ".flow.reference_temperature_K", "Controller ref. T (K)", "number") + field(p + ".flow.reference_pressure_Pa", "Controller ref. P (Pa)", "number") : ""}
         </div>`
      : g.total_volume_L != null
        ? `<div class="grid">${field(p + ".total_volume_L", "Total gas volume (L)", "number")}${field(p + ".total_volume_temperature_K", "T (K)", "number")}${field(p + ".total_volume_pressure_Pa", "P (Pa)", "number")}</div>`
        : `<p class="missing">Gas flow during the measurement not stated.</p><button class="small" data-act="add-flow" data-i="${i}">Add flow</button>`;
    return `<div class="sub"><header class="row" style="justify-content:space-between"><h3>${esc(g.label)}</h3>
        <button class="small" data-act="rm-gas" data-i="${i}">Remove source</button></header>
      <div class="grid">${field(p + ".label", "Label", "text", { cls: "wide" })}
        ${field(p + ".exposure.start_min", "Exposure start (min)", "number", { placeholder: "= collection" })}
        ${field(p + ".exposure.end_min", "Exposure end (min)", "number", { placeholder: "= collection" })}
        ${field(p + ".eta_upper", "η upper (fraction to product)", "number", { placeholder: "1 (conservative)" })}
        ${field(p + ".eta_basis", "η basis", "select", { options: ["conservative_upper_bound_1", "measured", "assumed"] })}</div>
      <h3 style="margin-top:.7rem">Impurity</h3>${imp}
      <h3 style="margin-top:.7rem">Delivery</h3>${flow}${g.flow ? evidenceBlock(p + ".flow.evidence") : ""}</div>`;
  }).join("");

  const ctls = r.controls.map((c, i) => {
    const p = `controls.${i}`;
    const covers = ["gas", "electrolyte", "components"].map((cat) =>
      `<label class="field check"><input type="checkbox" data-cover="${i}" value="${cat}" ${c.covers.includes(cat) ? "checked" : ""}> ${cat}</label>`).join("");
    return `<div class="sub"><header class="row" style="justify-content:space-between"><h3>${esc(c.label)}</h3>
        <button class="small" data-act="rm-ctl" data-i="${i}">Remove control</button></header>
      <div class="grid">${field(p + ".label", "Label", "text", { cls: "wide" })}
        ${field(p + ".kind", "Kind", "select", { options: ["system_blank", "electrolyte_blank", "component_release"] })}
        ${field(p + ".conditions_match", "Matched to experiment?", "select", { options: ["matched", "different", "unknown"] })}
        ${field(p + ".time_scaling", "Time scaling", "select", { options: ["none", "linear"] })}
        ${field(p + ".include_as_scenario", "Use as what-if if unmatched", "check")}
        ${field(p + ".contains_nafion", "Contains Nafion", "check")}
        ${field(p + ".conditions_note", "Conditions note", "text", { cls: "wide" })}
        <div class="field wide"><label>Covers source categories</label><div class="row">${covers}</div></div></div>
      ${measurementFields(p + ".measurement", "control")}</div>`;
  }).join("");

  const cov = ["gas", "electrolyte", "components"].map((cat) => field(`declared_coverage.${cat}`, `Declared: ${cat}`, "select",
    { options: [["", "— from measurements —"], ["not_applicable", "not applicable (assumption)"], ["specified_synthetic", "specified (synthetic only)"]] })).join("");

  $("#editor").innerHTML = `
    <div class="panel"><header><h2 style="margin:0">Record</h2><span class="tag ${r.record_type === "synthetic" ? "synthetic" : "published"}">${esc(r.record_type)}</span></header>
      <div class="grid">${field("title", "Title", "text", { cls: "wide" })}${field("case_id", "Case id")}
      ${field("description", "Description", "textarea", { cls: "wide", rows: 3 })}</div></div>

    <div class="panel"><header><h2 style="margin:0">Measured product</h2>
      ${op ? `<button class="small" data-act="rm-obs">Remove</button>` : `<button class="small" data-act="add-obs">Add product measurement</button>`}</header>
      ${op ? measurementFields("observed_product", "product") : `<p class="missing">No product measurement entered.</p>`}</div>

    <div class="panel"><header><h2 style="margin:0">Gas-delivered sources</h2><button class="small" data-act="add-gas">Add gas source</button></header>
      ${gas || `<p class="missing">No gas source entered — gas coverage is unknown.</p>`}</div>

    <div class="panel"><header><h2 style="margin:0">Blanks, backgrounds &amp; component tests</h2><button class="small" data-act="add-ctl">Add control</button></header>
      ${ctls || `<p class="muted">No control measurements entered.</p>`}</div>

    <div class="panel"><header><h2 style="margin:0">Coverage &amp; aggregation</h2></header>
      <div class="grid">${cov}
      ${field("coverage_model", "Overlap model", "select", { valueType: "nullable", options: [["", "— none selected —"], ["system_blank", "use whole-system blank"], ["sum_components", "sum individual components"]] })}</div></div>

    <div class="panel"><header><h2 style="margin:0">Isotope evidence &amp; context</h2></header>
      <p class="muted" style="margin-top:0">Isotope evidence is kept separate from the contamination budget.</p>
      <div class="grid">${field("isotope.performed", "15N2 experiment performed", "check")}${field("isotope.quantitative", "Quantitative", "check")}
        ${field("isotope.time_series", "Several durations", "check")}${field("isotope.labeled_impurities_quantified", "15NOx/15NH3 in gas quantified", "check")}
        ${field("context.flowing_gas", "Continuous gas flow", "check")}${field("context.nox_measured_alongside", "NOx measured alongside NH3", "check")}
        ${field("context.electrode_contains_nitrogen", "Electrode/catalyst contains N", "check")}${field("context.gde_contains_nafion", "GDE contains Nafion", "check")}</div></div>

    <div class="panel"><header><h2 style="margin:0">Assumptions</h2></header>
      <textarea id="assumptions" rows="4">${esc((r.assumptions || []).join("\n"))}</textarea>
      <p class="muted" style="font-size:.8rem">One per line.</p></div>

    <details class="panel"><summary><strong>Raw JSON record</strong></summary>
      <textarea id="rawJson" rows="18" style="margin-top:.6rem">${esc(JSON.stringify(r, null, 2))}</textarea>
      <button id="applyJson" style="margin-top:.5rem">Apply JSON</button></details>`;
  bindEditor();
}

function bindEditor() {
  $$("#editor [data-path]").forEach((el) => el.addEventListener("change", () => {
    const t = el.dataset.type; let v;
    if (t === "check") v = el.checked;
    else if (t === "number") v = el.value === "" ? null : Number(el.value);
    else if (t === "nullable") v = el.value === "" ? null : el.value;
    else v = el.value === "" && /species|basis|declared_coverage|source_id/.test(el.dataset.path) ? null : el.value;
    const path = el.dataset.path;
    if (/\.exposure\.(start|end)_min$/.test(path)) {
      const gp = path.replace(/\.exposure\..*$/, "");
      const g = getPath(state.record, gp);
      if (v === null) g.exposure = null;
      else { g.exposure = g.exposure || { start_min: 0, end_min: v }; setPath(state.record, path, v); }
    } else if (path.startsWith("declared_coverage.") && v === null) {
      delete state.record.declared_coverage[path.split(".")[1]];
    } else setPath(state.record, path, v);
    if (/\.basis$/.test(path)) { const m = getPath(state.record, path.replace(/\.basis$/, "")); if (UNITS[v]) m.unit = UNITS[v][0]; }
    if (/\.nondetect$/.test(path) && v) setPath(state.record, path.replace(/nondetect$/, "lower"), null);
    markStale();
    if (/basis|nondetect|volume_basis|label|kind/.test(path)) renderEditor();
  }));
  $$("#editor [data-cover]").forEach((el) => el.addEventListener("change", () => {
    const c = state.record.controls[el.dataset.cover];
    c.covers = $$(`#editor [data-cover="${el.dataset.cover}"]`).filter((x) => x.checked).map((x) => x.value);
    markStale();
  }));
  $$("#editor [data-act]").forEach((b) => b.addEventListener("click", () => {
    const r = state.record, i = Number(b.dataset.i);
    ({
      "rm-imp": () => { r.gas_sources[i].impurity = null; },
      "add-imp": () => { r.gas_sources[i].impurity = newImpurity(); },
      "add-flow": () => { r.gas_sources[i].flow = newGas().flow; },
      "rm-gas": () => { r.gas_sources.splice(i, 1); },
      "add-gas": () => { r.gas_sources.push(newGas()); },
      "rm-ctl": () => { r.controls.splice(i, 1); },
      "add-ctl": () => { r.controls.push(newControl()); },
      "rm-obs": () => { r.observed_product = null; },
      "add-obs": () => { r.observed_product = blankRecord().observed_product; },
    })[b.dataset.act]();
    markStale(); renderEditor();
  }));
  $("#assumptions").addEventListener("change", (e) => { state.record.assumptions = e.target.value.split("\n").map((s) => s.trim()).filter(Boolean); markStale(); });
  $("#applyJson").addEventListener("click", () => {
    try { setRecord(JSON.parse($("#rawJson").value)); toast("JSON applied"); } catch (e) { toast("Invalid JSON: " + e.message); }
  });
}

function markStale() {
  state.stale = true;
  $$(".stale-flag").forEach((e) => e.remove());
  if (state.result) $("#auditView").insertAdjacentHTML("afterbegin", `<p class="stale stale-flag">Inputs changed — press “Run evidence audit” to recompute.</p>`);
}

/* ----------------------------------------------------------------- audit */
async function runAudit() {
  if (!state.record) return;
  $("#runBtn").disabled = true;
  try {
    const r = await api("/audit", { method: "POST", body: JSON.stringify(state.record) });
    state.result = await r.json(); state.stale = false;
  } catch (e) {
    state.result = null;
    $("#auditView").innerHTML = `<div class="status input_inconsistency"><div class="label">Record could not be validated</div><pre class="md">${esc(e.message)}</pre></div>`;
    $("#tab-audit [data-empty]").style.display = "none";
    $("#runBtn").disabled = false;
    return;
  }
  $("#runBtn").disabled = false;
  renderAudit(); renderControls(); renderReport();
}

function unconfirmedCount(rec) {
  let n = 0;
  const walk = (o) => { if (o && typeof o === "object") { if ("provenance" in o && "confirmed" in o && !o.confirmed) n++; Object.values(o).forEach(walk); } };
  walk(rec); return n;
}

function chartSVG(res) {
  const obs = res.observed_nmol, valid = res.budget_valid_nmol, scen = res.budget_with_scenarios_nmol;
  const rows = [];
  if (obs) rows.push({ label: "Measured ammonia", lo: obs.lower, hi: obs.upper, color: "var(--obs)" });
  rows.push({ label: res.budget_complete ? "Contamination budget" : "Quantified budget (partial)", lo: valid.lower, hi: valid.upper, color: "var(--budget)" });
  if (scen.upper !== valid.upper) rows.push({ label: "incl. what-if (not valid)", lo: scen.lower, hi: scen.upper, color: "var(--scenario)", dashed: true });
  const vals = rows.flatMap((r) => [r.lo, r.hi]).filter((v) => v > 0);
  const maxV = Math.max(...vals, 1e-6), minPos = Math.min(...vals, maxV);
  const useLog = vals.length > 1 && maxV / minPos > 30;
  const W = 720, L = 190, R = 30, rowH = 46, top = 16, H = top + rows.length * rowH + 34;
  let x, ticks;
  if (useLog) {
    const lo = Math.floor(Math.log10(minPos)) , hi = Math.ceil(Math.log10(maxV));
    const a = Math.min(lo, hi - 1);
    x = (v) => L + (Math.log10(Math.max(v, 10 ** a)) - a) / (hi - a) * (W - L - R);
    ticks = []; for (let e = a; e <= hi; e++) ticks.push(10 ** e);
  } else {
    const hi = maxV * 1.12;
    x = (v) => L + v / hi * (W - L - R);
    const step = 10 ** Math.floor(Math.log10(hi / 4)); const n = Math.ceil(hi / step);
    const k = n > 8 ? 2 * step : step;
    ticks = []; for (let t = 0; t <= hi; t += k) ticks.push(t);
  }
  const g = rows.map((r, i) => {
    const y = top + i * rowH + 10;
    const x1 = x(r.lo), x2 = Math.max(x(r.hi), x1 + 3);
    const zero = r.hi === 0;
    return `<text x="${L - 10}" y="${y + 14}" text-anchor="end">${esc(r.label)}</text>
      ${zero ? `<text x="${L + 4}" y="${y + 14}" style="fill:var(--muted)">0 nmol N — no valid bound</text>` :
        `<rect x="${x1}" y="${y}" width="${x2 - x1}" height="20" rx="4" fill="${r.color}" fill-opacity="${r.dashed ? 0.18 : 0.85}"
          stroke="${r.color}" ${r.dashed ? 'stroke-dasharray="5 3"' : ""}/>
         <text x="${Math.min(x2 + 6, W - R - 70)}" y="${y + 14}" style="font-size:11px">${fmt(r.lo)}–${fmt(r.hi)}</text>`}`;
  }).join("");
  const axisY = top + rows.length * rowH + 6;
  const tk = ticks.map((t) => `<g class="tick"><line x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${axisY}" stroke="var(--border)" stroke-dasharray="2 4"/>
      <text x="${x(t)}" y="${axisY + 16}" text-anchor="middle">${fmt(t)}</text></g>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Measured ammonia versus contamination budget">
    ${tk}<line x1="${L}" x2="${W - R}" y1="${axisY}" y2="${axisY}" stroke="var(--border)"/>${g}
    <text x="${W - R}" y="${H - 2}" text-anchor="end" style="fill:var(--muted);font-size:11px">nmol N${useLog ? " (log scale)" : ""}</text></svg>`;
}

function renderAudit() {
  const res = state.result;
  $("#tab-audit [data-empty]").style.display = res ? "none" : "";
  if (!res) { if (!state.record) $("#auditView").innerHTML = ""; return; }
  const unc = unconfirmedCount(state.record);
  const chips = Object.entries(res.coverage).map(([k, v]) => `<span class="chip ${v}">${k}: ${COVERAGE_TEXT[v] || v}</span>`).join("");
  const unq = res.unquantified.length
    ? `<div class="unq"><h3>Unquantified sources</h3><ul>${res.unquantified.map((u) => `<li><strong>${esc(u.category)}</strong> — ${esc(u.reason)}</li>`).join("")}</ul>
       <p class="muted" style="font-size:.8rem;margin:0">These are not zero. They are omitted from the plotted budget because no valid bound exists.</p></div>`
    : `<div class="unq none"><h3>Unquantified sources</h3><p style="margin:0;font-size:.88rem">None among the modeled categories (under stated assumptions).</p></div>`;
  const contrib = res.contributions.map((c) => `<tr><td>${esc(c.label)}</td><td>${esc(c.category)}</td>
    <td><span class="role ${c.role}">${c.role.replace(/_/g, " ")}</span></td><td class="num">${fmt(c.lower_nmol)}</td><td class="num">${fmt(c.upper_nmol)}</td>
    <td class="muted">${esc(c.reason || "")}</td></tr>`).join("");
  const trail = [...res.observed_trail, ...res.contributions.flatMap((c) => c.trail)].map((s) =>
    `<li><strong>${esc(s.label)}</strong>${s.formula ? ` — <code>${esc(s.formula)}</code>` : ""}<br>
     ${Object.entries(s.values).map(([k, v]) => `<code>${esc(k)}: ${esc(v)}</code>`).join("<br>")}${s.result ? `<br>→ <strong>${esc(s.result)}</strong>` : ""}</li>`).join("");
  const fa = res.feasible_allocation;
  $("#auditView").innerHTML = `
    <div class="status ${res.status}"><div class="label">${esc(res.status_label)}</div><p style="margin:.3rem 0 0">${esc(res.headline)}</p>
      <div class="chips">${chips}</div></div>
    ${unc ? `<p class="stale">${unc} input value(s) are not yet human-confirmed (see Experiment &amp; evidence).</p>` : ""}
    <div class="two">
      <div class="panel"><header><h2 style="margin:0">Measured ammonia vs contamination budget</h2></header>
        ${res.budget_complete ? "" : `<div class="partial-label">Partial budget — incomplete source coverage</div>`}
        <div class="chart-wrap">${chartSVG(res)}</div>
        <div class="legend"><span>■ <span style="color:var(--obs)">measured interval</span></span><span>■ <span style="color:var(--budget)">valid budget bounds</span></span>
          <span>▭ what-if measurements (not interchangeable)</span></div>
        <ul style="margin:.8rem 0 0;padding-left:1.1rem">${res.explanation.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>
      </div>
      <div>${unq}
        ${fa ? `<div class="panel" style="margin-top:1rem"><h3>Illustrative feasible explanation</h3>
          <p class="muted" style="font-size:.82rem;margin:.2rem 0 .5rem">${esc(fa.label)}</p>
          <ul style="margin:0;padding-left:1.1rem">${Object.entries(fa.allocation_nmol).map(([k, v]) => `<li>${esc(k)}: ${fmt(v)} nmol N</li>`).join("")}</ul></div>` : ""}
      </div>
    </div>
    <div class="panel"><header><h2 style="margin:0">Contributions</h2><span class="muted">${esc(res.budget_label)}</span></header>
      <div class="table-wrap"><table><thead><tr><th>Source</th><th>Category</th><th>Role</th><th class="num">Lower (nmol N)</th><th class="num">Upper (nmol N)</th><th>Note</th></tr></thead>
      <tbody>${res.observed_nmol ? `<tr><td><strong>Observed product</strong></td><td></td><td></td><td class="num">${fmt(res.observed_nmol.lower)}</td><td class="num">${fmt(res.observed_nmol.upper)}</td><td></td></tr>` : ""}
      ${contrib}<tr><td><strong>Valid budget</strong></td><td></td><td></td><td class="num"><strong>${fmt(res.budget_valid_nmol.lower)}</strong></td><td class="num"><strong>${fmt(res.budget_valid_nmol.upper)}</strong></td><td></td></tr></tbody></table></div></div>
    ${res.issues.length ? `<div class="panel"><h2 style="margin-top:0">Issues</h2>${res.issues.map((i) => `<div class="issue ${i.severity}"><code>${esc(i.code)}</code> ${esc(i.message)}</div>`).join("")}</div>` : ""}
    <details class="panel" open><summary><strong>Calculation trail</strong></summary><ol class="trail">${trail || "<li>No calculations.</li>"}</ol></details>
    <details class="panel"><summary><strong>Assumptions (${res.assumptions.length})</strong></summary><ul>${res.assumptions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></details>
    ${demoControls()}`;
  $$("[data-demo]").forEach((b) => b.addEventListener("click", () => demoAction(b.dataset.demo)));
}

function demoControls() {
  const g = state.record?.gas_sources?.[0];
  if (!state.record || state.record.record_type !== "synthetic" || !g) return "";
  return `<div class="panel"><h2 style="margin-top:0">Change the evidence</h2><div class="row">
    <button data-demo="1">Impurity ≤ 1 ppm</button><button data-demo="0.01">Tighten to ≤ 0.01 ppm</button>
    <button data-demo="remove">Remove impurity measurement</button></div></div>`;
}
async function demoAction(kind) {
  const g = state.record.gas_sources[0];
  if (kind === "remove") g.impurity = null;
  else { g.impurity = g.impurity || { ...newImpurity(), covers_all_reactive_nitrogen: true, species: "generic N-impurity (1 N atom)" }; g.impurity.upper = Number(kind); }
  renderEditor(); await runAudit();
}

/* -------------------------------------------------------------- controls */
function renderControls() {
  const res = state.result;
  $("#tab-controls [data-empty]").style.display = res ? "none" : "";
  if (!res) { $("#controlsView").innerHTML = ""; return; }
  $("#controlsView").innerHTML = `<p class="muted">Protocol-based suggestions triggered by this record. Each links to the passage that motivates it.
    They are not an optimized experimental design.</p>` +
    (res.recommendations.length ? res.recommendations.map((r) => `<div class="rec"><h3>${esc(r.title)}</h3>
      <div class="trigger">Trigger: ${esc(r.trigger)}</div><p style="margin:.2rem 0 .5rem">${esc(r.text)}</p>
      ${r.references.map((ref) => `<blockquote>“${esc(ref.quote)}”<cite>[${esc(ref.source_id)}] ${esc(ref.location)} —
        <a href="${esc(ref.url)}" target="_blank" rel="noopener">${esc(ref.citation || ref.source_id)}</a></cite></blockquote>`).join("")}</div>`).join("")
      : `<div class="empty">No rule triggered for this record.</div>`);
}

/* ---------------------------------------------------------------- report */
async function renderReport() {
  const res = state.result;
  if (!res) { $("#reportView").innerHTML = `<div class="empty">Run the audit to export a report.</div>`; return; }
  $("#reportView").innerHTML = `<div class="panel"><header><h2 style="margin:0">Downloads</h2></header>
    <div class="row"><button data-dl="json">Full record + audit (JSON)</button><button data-dl="csv">Contribution table (CSV)</button>
      <button data-dl="md">Report (Markdown)</button></div>
    <p class="muted" style="font-size:.82rem">Exports are generated from the same record and engine as the displayed result; re-importing the JSON reproduces it.</p>
    <h3>Markdown preview</h3><pre class="md" id="mdPreview">Loading…</pre></div>`;
  $$("[data-dl]").forEach((b) => b.addEventListener("click", () => download(b.dataset.dl)));
  try {
    const r = await api("/report/md", { method: "POST", body: JSON.stringify(state.record) });
    $("#mdPreview").textContent = await r.text();
  } catch (e) { $("#mdPreview").textContent = e.message; }
}
async function download(fmtName) {
  const r = await api(`/report/${fmtName}`, { method: "POST", body: JSON.stringify(state.record) });
  const blob = await r.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `nitrotrace_${state.record.case_id}.${fmtName}`;
  a.click(); URL.revokeObjectURL(a.href);
}

async function loadBenchmark() {
  try {
    const b = await api("/benchmark").then((r) => r.json());
    const s = b.synthetic;
    $("#benchmarkView").innerHTML = `<div class="panel"><p class="muted" style="margin-top:0">${esc(b.disclaimer)}</p>
      <div class="table-wrap"><table><tbody>
      <tr><td>Synthetic statuses correct</td><td class="num">${s.status_correct}/${s.n_records}</td></tr>
      <tr><td>Max relative numeric error (${s.numeric_checks} reference checks)</td><td class="num">${fmt(s.max_rel_error)}</td></tr>
      <tr><td>Unit / basis problems detected</td><td class="num">${s.unit_or_basis_problems_detected}</td></tr>
      <tr><td>Missing-data cases kept inconclusive</td><td class="num">${s.inconclusive_correct}</td></tr>
      <tr><td>Duplicate-source cases handled</td><td class="num">${s.duplicate_source_cases_correct}</td></tr>
      <tr><td>Naive baseline (sum everything, unknown = 0)</td><td class="num">${s.naive_baseline_correct}</td></tr>
      <tr><td>Median audit runtime</td><td class="num">${fmt(s.median_runtime_ms)} ms</td></tr>
      </tbody></table></div>
      ${s.failures.length ? `<p class="missing">Failures: ${s.failures.map((f) => esc(f.case_id)).join(", ")}</p>` : `<p class="muted">No synthetic failures.</p>`}
      <h3>Published records (not scored)</h3><div class="table-wrap"><table><thead><tr><th>Record</th><th>Status</th><th class="num">Budget ≤ (nmol N)</th></tr></thead><tbody>
      ${b.published.records.map((r) => `<tr><td>${esc(r.case_id)}${r.published_reference ? `<br><span class="muted">reproduces S2 Table 1: ${fmt(r.published_reference.engine_nmol)} vs ${fmt(r.published_reference.reference_nmol)} nmol</span>` : ""}</td>
        <td>${esc(r.status.replace(/_/g, " "))}</td><td class="num">${fmt(r.budget_upper_nmol)}</td></tr>`).join("")}
      </tbody></table></div></div>`;
  } catch { $("#benchmarkView").textContent = "Benchmark not generated yet — run `python scripts/evaluate.py`."; }
}

init();
