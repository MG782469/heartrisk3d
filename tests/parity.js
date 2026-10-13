// Parity test: browser engine (web/js/engine.js) must reproduce the Python pipeline probabilities.
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "..");
const { Engine } = require(path.join(root, "web/js/engine.js"));
const bundle = JSON.parse(fs.readFileSync(path.join(root, "artifacts/model_bundle.json"), "utf8"));
const fx = JSON.parse(fs.readFileSync(path.join(root, "artifacts/reports/parity_fixture.json"), "utf8"));
const eng = new Engine(bundle);
const names = bundle.configs[fx.set].features;
let maxDiff = 0, n = 0;
fx.X.forEach((row, i) => {
  const values = {};
  names.forEach((nm, j) => (values[nm] = row[j]));
  const out = eng.predict(values, { ecg: true, labs: true, echo: true });
  for (const t of bundle.targets) {
    maxDiff = Math.max(maxDiff, Math.abs(out.targets[t].p - fx.expected[t][i]));
    n++;
  }
});
console.log(`parity: ${n} predictions, max |diff| = ${maxDiff.toExponential(2)}`);
process.exit(maxDiff < 1e-6 ? 0 : 1);
