/* Tiny dependency-free SVG line charts for the Model tab. */
(function () {
  "use strict";
  // series: [{x:[], y:[], color, dash, label}], opts: {xlab, ylab, xmin, xmax, ymin, ymax, diag}
  function line(series, opts) {
    const W = 300, H = 190, L = 34, R = 8, Tp = 10, B = 28;
    const o = Object.assign({ xmin: 0, xmax: 1, ymin: 0, ymax: 1 }, opts || {});
    const sx = (v) => L + ((v - o.xmin) / (o.xmax - o.xmin)) * (W - L - R);
    const sy = (v) => H - B - ((v - o.ymin) / (o.ymax - o.ymin)) * (H - Tp - B);
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.title || "chart"}">`;
    for (let i = 0; i <= 4; i++) {
      const yv = o.ymin + ((o.ymax - o.ymin) * i) / 4, xv = o.xmin + ((o.xmax - o.xmin) * i) / 4;
      s += `<line x1="${L}" x2="${W - R}" y1="${sy(yv)}" y2="${sy(yv)}" stroke="#22363d"/>`;
      s += `<text x="${L - 4}" y="${sy(yv) + 3}" text-anchor="end">${+yv.toFixed(2)}</text>`;
      s += `<text x="${sx(xv)}" y="${H - B + 12}" text-anchor="middle">${+xv.toFixed(2)}</text>`;
    }
    if (o.diag) s += `<line x1="${sx(o.xmin)}" y1="${sy(o.ymin)}" x2="${sx(o.xmax)}" y2="${sy(o.ymax)}" stroke="#55707a" stroke-dasharray="3 3"/>`;
    series.forEach((se) => {
      const pts = se.x.map((x, i) => `${sx(x).toFixed(1)},${sy(Math.max(o.ymin, Math.min(o.ymax, se.y[i]))).toFixed(1)}`).join(" ");
      s += `<polyline fill="none" stroke="${se.color}" stroke-width="2" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""} points="${pts}"/>`;
      if (se.dots) se.x.forEach((x, i) => (s += `<circle cx="${sx(x)}" cy="${sy(se.y[i])}" r="3" fill="${se.color}"/>`));
    });
    s += `<text x="${(L + W - R) / 2}" y="${H - 3}" text-anchor="middle">${o.xlab || ""}</text>`;
    s += `<text x="10" y="${(Tp + H - B) / 2}" text-anchor="middle" transform="rotate(-90 10 ${(Tp + H - B) / 2})">${o.ylab || ""}</text>`;
    let lx = L + 6;
    series.forEach((se) => {
      if (!se.label) return;
      s += `<rect x="${lx}" y="${Tp + 2}" width="9" height="3" fill="${se.color}"/><text x="${lx + 12}" y="${Tp + 7}">${se.label}</text>`;
      lx += 12 + se.label.length * 5 + 10;
    });
    return s + "</svg>";
  }
  window.Charts = { line };
})();
