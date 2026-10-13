/* AHA 17-segment bullseye (Cerqueira et al., Circulation 2002) with standard coronary territories.
 * Shaded by the predicted probability of the vessel that TYPICALLY supplies each segment.
 * Illustrative: real supply varies with coronary dominance; this is not a lesion map. */
(function () {
  "use strict";
  const NAMES = {
    1: "basal anterior", 2: "basal anteroseptal", 3: "basal inferoseptal", 4: "basal inferior", 5: "basal inferolateral", 6: "basal anterolateral",
    7: "mid anterior", 8: "mid anteroseptal", 9: "mid inferoseptal", 10: "mid inferior", 11: "mid inferolateral", 12: "mid anterolateral",
    13: "apical anterior", 14: "apical septal", 15: "apical inferior", 16: "apical lateral", 17: "apex",
  };
  function arc(cx, cy, r0, r1, a0, a1) {
    const rad = (d) => (d * Math.PI) / 180;
    const P = (r, a) => [cx + r * Math.cos(rad(a)), cy - r * Math.sin(rad(a))];
    const [x0, y0] = P(r1, a0), [x1, y1] = P(r1, a1), [x2, y2] = P(r0, a1), [x3, y3] = P(r0, a0);
    const large = a1 - a0 > 180 ? 1 : 0;
    return `M${x0},${y0} A${r1},${r1} 0 ${large} 0 ${x1},${y1} L${x2},${y2} A${r0},${r0} 0 ${large} 1 ${x3},${y3} Z`;
  }
  function segments() {
    const s = [];
    // basal 1-6 and mid 7-12: 60 deg each, segment 1/7 centred at the top (anterior), counter-clockwise to septum
    for (let ring = 0; ring < 2; ring++)
      for (let i = 0; i < 6; i++) {
        const a0 = 60 + 60 * i;
        s.push({ id: ring * 6 + i + 1, r0: ring === 0 ? 72 : 46, r1: ring === 0 ? 98 : 72, a0, a1: a0 + 60 });
      }
    // apical 13-16: 90 deg each
    for (let i = 0; i < 4; i++) {
      const a0 = 45 + 90 * i;
      s.push({ id: 13 + i, r0: 20, r1: 46, a0, a1: a0 + 90 });
    }
    return s;
  }
  function render(el, risk, territories, colorOf) {
    const owner = {};
    for (const [v, list] of Object.entries(territories)) list.forEach((id) => (owner[id] = v));
    const cx = 110, cy = 110;
    let svg = `<svg viewBox="0 0 220 236" role="img" aria-label="AHA 17-segment bullseye shaded by typical coronary territory risk">`;
    for (const s of segments()) {
      const v = owner[s.id], p = risk[v] || 0;
      const mid = ((s.a0 + s.a1) / 2) * (Math.PI / 180), rm = (s.r0 + s.r1) / 2;
      svg += `<path d="${arc(cx, cy, s.r0, s.r1, s.a0, s.a1)}" fill="${colorOf(p)}" fill-opacity="${0.25 + 0.75 * p}" stroke="#fff" stroke-width="1.5"><title>${s.id} ${NAMES[s.id]} - ${v} ${Math.round(p * 100)}%</title></path>`;
      svg += `<text x="${cx + rm * Math.cos(mid)}" y="${cy - rm * Math.sin(mid) + 3}" text-anchor="middle" font-size="9" fill="#1d2a2c">${s.id}</text>`;
    }
    const p17 = risk[owner[17]] || 0;
    svg += `<circle cx="${cx}" cy="${cy}" r="20" fill="${colorOf(p17)}" fill-opacity="${0.25 + 0.75 * p17}" stroke="#fff" stroke-width="1.5"><title>17 apex - ${owner[17]} ${Math.round(p17 * 100)}%</title></circle>`;
    svg += `<text x="${cx}" y="${cy + 3}" text-anchor="middle" font-size="9" fill="#1d2a2c">17</text>`;
    svg += `<text x="${cx}" y="8" text-anchor="middle" font-size="8" fill="currentColor">Anterior</text>`;
    svg += `<text x="4" y="${cy + 3}" font-size="8" fill="currentColor">Septal</text>`;
    svg += `<text x="216" y="${cy + 3}" text-anchor="end" font-size="8" fill="currentColor">Lateral</text>`;
    svg += `<text x="${cx}" y="218" text-anchor="middle" font-size="8" fill="currentColor">Inferior</text>`;
    svg += `<text x="${cx}" y="232" text-anchor="middle" font-size="7.5" fill="currentColor" opacity=".75">typical territories, varies with dominance</text></svg>`;
    el.innerHTML = svg;
  }
  window.Bullseye = { render };
})();
