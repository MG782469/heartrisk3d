/* Anatomy registry: single source of truth for model output -> 3D structure.
 * Each predicted target maps to exactly one vessel group. Paths are given as surface coordinates
 * [y, theta] on the procedural heart: y = height (apex -1.3 .. base +1.0),
 * theta = angle around the long axis, 0 = anterior, +pi/2 = patient's left, -pi/2 = patient's right.
 * Coronary courses follow standard right-dominant anatomy (illustrative, not patient-specific).
 * To add a structure: add an entry here (and a model target if it is predicted). No code change needed.
 */
window.ANATOMY = {
  vessels: {
    LAD: {
      target: "LAD",
      name: "Left Anterior Descending (LAD)",
      course: "Runs down the anterior interventricular groove to the apex.",
      territory: "Anterior wall, anterior septum and apex (AHA segments 1, 2, 7, 8, 13, 14, 17)",
      origin: "LM",
      path: [[0.58, 0.72], [0.42, 0.48], [0.12, 0.36], [-0.3, 0.34], [-0.7, 0.4], [-1.02, 0.55], [-1.2, 0.85]],
      branches: [[[0.12, 0.38], [-0.12, 0.7], [-0.45, 1.0]], [[-0.3, 0.36], [-0.55, 0.7], [-0.85, 0.95]]],
      radius: 0.036,
      view: [0.8, 0.2, 6.4],
    },
    LCX: {
      target: "LCX",
      name: "Left Circumflex (LCX)",
      course: "Runs in the left atrioventricular groove around to the back of the heart.",
      territory: "Lateral wall (AHA segments 5, 6, 11, 12, 16)",
      origin: "LM",
      path: [[0.6, 0.85], [0.58, 1.2], [0.55, 1.6], [0.52, 2.0], [0.47, 2.4], [0.42, 2.7]],
      branches: [[[0.54, 1.7], [0.15, 1.8], [-0.35, 1.95]]],
      radius: 0.034,
      view: [6.2, 0.5, 0.9],
    },
    RCA: {
      target: "RCA",
      name: "Right Coronary Artery (RCA)",
      course: "Runs in the right atrioventricular groove, then down the posterior interventricular groove.",
      territory: "Inferior wall and inferior septum (AHA segments 3, 4, 9, 10, 15)",
      origin: "aorta-right",
      path: [[0.62, -0.35], [0.6, -0.8], [0.57, -1.3], [0.53, -1.8], [0.49, -2.3], [0.45, -2.8], [0.4, -3.08], [0.1, -3.14], [-0.3, -3.12], [-0.7, -3.0]],
      branches: [[[0.55, -1.55], [0.15, -1.55], [-0.35, -1.45]]],
      radius: 0.036,
      view: [-3.2, -3.6, -3.4],
    },
  },
  context: {
    LM: { name: "Left main (not predicted)", path: [[0.66, 0.45], [0.62, 0.62], [0.6, 0.78]], radius: 0.042 },
  },
  // Typical supply territories, used only for the optional myocardium shading and the bullseye inset.
  territories: {
    LAD: { thetaMin: -0.25, thetaMax: 1.15, yMax: 0.5 },
    LCX: { thetaMin: 1.15, thetaMax: 2.55, yMax: 0.55 },
    RCA: { thetaMin: 2.55, thetaMax: 3.9, yMax: 0.55 },
  },
  bullseye: {
    LAD: [1, 2, 7, 8, 13, 14, 17],
    RCA: [3, 4, 9, 10, 15],
    LCX: [5, 6, 11, 12, 16],
  },
};
