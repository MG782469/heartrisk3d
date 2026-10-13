/* HeartRisk 3D - procedural heart with coronary arteries (Three.js r128).
 * No third-party mesh: geometry is generated in code (no licence issues, light enough without a GPU).
 * Colour of each vessel = predicted stenosis probability (vessel-level only; the dataset has no
 * lesion position, so the whole vessel is coloured, never a single segment).
 */
(function () {
  "use strict";
  const T = window.THREE;

  // colour-blind-friendly ramp: teal (low) -> amber (mid) -> crimson (high)
  const RAMP = [new T.Color("#159e9a"), new T.Color("#f0a202"), new T.Color("#c8102e")];
  function riskColor(p) {
    const c = new T.Color();
    if (p <= 0.5) return c.copy(RAMP[0]).lerp(RAMP[1], p / 0.5);
    return c.copy(RAMP[1]).lerp(RAMP[2], (p - 0.5) / 0.5);
  }
  const MUSCLE = new T.Color("#8a2f2c");

  // ---- heart surface shape -------------------------------------------------
  // profile: radius as a function of height y (apex at -1.3, base at +1.0)
  const PROFILE = [[0, -1.3], [0.22, -1.2], [0.47, -0.95], [0.7, -0.55], [0.84, -0.15], [0.9, 0.2], [0.88, 0.45], [0.78, 0.7], [0.55, 0.92], [0.0, 1.02]];
  function profileR(y) {
    for (let i = 0; i < PROFILE.length - 1; i++) {
      const [r0, y0] = PROFILE[i], [r1, y1] = PROFILE[i + 1];
      if (y >= y0 && y <= y1) return r0 + (r1 - r0) * ((y - y0) / (y1 - y0));
    }
    return 0;
  }
  const apexShift = (y) => (y < 0.2 ? 0.32 * Math.pow((0.2 - y) / 1.5, 1.2) : 0); // apex points to patient's left (+x)
  function wrap(a) {
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    return a;
  }
  // radius multiplier for grooves (interventricular + atrioventricular) and a flatter posterior wall
  function grooveFactor(y, th) {
    let f = 1;
    const ant = wrap(th - 0.36), post = wrap(th - Math.PI + 0.05);
    if (y < 0.5) f -= 0.045 * Math.exp(-Math.pow(ant / 0.14, 2)) + 0.035 * Math.exp(-Math.pow(post / 0.16, 2));
    f -= 0.05 * Math.exp(-Math.pow((y - 0.58) / 0.07, 2));
    if (Math.cos(th) < 0) f -= 0.08 * Math.pow(-Math.cos(th), 2) * (y < 0.4 ? 1 : 0.5);
    return f;
  }
  function surfacePoint(y, th, lift) {
    const r = profileR(y) * grooveFactor(y, th) + (lift || 0);
    return new T.Vector3(apexShift(y) + r * Math.sin(th), y, r * Math.cos(th));
  }

  function buildHeartGeometry(segY, segT) {
    const pos = [], uv = [], idx = [];
    for (let i = 0; i <= segY; i++) {
      const y = -1.3 + (2.32 * i) / segY;
      for (let j = 0; j <= segT; j++) {
        const th = -Math.PI + (2 * Math.PI * j) / segT;
        const p = surfacePoint(y, th, 0);
        pos.push(p.x, p.y, p.z);
        uv.push(j / segT, i / segY);
      }
    }
    for (let i = 0; i < segY; i++)
      for (let j = 0; j < segT; j++) {
        const a = i * (segT + 1) + j, b = a + segT + 1;
        idx.push(a, b, a + 1, b, b + 1, a + 1);
      }
    const g = new T.BufferGeometry();
    g.setAttribute("position", new T.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new T.Float32BufferAttribute(uv, 2));
    g.setAttribute("color", new T.Float32BufferAttribute(new Array(pos.length).fill(1), 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  }

  function tube(points, radius, material, segs) {
    const curve = new T.CatmullRomCurve3(points, false, "catmullrom", 0.5);
    const geo = new T.TubeGeometry(curve, segs || Math.max(24, points.length * 12), radius, 10, false);
    const m = new T.Mesh(geo, material);
    m.userData.curve = curve;
    return m;
  }
  const pathPoints = (path, lift) => path.map(([y, th]) => surfacePoint(y, th, lift));

  // ---- scene ------------------------------------------------------------------
  function HeartView(container, anatomy, opts) {
    this.container = container;
    this.anatomy = anatomy;
    this.opts = Object.assign({ lowGraphics: false, territories: true, flow: true, onSelect: () => {} }, opts || {});
    this.risk = { LAD: 0, LCX: 0, RCA: 0 };
    this.selected = null;
    this.vessels = {};
    this.particles = [];
    this._init();
  }

  HeartView.prototype._init = function () {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.renderer = new T.WebGLRenderer({ antialias: !this.opts.lowGraphics, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(this.opts.lowGraphics ? 1 : Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h);
    this.renderer.outputEncoding = T.sRGBEncoding;
    this.renderer.toneMapping = T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.container.appendChild(this.renderer.domElement);

    this.scene = new T.Scene();
    this.camera = new T.PerspectiveCamera(38, w / h, 0.1, 100);
    this.camera.position.set(0.8, 0.2, 6.4);
    this.controls = new T.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 2.2;
    this.controls.maxDistance = 11;
    this.controls.target.set(0.1, 0.05, 0);

    this.scene.add(new T.HemisphereLight(0xfff1ea, 0x1a1012, 0.7));
    const key = new T.DirectionalLight(0xfff3e8, 1.0);
    key.position.set(3, 4, 5);
    this.scene.add(key);
    const fill = new T.DirectionalLight(0xffd9cc, 0.35);
    fill.position.set(-4, -1, 4);
    this.scene.add(fill);
    const rim = new T.DirectionalLight(0x8fb4ff, 0.6);
    rim.position.set(-3, 2, -5);
    this.scene.add(rim);

    this.root = new T.Group();
    this.root.rotation.set(0.12, -0.18, -0.38); // anatomical tilt: apex down, left and forward
    this.scene.add(this.root);

    // myocardium
    const segY = this.opts.lowGraphics ? 48 : 96, segT = this.opts.lowGraphics ? 64 : 128;
    this.heartGeo = buildHeartGeometry(segY, segT);
    // wet-tissue look: clearcoat over a slightly rough muscle
    this.heartMat = new T.MeshPhysicalMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.62, metalness: 0.0, clearcoat: 0.35, clearcoatRoughness: 0.35 });
    this.heart = new T.Mesh(this.heartGeo, this.heartMat);
    this.root.add(this.heart);
    this._paintTerritories();

    // great vessels and atria (context, not predicted)
    const aortaMat = new T.MeshPhysicalMaterial({ color: 0xc0574e, roughness: 0.45, clearcoat: 0.3 });
    const pulmMat = new T.MeshPhysicalMaterial({ color: 0x5a6aa0, roughness: 0.5, clearcoat: 0.25 });
    const atriumMat = new T.MeshPhysicalMaterial({ color: 0x7e2e2e, roughness: 0.6, clearcoat: 0.25 });
    const V = (x, y, z) => new T.Vector3(x, y, z);
    this.root.add(tube([V(-0.05, 0.85, 0.05), V(-0.08, 1.28, 0.05), V(0.05, 1.6, -0.12), V(0.42, 1.66, -0.4), V(0.66, 1.42, -0.55), V(0.72, 0.85, -0.66)], 0.13, aortaMat, 80));
    this.root.add(tube([V(0.3, 0.85, 0.42), V(0.32, 1.25, 0.35), V(0.2, 1.45, 0.05), V(-0.2, 1.48, -0.15)], 0.12, pulmMat, 60));
    this.root.add(tube([V(0.32, 1.2, 0.35), V(0.6, 1.38, 0.05), V(0.88, 1.38, -0.2)], 0.09, pulmMat, 40));
    this.root.add(tube([V(-0.55, 0.75, 0.0), V(-0.55, 1.2, -0.05), V(-0.5, 1.55, -0.05)], 0.11, pulmMat, 30));
    const ra = new T.Mesh(new T.SphereGeometry(0.35, 32, 24), atriumMat);
    ra.position.set(-0.55, 0.78, 0.05);
    ra.scale.set(1, 0.85, 1.1);
    this.root.add(ra);
    const la = new T.Mesh(new T.SphereGeometry(0.33, 32, 24), atriumMat);
    la.position.set(0.35, 0.85, -0.45);
    la.scale.set(1.2, 0.8, 0.9);
    this.root.add(la);

    // coronary arteries
    const lift = 0.025;
    for (const [key2, v] of Object.entries(this.anatomy.vessels)) {
      const mat = new T.MeshStandardMaterial({ color: riskColor(0), roughness: 0.35, metalness: 0.1, emissive: 0x000000 });
      const group = new T.Group();
      group.userData.target = v.target;
      const main = tube(pathPoints(v.path, lift), v.radius, mat);
      main.userData.target = v.target;
      group.add(main);
      (v.branches || []).forEach((b) => {
        const m = tube(pathPoints(b, lift), v.radius * 0.7, mat);
        m.userData.target = v.target;
        group.add(m);
      });
      // origin connector from the aortic root
      const o = v.origin === "LM" ? surfacePoint(0.6, 0.78, lift) : null;
      if (v.origin === "aorta-right") {
        const c = tube([V(-0.12, 0.9, 0.2), surfacePoint(0.66, -0.15, lift), surfacePoint(...v.path[0], lift)], v.radius, mat, 20);
        c.userData.target = v.target;
        group.add(c);
      }
      if (o) {
        const c = tube([o, surfacePoint(...v.path[0], lift)], v.radius, mat, 8);
        c.userData.target = v.target;
        group.add(c);
      }
      this.root.add(group);
      this.vessels[key2] = { group, mat, curve: main.userData.curve, def: v };
    }
    const lm = this.anatomy.context.LM;
    const lmMat = new T.MeshStandardMaterial({ color: 0x9a9a9a, roughness: 0.4 });
    this.root.add(tube([V(0.08, 0.92, 0.22), ...pathPoints(lm.path, lift)], lm.radius, lmMat, 20));

    // flow particles (speed slows as stenosis probability rises)
    const pg = new T.SphereGeometry(0.018, 8, 6);
    const pm = new T.MeshBasicMaterial({ color: 0xfff6e0 });
    for (const [k, v] of Object.entries(this.vessels)) {
      for (let i = 0; i < 9; i++) {
        const s = new T.Mesh(pg, pm);
        s.userData = { vessel: k, t: i / 9 };
        this.root.add(s);
        this.particles.push(s);
      }
    }

    // labels (HTML overlay)
    this.labelLayer = document.createElement("div");
    this.labelLayer.className = "vlabels";
    this.container.appendChild(this.labelLayer);
    this.labels = {};
    for (const k of Object.keys(this.vessels)) {
      const el = document.createElement("button");
      el.className = "vlabel";
      el.type = "button";
      el.textContent = k;
      el.addEventListener("click", () => this.select(k));
      this.labelLayer.appendChild(el);
      this.labels[k] = el;
    }

    // picking
    this.ray = new T.Raycaster();
    this.mouse = new T.Vector2();
    this.renderer.domElement.addEventListener("pointerdown", (e) => (this._down = [e.clientX, e.clientY]));
    this.renderer.domElement.addEventListener("pointerup", (e) => {
      if (!this._down || Math.hypot(e.clientX - this._down[0], e.clientY - this._down[1]) > 5) return;
      const hit = this._pick(e);
      if (hit) this.select(hit);
    });
    this.renderer.domElement.addEventListener("pointermove", (e) => {
      this.renderer.domElement.style.cursor = this._pick(e) ? "pointer" : "grab";
    });

    window.addEventListener("resize", () => this.resize());
    this.clock = new T.Clock();
    this._camGoal = null;
    const loop = () => {
      requestAnimationFrame(loop);
      this._frame();
    };
    loop();
  };

  HeartView.prototype._pick = function (e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(this.mouse, this.camera);
    const objs = Object.values(this.vessels).map((v) => v.group);
    const hits = this.ray.intersectObjects(objs, true);
    return hits.length ? hits[0].object.userData.target : null;
  };

  HeartView.prototype._paintTerritories = function () {
    const pos = this.heartGeo.attributes.position, col = this.heartGeo.attributes.color;
    const terr = this.anatomy.territories;
    const c = new T.Color();
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i), x = pos.getX(i) - apexShift(y), z = pos.getZ(i);
      let th = Math.atan2(x, z);
      // low-frequency mottling so the muscle does not look like plastic
      const n = 0.06 * Math.sin(7.1 * y + 3.3 * th) * Math.cos(5.3 * th - 2.1 * y) + 0.03 * Math.sin(17 * y + 11 * th);
      c.copy(MUSCLE).offsetHSL(0, 0, n);
      if (this.opts.territories) {
        for (const [k, t] of Object.entries(terr)) {
          let a = th;
          if (t.thetaMax > Math.PI && a < 0) a += 2 * Math.PI;
          const inside = a >= t.thetaMin && a < t.thetaMax && y <= t.yMax;
          if (k === "LAD" && y < -0.95) {
            c.lerp(riskColor(this.risk.LAD), 0.42 * this.risk.LAD);
            break;
          }
          if (inside) {
            const fade = Math.min(1, (t.yMax - y) / 0.25);
            c.lerp(riskColor(this.risk[k]), 0.42 * this.risk[k] * fade);
            break;
          }
        }
      }
      col.setXYZ(i, c.r, c.g, c.b);
    }
    col.needsUpdate = true;
  };

  HeartView.prototype.setRisk = function (risk) {
    Object.assign(this.risk, risk);
    for (const [k, v] of Object.entries(this.vessels)) {
      const p = this.risk[k] || 0;
      v.mat.color.copy(riskColor(p));
      v.mat.emissive.copy(riskColor(p)).multiplyScalar(this.selected === k ? 0.55 : 0.18 + 0.25 * p);
      this.labels[k].textContent = `${k} ${Math.round(p * 100)}%`;
      this.labels[k].style.borderColor = "#" + riskColor(p).getHexString();
    }
    this._paintTerritories();
  };

  HeartView.prototype.setOptions = function (o) {
    Object.assign(this.opts, o);
    this.particles.forEach((s) => (s.visible = this.opts.flow));
    this._paintTerritories();
  };

  HeartView.prototype.select = function (k) {
    this.selected = k;
    Object.values(this.labels).forEach((el) => el.classList.remove("on"));
    if (k && this.labels[k]) this.labels[k].classList.add("on");
    this.setRisk({});
    if (k) this.view(k);
    this.opts.onSelect(k);
  };

  HeartView.prototype.view = function (name) {
    const presets = { anterior: [0.8, 0.2, 6.4], lateral: [6.2, 0.5, 0.9], inferior: [-3.2, -3.6, -3.4] };
    const v = presets[name] || (this.vessels[name] && this.vessels[name].def.view);
    if (v) this._camGoal = new T.Vector3(...v);
  };

  HeartView.prototype.resize = function () {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  HeartView.prototype._frame = function () {
    const dt = Math.min(this.clock.getDelta(), 0.05), t = this.clock.elapsedTime;
    // heartbeat: double bump (systole) every ~0.9 s
    const ph = (t % 0.9) / 0.9;
    const beat = 1 + 0.022 * Math.exp(-Math.pow((ph - 0.12) / 0.05, 2)) + 0.012 * Math.exp(-Math.pow((ph - 0.32) / 0.06, 2));
    this.root.scale.setScalar(beat);
    // flow particles
    for (const s of this.particles) {
      if (!s.visible) continue;
      const v = this.vessels[s.userData.vessel];
      const p = this.risk[s.userData.vessel] || 0;
      s.userData.t = (s.userData.t + dt * (0.32 - 0.26 * p)) % 1;
      s.position.copy(v.curve.getPointAt(s.userData.t));
      s.scale.setScalar(1 + 0.5 * p);
    }
    if (this._camGoal) {
      // orbit around the target (never through the heart): interpolate direction and distance separately
      const tgt = this.controls.target;
      const cur = this.camera.position.clone().sub(tgt), goal = this._camGoal.clone().sub(tgt);
      const k = Math.min(1, dt * 5);
      const r = cur.length() + (goal.length() - cur.length()) * k;
      const dir = cur.clone().normalize().lerp(goal.clone().normalize(), k);
      if (dir.lengthSq() < 1e-4) dir.set(0, 1, 0);
      this.camera.position.copy(tgt).add(dir.normalize().multiplyScalar(r));
      if (this.camera.position.distanceTo(this._camGoal) < 0.03) this._camGoal = null;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    // label positions: midpoint of each vessel curve
    const r = this.renderer.domElement.getBoundingClientRect();
    for (const [k, v] of Object.entries(this.vessels)) {
      const p = v.curve.getPointAt(0.45).clone();
      this.root.localToWorld(p);
      p.project(this.camera);
      const el = this.labels[k];
      el.style.display = p.z < 1 ? "block" : "none";
      el.style.transform = `translate(${((p.x + 1) / 2) * r.width}px, ${((1 - p.y) / 2) * r.height}px) translate(-50%, -140%)`;
    }
  };

  // square, fixed-size image for reports (independent of the current panel size)
  HeartView.prototype.snapshot = function () {
    const size = new T.Vector2();
    this.renderer.getSize(size);
    const aspect = this.camera.aspect;
    this.renderer.setSize(900, 900, false);
    this.camera.aspect = 1;
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
    const url = this.renderer.domElement.toDataURL("image/png");
    this.renderer.setSize(size.x, size.y, false);
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
    return url;
  };

  window.HeartView = HeartView;
  window.riskColorHex = (p) => "#" + riskColor(p).getHexString();
})();
