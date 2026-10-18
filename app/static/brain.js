// Brain tab: 3D view of a fly connectome (raw, or a training checkpoint) with edges coloured by synapse weight.
// Neurons are points at their real positions; the selected subset of edges is drawn as lines.
// Weight colour: diverging blue (inhibitory) ← gray (weak) → red (excitatory). "Learned change" mode: blue = weakened
// by training, red = strengthened. Uses three.js r128 (vendored in /vendor).
(() => {
  const ui = { model: null, k: 30000, select: "strongest", color: "weight", rotate: !matchMedia("(prefers-reduced-motion: reduce)").matches, hidden: new Set() };
  const cache = { neurons: new Map() };
  let view = null; // { renderer, scene, camera, controls, points, lines, raf, holder }

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmtNum = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en-US") : v.toPrecision(3));

  async function getBinary(path, params) {
    const res = await fetch(`/api/${path}?${new URLSearchParams(params)}`);
    if (!res.ok) throw new Error((await res.text()) || res.statusText);
    return { buf: await res.arrayBuffer(), meta: JSON.parse(res.headers.get("X-Meta") || "{}") };
  }

  async function neurons(graph) {
    if (!cache.neurons.has(graph)) {
      const { buf, meta } = await getBinary("brain/neurons", { graph });
      const n = meta.n;
      cache.neurons.set(graph, { meta, pos: new Float32Array(buf, 0, n * 3), group: new Uint8Array(buf, n * 12, n), role: new Uint8Array(buf, n * 13, n) });
    }
    return cache.neurons.get(graph);
  }

  function hexToRgb(hex) {
    const v = parseInt(hex.replace("#", ""), 16);
    return [(v >> 16) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
  }
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

  // ---------- page ----------
  async function viewBrain(main, ctx) {
    const { api, setTitle } = ctx;
    setTitle("Brain");
    teardown();
    const models = (await api("brain/models")).models;
    if (!models.length) return main.append(h("p", { class: "empty" }, "No connectomes found. Run scripts/build_connectome.py first."));
    if (!ui.model || !models.find((m) => m.id === ui.model)) ui.model = models[0].id;

    const select = h("select", { class: "select", "aria-label": "Brain to show", onchange: (e) => { ui.model = e.target.value; refresh(); } });
    const groups = {};
    for (const m of models) (groups[m.group] ||= []).push(m);
    for (const [g, ms] of Object.entries(groups)) {
      select.append(h("optgroup", { label: g }, ms.map((m) => h("option", { value: m.id, selected: m.id === ui.model }, m.time ? `${m.label} · ${m.time}` : m.label))));
    }
    const seg = (label, key, opts) => h("div", { class: "segmented", role: "group", "aria-label": label },
      ...opts.map(([v, l]) => h("button", { "aria-pressed": String(ui[key] === v), "data-key": key, "data-v": String(v),
        onclick: () => { ui[key] = v; syncSeg(); refresh(key === "color" ? "recolor" : undefined); } }, l)));
    const controls = h("div", { class: "filter-row" }, select,
      seg("Number of connections", "k", [[10000, "10k"], [30000, "30k"], [100000, "100k"]]),
      seg("Which connections", "select", [["strongest", "Strongest"], ["random", "Random sample"], ["changed", "Most changed"]]),
      seg("Colour by", "color", [["weight", "Weight"], ["change", "Learned change"]]));
    const stage = h("div", { class: "brain-stage" });
    const stats = h("div", { class: "brain-stats", "aria-live": "polite" });
    const legend = h("div", { class: "brain-legend" });
    const groupsRow = h("div", { class: "run-legend brain-groups" });
    const tools = h("div", { class: "brain-tools" },
      h("button", { class: "run-chip", onclick: () => resetCamera() }, "Reset view"),
      h("button", { class: "run-chip", "aria-pressed": String(ui.rotate), id: "rotate-btn",
        onclick: (e) => { ui.rotate = !ui.rotate; e.currentTarget.setAttribute("aria-pressed", String(ui.rotate)); if (view) view.controls.autoRotate = ui.rotate; } }, "Auto-rotate"));
    stage.append(stats, legend, tools);
    main.append(controls, h("div", { class: "brain-card" }, stage), groupsRow,
      h("p", { class: "summary" }, "Drag to rotate, scroll or pinch to zoom, right-drag to pan. Points are neurons at their positions in the brain; ",
        "olfactory receptor neurons (where syllables enter) and descending neurons (where replies are read out) are highlighted. ",
        "Lines are connections, coloured by the signed weight the model uses: synapse count normalised by the neuron’s total input, times the learned scale for checkpoints."));

    function syncSeg() {
      controls.querySelectorAll("button[data-key]").forEach((b) => b.setAttribute("aria-pressed", String(String(ui[b.dataset.key]) === b.dataset.v)));
    }

    let edgeData = null, current = null;
    async function refresh(mode) {
      const model = models.find((m) => m.id === ui.model);
      stage.classList.add("refreshing");
      try {
        const nd = await neurons(model.graph);
        if (mode !== "recolor" || !edgeData) {
          const { buf, meta } = await getBinary("brain/edges", { model: model.id, k: ui.k, select: ui.select });
          const k = meta.shown;
          edgeData = { meta, pre: new Uint32Array(buf, 0, k), post: new Uint32Array(buf, k * 4, k), w: new Float32Array(buf, k * 8, k), scale: new Float32Array(buf, k * 12, k) };
        }
        if (!view || current !== model.graph) { buildScene(stage, nd); current = model.graph; }
        drawNeurons(nd);
        drawEdges(nd, edgeData);
        renderGroups(nd);
        renderStats(model, nd, edgeData.meta);
        renderLegend(edgeData.meta);
      } catch (err) {
        stats.replaceChildren(h("strong", {}, "Couldn’t load this brain: "), err.message);
      }
      stage.classList.remove("refreshing");
    }

    function renderGroups(nd) {
      const counts = new Array(nd.meta.groups.length + 1).fill(0);
      nd.group.forEach((g) => counts[g]++);
      groupsRow.replaceChildren(h("span", { class: "chip-state", style: "align-self:center" }, "Neurons:"),
        ...nd.meta.groups.map((name, i) => counts[i + 1] ? h("button", { class: "run-chip", "aria-pressed": String(!ui.hidden.has(i + 1)),
          onclick: () => { ui.hidden.has(i + 1) ? ui.hidden.delete(i + 1) : ui.hidden.add(i + 1); drawNeurons(nd); drawEdges(nd, edgeData); renderGroups(nd); } },
          name, h("span", { class: "chip-state" }, counts[i + 1].toLocaleString("en-US"))) : null));
    }

    function renderStats(model, nd, meta) {
      const rows = [
        ["Neurons", nd.meta.n.toLocaleString("en-US") + (nd.meta.estimated ? ` (${nd.meta.estimated.toLocaleString("en-US")} placed by their connections)` : "")],
        ["Connections", `${meta.shown.toLocaleString("en-US")} shown of ${meta.total_edges.toLocaleString("en-US")}`],
        ["Inhibitory", `${(meta.inhibitory_frac * 100).toFixed(0)}% of connections`],
        ["|weight|", `median ${fmtNum(meta.w_abs_p50)} · 99th pct ${fmtNum(meta.w_abs_p99)} · max ${fmtNum(meta.w_abs_max)}`],
      ];
      if (meta.learned) rows.push(["Learned scale", `${fmtNum(meta.scale_min)} – ${fmtNum(meta.scale_max)}× the synapse-count weight`]);
      if (meta.step) rows.push(["Training step", meta.step.toLocaleString("en-US")]);
      stats.replaceChildren(h("div", { class: "brain-title" }, model.group.startsWith("Connectomes") ? model.label : `${model.group} — ${model.label}`),
        ...rows.map(([k, v]) => h("div", {}, h("span", { class: "tip-label" }, k + " "), h("strong", {}, v))));
    }

    function renderLegend(meta) {
      const change = ui.color === "change";
      const left = change ? "weakened" : "inhibitory", right = change ? "strengthened" : "excitatory";
      const neg = css("--div-neg"), mid = css("--div-mid"), pos = css("--div-pos");
      legend.replaceChildren(
        h("div", { class: "legend-bar", style: `background: linear-gradient(90deg, ${neg}, ${mid}, ${pos})`, "aria-hidden": "true" }),
        h("div", { class: "legend-labels" }, h("span", {}, `← ${left}`), h("span", {}, change ? "unchanged" : "weak"), h("span", {}, `${right} →`)),
        h("div", { class: "legend-keys" },
          h("span", {}, h("span", { class: "dot", style: `background:${css("--role-in")}` }), "input (olfactory receptors)"),
          h("span", {}, h("span", { class: "dot", style: `background:${css("--role-out")}` }), "output (descending)")),
        ...(change && !meta.learned ? [h("div", { class: "tip-label" }, "This model has no learned changes; pick a checkpoint.")] : []));
    }

    // ---------- three.js ----------
    function buildScene(stageEl, nd) {
      teardown();
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.domElement.setAttribute("role", "img");
      renderer.domElement.setAttribute("aria-label", "3D view of the fly brain connectome");
      stageEl.prepend(renderer.domElement);
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.autoRotate = ui.rotate;
      controls.autoRotateSpeed = 0.6;
      view = { renderer, scene, camera, controls, stage: stageEl, points: null, lines: null };
      resetCamera();
      const resize = () => {
        const w = stageEl.clientWidth, hgt = stageEl.clientHeight;
        renderer.setSize(w, hgt, false);
        camera.aspect = w / Math.max(1, hgt);
        camera.updateProjectionMatrix();
      };
      view.ro = new ResizeObserver(resize);
      view.ro.observe(stageEl);
      resize();
      const loop = () => {
        if (!stageEl.isConnected) return teardown();
        controls.update();
        renderer.render(scene, camera);
        view.raf = requestAnimationFrame(loop);
      };
      loop();
    }

    function drawNeurons(nd) {
      for (const o of [view.points, view.roles]) if (o) { view.scene.remove(o); o.geometry.dispose(); o.material.dispose(); }
      const base = hexToRgb(css("--neuron")), cin = hexToRgb(css("--role-in")), cout = hexToRgb(css("--role-out"));
      const n = nd.meta.n;
      const layer = (want, size, opacity) => {
        const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
        let m = 0;
        for (let i = 0; i < n; i++) {
          if (!want(i)) continue;
          pos.set(nd.pos.subarray(i * 3, i * 3 + 3), m * 3);
          col.set(nd.role[i] === 1 ? cin : nd.role[i] === 2 ? cout : base, m * 3);
          m++;
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(pos.subarray(0, m * 3), 3));
        geo.setAttribute("color", new THREE.BufferAttribute(col.subarray(0, m * 3), 3));
        const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size, vertexColors: true, transparent: true, opacity, depthWrite: false }));
        view.scene.add(pts);
        return pts;
      };
      view.points = layer((i) => !nd.role[i] && !ui.hidden.has(nd.group[i]), 0.01, 0.45);
      view.roles = layer((i) => nd.role[i] > 0, 0.035, 0.95);
    }

    function drawEdges(nd, e) {
      if (view.lines) { view.scene.remove(view.lines); view.lines.geometry.dispose(); view.lines.material.dispose(); }
      const neg = hexToRgb(css("--div-neg")), mid = hexToRgb(css("--div-mid")), posc = hexToRgb(css("--div-pos"));
      const k = e.pre.length, P = new Float32Array(k * 6), C = new Float32Array(k * 6);
      const change = ui.color === "change";
      const ref = change ? Math.max(1e-6, ...[Math.log(e.meta.scale_max || 1), -Math.log(e.meta.scale_min || 1)].map(Math.abs)) : e.meta.w_abs_p99 || 1;
      let m = 0;
      for (let i = 0; i < k; i++) {
        const a = e.pre[i], b = e.post[i];
        if ((ui.hidden.has(nd.group[a]) && !nd.role[a]) || (ui.hidden.has(nd.group[b]) && !nd.role[b])) continue;
        const v = change ? Math.log(e.scale[i]) : e.w[i];
        const t = Math.sqrt(Math.min(1, Math.abs(v) / ref));
        const c = mix(mid, v < 0 ? neg : posc, t);
        P.set(nd.pos.subarray(a * 3, a * 3 + 3), m * 6); P.set(nd.pos.subarray(b * 3, b * 3 + 3), m * 6 + 3);
        C.set(c, m * 6); C.set(c, m * 6 + 3);
        m++;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(P.subarray(0, m * 6), 3));
      geo.setAttribute("color", new THREE.BufferAttribute(C.subarray(0, m * 6), 3));
      const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: Math.max(0.15, Math.min(0.6, 8000 / Math.max(1, m))),
        depthWrite: false });
      view.lines = new THREE.LineSegments(geo, mat);
      view.scene.add(view.lines);
    }

    await refresh();
  }

  function resetCamera() {
    if (!view) return;
    view.camera.position.set(0, 0.15, 3.2);
    view.controls.target.set(0, 0, 0);
    view.controls.update();
  }

  function teardown() {
    if (!view) return;
    cancelAnimationFrame(view.raf);
    view.ro?.disconnect();
    view.controls.dispose();
    for (const o of [view.points, view.roles, view.lines]) if (o) { o.geometry.dispose(); o.material.dispose(); }
    view.renderer.dispose();
    view.renderer.domElement.remove();
    view = null;
  }

  window.MougiaBrain = { viewBrain };
})();
