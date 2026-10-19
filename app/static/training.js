// Training tab: live convergence charts + periodic reply samples, read from runs/<id>/ (see mougia_train/runlog.py).
// Charts are hand-built SVG: 2px lines, hairline grid, crosshair tooltip listing every run, shared legend, direct end labels.
(() => {
  const SVGNS = "http://www.w3.org/2000/svg";
  const KNOWN = [
    // [source, key, title, format]
    ["metrics", "train_loss", "Training loss", (v) => v.toFixed(3)],
    ["eval", "val_loss", "Validation loss", (v) => v.toFixed(3)],
    ["eval", "val_ppl", "Validation perplexity", (v) => compact(v)],
    ["eval", "context_gain_duel", "Context gain: duels (nats; >0 = answers the call)", (v) => v.toFixed(2)],
    ["eval", "context_gain_cont", "Context gain: continuations (nats)", (v) => v.toFixed(2)],
    ["eval", "meter_rate", "Replies with 15-syllable meter", pct],
    ["eval", "rhyme_rate", "Replies that rhyme", pct],
    ["eval", "cy_rate", "Replies that read as Cypriot", pct],
    ["metrics", "lr", "Learning rate", function lrFmt(v) { return v.toExponential(1); }],
    ["metrics", "grad_norm", "Gradient norm", (v) => v.toFixed(2)],
    ["metrics", "tokens_per_s", "Throughput (syllables/s)", (v) => compact(v)],
    ["metrics", "gpu_mem_gb", "GPU memory (GB)", (v) => v.toFixed(1)],
  ];
  const SKIP = new Set(["step", "elapsed_s", "time", "phase"]);
  const ui = { x: "hours", hidden: new Set(), timer: null, data: null };

  function pct(v) { return `${(v * 100).toFixed(0)}%`; }
  function compact(v) {
    const a = Math.abs(v);
    return a >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : a >= 1e3 ? `${(v / 1e3).toFixed(1)}K` : a >= 10 ? v.toFixed(0) : a >= 0.1 || a === 0 ? v.toFixed(2) : v.toPrecision(3);
  }
  function fmtHours(s) {
    if (s == null) return "–";
    const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60);
    return h ? `${h} h ${m} min` : `${m} min`;
  }
  const svg = (tag, attrs = {}) => {
    const el = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  };

  function niceTicks(lo, hi, n = 4) {
    if (lo === hi) { lo -= 1; hi += 1; }
    const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const ticks = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(+t.toPrecision(12));
    return ticks;
  }
  // Tick labels carry just enough decimals for the tick step, so neighbouring ticks never print the same text.
  function tickFormatter(ticks, fmt) {
    if (fmt === pct || fmt.name === "lrFmt") return fmt;
    const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
    const dec = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
    return (v) => (Math.abs(v) >= 1e4 ? compact(v) : v.toFixed(dec));
  }

  // ---------- one line chart ----------
  function lineChart({ title, series, fmt, xFmt }) {
    const card = h("figure", { class: "chart" }, h("figcaption", {}, title));
    const box = h("div", { class: "chart-box", tabindex: "0", role: "img", "aria-label": `${title} chart; use arrow keys to read values` });
    const tip = h("div", { class: "chart-tip", hidden: true });
    card.append(box, tip);
    const draw = () => {
      box.replaceChildren();
      const W = Math.max(280, box.clientWidth || 440), H = 200, m = { l: 52, r: 16, t: 10, b: 26 };
      const all = series.flatMap((s) => s.points);
      if (!all.length) return;
      let [x0, x1] = [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0]))];
      let [y0, y1] = [Math.min(...all.map((p) => p[1])), Math.max(...all.map((p) => p[1]))];
      if (x0 === x1) x1 = x0 + 1;
      const pad = (y1 - y0 || Math.abs(y1) || 1) * 0.06;
      const yt = niceTicks(y0 - pad, y1 + pad);
      y0 = Math.min(yt[0], y0 - pad); y1 = Math.max(yt[yt.length - 1], y1 + pad);
      const tfmt = tickFormatter(yt, fmt);
      const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
      const Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
      const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "chart-svg" });
      for (const t of yt) {
        if (t < y0 || t > y1) continue;
        root.append(svg("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" }));
        const lab = svg("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", class: "tick" }); lab.textContent = tfmt(t); root.append(lab);
      }
      for (const t of niceTicks(x0, x1, 4)) {
        if (t < x0 || t > x1) continue;
        const lab = svg("text", { x: X(t), y: H - 6, "text-anchor": "middle", class: "tick" }); lab.textContent = xFmt(t); root.append(lab);
      }
      root.append(svg("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }));
      for (const s of series) {
        if (!s.points.length) continue;
        const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
        root.append(svg("path", { d, class: "series-line", style: `stroke: var(--s${(s.slot % 8) + 1})` }));
        const last = s.points[s.points.length - 1];
        root.append(svg("circle", { cx: X(last[0]), cy: Y(last[1]), r: 4, class: "end-dot", style: `fill: var(--s${(s.slot % 8) + 1})` }));
      }
      // Crosshair + tooltip listing every run at the nearest x.
      const hair = svg("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" });
      const hit = svg("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" });
      root.append(hair, hit);
      const xs = [...new Set(all.map((p) => p[0]))].sort((a, b) => a - b);
      let cur = -1;
      const show = (i) => {
        cur = Math.max(0, Math.min(xs.length - 1, i));
        const xv = xs[cur];
        hair.setAttribute("x1", X(xv)); hair.setAttribute("x2", X(xv)); hair.setAttribute("visibility", "visible");
        tip.replaceChildren(h("div", { class: "tip-x" }, xFmt(xv)));
        for (const s of series) {
          const p = nearest(s.points, xv);
          if (!p) continue;
          tip.append(h("div", { class: "tip-row" }, h("span", { class: "key", style: `background: var(--s${(s.slot % 8) + 1})` }),
            h("strong", {}, fmt(p[1])), h("span", { class: "tip-label" }, s.label)));
        }
        tip.hidden = false;
        const left = X(xv) + 14, w = tip.offsetWidth;
        tip.style.left = `${left + w > W ? X(xv) - w - 14 : left}px`;
      };
      const hide = () => { hair.setAttribute("visibility", "hidden"); tip.hidden = true; };
      hit.addEventListener("pointermove", (e) => {
        const r = root.getBoundingClientRect(), xv = x0 + ((e.clientX - r.left) * (W / r.width) - m.l) / (W - m.l - m.r) * (x1 - x0);
        let lo = 0, hi = xs.length - 1;
        while (lo < hi) { const mid = (lo + hi) >> 1; xs[mid] < xv ? (lo = mid + 1) : (hi = mid); }
        show(lo > 0 && Math.abs(xs[lo - 1] - xv) < Math.abs(xs[lo] - xv) ? lo - 1 : lo);
      });
      hit.addEventListener("pointerleave", hide);
      box.onkeydown = (e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); show((cur < 0 ? xs.length : cur) + (e.key === "ArrowRight" ? 1 : -1)); }
        if (e.key === "Escape") hide();
      };
      box.onblur = hide;
      box.append(root);
    };
    card._draw = draw;
    return card;
  }

  function nearest(points, x) {
    if (!points.length) return null;
    let lo = 0, hi = points.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; points[mid][0] < x ? (lo = mid + 1) : (hi = mid); }
    if (lo > 0 && Math.abs(points[lo - 1][0] - x) < Math.abs(points[lo][0] - x)) lo--;
    return points[lo];
  }

  // ---------- the view ----------
  async function viewTraining(main, ctx) {
    const { api, setTitle, verseLines, fmt } = ctx;
    setTitle("Training");
    const holder = h("div", { class: "training" });
    main.append(holder);
    let first = true;

    const load = async () => {
      if (!holder.isConnected) return stop();
      holder.classList.add("refreshing");
      const runs = (await api("runs")).runs;
      const series = await Promise.all(runs.map((r) => api("run_series", { run: r.run })));
      const samples = await api("run_samples", { size: 12 });
      ui.data = { runs, series, samples };
      holder.classList.remove("refreshing");
      render(holder, ctx);
      if (first) { first = false; window.addEventListener("resize", redrawCharts); }
    };
    const stop = () => { clearInterval(ui.timer); ui.timer = null; window.removeEventListener("resize", redrawCharts); };
    clearInterval(ui.timer);
    ui.timer = setInterval(() => load().catch(() => {}), 30000);
    await load();
  }

  let charts = [];
  function redrawCharts() { charts.forEach((c) => c._draw()); }

  function render(holder, { verseLines, fmt }) {
    const { runs, series, samples } = ui.data;
    holder.replaceChildren();
    if (!runs.length) {
      holder.append(h("div", { class: "empty" },
        h("p", {}, "No training runs yet."),
        h("p", { class: "summary", style: "margin: 8px auto; text-align: center" },
          "Runs appear here as soon as training starts writing to runs/<run>/. Charts refresh every 30 seconds, and every 15 minutes of training each model is asked to reply to a fixed set of couplets.")));
      return;
    }
    const xFmt = ui.x === "hours" ? (v) => `${v.toFixed(v < 10 ? 1 : 0)} h` : (v) => compact(v);
    const xOf = (r) => (ui.x === "hours" ? r.elapsed_s / 3600 : r.step);

    // Filter row: x-axis choice + run legend/toggles (the legend for every chart below).
    const seg = h("div", { class: "segmented", role: "group", "aria-label": "Horizontal axis" },
      ...[["hours", "Training time"], ["step", "Step"]].map(([k, l]) =>
        h("button", { "aria-pressed": String(ui.x === k), onclick: () => { ui.x = k; render(holder, { verseLines, fmt }); } }, l)));
    const legend = h("div", { class: "run-legend" }, runs.map((r) => {
      const on = !ui.hidden.has(r.run);
      return h("button", { class: "run-chip", "aria-pressed": String(on), title: r.run,
        onclick: () => { on ? ui.hidden.add(r.run) : ui.hidden.delete(r.run); render(holder, { verseLines, fmt }); } },
        h("span", { class: "key", style: `background: var(--s${(r.slot % 8) + 1})` }), r.label,
        h("span", { class: "chip-state" }, r.status?.state || "–"));
    }));
    holder.append(h("div", { class: "filter-row" }, seg, legend));

    // Charts: known metrics first, then any other numeric field the trainer logged.
    const visible = runs.map((r, i) => ({ r, s: series[i] })).filter(({ r }) => !ui.hidden.has(r.run));
    const keys = [...KNOWN];
    const seen = new Set(KNOWN.map((k) => k[0] + ":" + k[1]));
    for (const { s } of visible) for (const src of ["metrics", "eval"]) for (const row of s[src].slice(-5)) for (const [k, v] of Object.entries(row)) {
      if (SKIP.has(k) || typeof v !== "number" || seen.has(src + ":" + k)) continue;
      seen.add(src + ":" + k); keys.push([src, k, k.replace(/_/g, " "), (x) => compact(x)]);
    }
    charts = [];
    const grid = h("div", { class: "chart-grid" });
    for (const [src, key, title, f] of keys) {
      const ser = visible.map(({ r, s }) => ({ label: r.label, slot: r.slot,
        points: s[src].filter((row) => typeof row[key] === "number").map((row) => [xOf(row), row[key]]) })).filter((x) => x.points.length);
      if (!ser.length) continue;
      const c = lineChart({ title, series: ser, fmt: f, xFmt });
      grid.append(c); charts.push(c);
    }
    holder.append(grid);
    requestAnimationFrame(redrawCharts);

    // Table view: latest values per run (every charted value is reachable without hovering).
    holder.append(h("h2", { class: "section" }, "Latest"));
    const cols = [["Run", (r) => r.label], ["State", (r) => r.status?.state || "–"], ["Phase", (r) => r.last_metrics?.phase || "–"],
      ["Step", (r) => (r.last_metrics ? fmt(r.last_metrics.step) : "–")], ["Training time", (r) => fmtHours(r.last_metrics?.elapsed_s)],
      ["Train loss", (r) => r.last_metrics?.train_loss?.toFixed(3) ?? "–"], ["Val loss", (r) => r.last_eval?.val_loss?.toFixed(3) ?? "–"],
      ["Meter", (r) => (r.last_eval?.meter_rate != null ? pct(r.last_eval.meter_rate) : "–")],
      ["Rhyme", (r) => (r.last_eval?.rhyme_rate != null ? pct(r.last_eval.rhyme_rate) : "–")], ["Samples", (r) => fmt(r.n_samples)]];
    holder.append(h("div", { class: "table-wrap" }, h("table", { class: "data-table" },
      h("thead", {}, h("tr", {}, cols.map(([c]) => h("th", { scope: "col" }, c)))),
      h("tbody", {}, runs.map((r) => h("tr", {}, cols.map(([, f], i) => i === 0
        ? h("th", { scope: "row" }, h("span", { class: "key", style: `background: var(--s${(r.slot % 8) + 1})` }), f(r))
        : h("td", {}, f(r)))))))));

    // Reply samples, newest first.
    holder.append(h("h2", { class: "section" }, "Replies during training"),
      h("p", { class: "summary" }, "Every 15 minutes of training, each model answers the same held-out couplets, so you can watch the replies change."));
    if (!samples.items.length) holder.append(h("p", { class: "empty" }, "No replies yet. The first arrive 15 minutes into training."));
    const bySlot = Object.fromEntries(runs.map((r) => [r.run, r]));
    for (const smp of samples.items) {
      const r = bySlot[smp.run] || { label: smp.run, slot: 0 };
      holder.append(h("div", { class: "sample" },
        h("div", { class: "sample-head" }, h("span", { class: "key", style: `background: var(--s${(r.slot % 8) + 1})` }),
          h("strong", {}, r.label), ` · after ${fmtHours(smp.elapsed_s)} · step ${fmt(smp.step)}${smp.phase ? ` · ${smp.phase}` : ""}`),
        h("div", { class: "pair" },
          h("div", { class: "verse" }, h("div", { class: "who" }, "Prompt"), verseLines({ lines: (smp.prompt || "").split("\n") })),
          h("span", { class: "arrow", "aria-label": "answered by" }, "→"),
          h("div", { class: "verse" }, h("div", { class: "who" }, [
            "Reply", smp.meter_ok != null ? (smp.meter_ok ? " · meter ✓" : " · meter ✗") : "",
            smp.rhyme_ok != null ? (smp.rhyme_ok ? " · rhyme ✓" : " · rhyme ✗") : "",
            smp.cy_score != null ? ` · Cypriot ${smp.cy_score}` : ""].join("")),
            verseLines({ lines: (smp.reply || "").split("\n"), syl: smp.syllables, rhyme_ok: smp.rhyme_ok })))));
    }
  }

  window.MougiaTraining = { viewTraining };
})();
