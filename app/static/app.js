// Mougia dataset browser — vanilla JS, hash routing, all dataset text rendered via textContent.
const $ = (sel, el = document) => el.querySelector(sel);
const main = $("#main");
const state = { showAll: false, overview: null };

const STATUS_LABEL = {
  kept: "In dataset", duplicate: "Duplicate", not_cypriot: "Filtered: not Cypriot", noisy_transcript: "Filtered: noisy transcript",
  whole_poem: "Whole poem (split elsewhere)", not_a_couplet: "Filtered: not a couplet", too_short: "Filtered: too short",
  low_mining_score: "Filtered: low mining score", empty: "Empty",
};

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) el.append(c.nodeType ? c : String(c));
  return el;
}

async function api(path, params = {}) {
  const q = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v != null));
  const res = await fetch(`/api/${path}?${q}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

const enc = encodeURIComponent;
const fmt = (n) => Number(n).toLocaleString("en-US");
const sourceName = (src) => state.overview?.sources.find((s) => s.src === src)?.name || src;
const fmtDate = (d) => (d ? String(d).slice(0, 10) : "");

function setTitle(text, back) {
  $("#title").textContent = text;
  document.title = `${text} — Mougia`;
  $("#back").hidden = !back;
}

// ---------- Sidebar ----------
function renderNav(route) {
  const nav = $("#nav");
  nav.replaceChildren();
  const o = state.overview;
  const link = (href, label, count) =>
    h("a", { href, "aria-current": location.hash === href || (href === "#/battles" && route === "battles" && !location.hash.includes("/", 2)) ? "page" : null },
      h("span", {}, label), count != null ? h("span", { class: "count" }, fmt(count)) : null);
  nav.append(h("h2", {}, "Library"),
    link("#/battles", "Battles", o.battles),
    link("#/pairs", "Call → reply pairs", o.pairs),
    link("#/couplets", "All couplets", o.kept),
    h("h2", {}, "Model"),
    link("#/training", "Training"));
  const groups = {};
  for (const s of o.sources) if (s.kept > 0 || state.showAll) (groups[s.group] ||= []).push(s);
  for (const g of ["Books", "Web", "Corpora", "YouTube"]) {
    if (!groups[g]) continue;
    nav.append(h("h2", {}, g), ...groups[g].map((s) => link(`#/source/${s.src}`, s.name, s.kept)));
  }
}

// ---------- Views ----------
async function viewBattles() {
  setTitle("Battles");
  const o = state.overview;
  main.append(h("p", { class: "summary" },
    h("strong", {}, fmt(o.kept)), " unique Cypriot couplets, ", h("strong", {}, fmt(o.good)), " passing strict 15-syllable meter and rhyme. ",
    h("strong", {}, fmt(o.battles)), " threads are battles: verse exchanged between two or more poets, shown in the order they were stored."));
  await threadList({ battles: 1 });
}

async function viewSource(src) {
  setTitle(sourceName(src));
  const s = state.overview.sources.find((x) => x.src === src);
  if (s) main.append(h("p", { class: "summary" }, h("strong", {}, fmt(s.kept)), ` couplets kept of ${fmt(s.records)} stored records, in ${fmt(s.threads)} threads`,
    s.battles ? `, ${fmt(s.battles)} of them battles.` : "."));
  await threadList({ src });
}

async function threadList(params, page = 0) {
  const holder = h("div");
  main.append(holder);
  const load = async (p) => {
    holder.replaceChildren(h("div", { class: "loading" }, "Loading…"));
    const data = await api("threads", { ...params, page: p, all: state.showAll ? 1 : "" });
    holder.replaceChildren();
    if (!data.total) return holder.append(h("p", { class: "empty" }, "No threads here."));
    holder.append(h("ul", { class: "list" }, data.items.map((m) => h("li", {},
      h("a", { class: "row", href: `#/thread/${m.src}/${enc(m.thread)}` },
        h("div", { class: "row-main" },
          h("div", { class: "row-title" }, m.battle ? h("span", { class: "tag battle" }, "Battle") : null, m.battle ? " " : null, m.title),
          h("div", { class: "row-sub" }, [params.src ? null : sourceName(m.src), m.n_authors ? `${m.n_authors} poet${m.n_authors > 1 ? "s" : ""}: ${m.authors.slice(0, 4).join(", ")}${m.n_authors > 4 ? "…" : ""}` : null, fmtDate(m.date)].filter(Boolean).join(" · "))),
        h("div", { class: "row-end" }, `${fmt(m.kept)} kept`, h("br"), `${fmt(m.n)} stored`),
        h("span", { class: "chev", "aria-hidden": "true" }, "›"))))));
    holder.append(pager(data, load));
  };
  await load(page);
}

function pager(data, load) {
  const pages = Math.ceil(data.total / data.size);
  if (pages <= 1) return h("div");
  return h("div", { class: "pager" },
    h("button", { disabled: data.page === 0, onclick: () => { load(data.page - 1); window.scrollTo(0, 0); } }, "Previous"),
    h("span", {}, `Page ${data.page + 1} of ${fmt(pages)} · ${fmt(data.total)} total`),
    h("button", { disabled: data.page >= pages - 1, onclick: () => { load(data.page + 1); window.scrollTo(0, 0); } }, "Next"));
}

function verseLines(r) {
  if (!r.lines) return h("p", { class: "raw" }, r.text);
  return h("div", {}, r.lines.map((line, k) => {
    const m = line.match(/^(.*\s)?(\S+)$/s) || [null, "", line];
    const n = r.syl?.[k];
    return h("div", { class: "line" },
      h("span", { class: "line-text" }, m[1] || "", h("span", { class: r.rhyme_ok ? "rhyme" : null }, m[2])),
      n != null ? h("span", { class: `syl${n < 13 || n > 17 ? " off" : ""}`, title: `${n} syllables (with synaloepha)` }, n) : null);
  }));
}

async function viewThread(src, thread, focus) {
  setTitle("…", true);
  const data = await api("thread", { src, thread });
  if (data.error) return main.append(h("p", { class: "empty" }, data.error));
  const m = data.meta;
  setTitle(m.title, true);
  const recs = data.records.filter((r) => state.showAll || r.status === "kept" || r.status === "duplicate");
  const hidden = data.records.length - recs.length;
  main.append(h("div", { class: "thread-head" },
    h("div", { class: "meta" }, [sourceName(m.src), m.n_authors ? `${m.n_authors} poet${m.n_authors > 1 ? "s" : ""}` : null, fmtDate(m.date)].filter(Boolean).join(" · "),
      m.url ? [" · ", h("a", { href: m.url, target: "_blank", rel: "noopener noreferrer" }, "Original source ↗")] : null),
    hidden ? h("div", { class: "meta" }, `${fmt(hidden)} stored records hidden (filtered out). Switch to “Everything stored” to see them.`) : null));
  if (!recs.length) return main.append(h("p", { class: "empty" }, "Nothing kept from this thread."));

  // Sides: the first poet on the left, the second on the right, further poets alternate.
  const sides = new Map();
  const byRid = new Map(data.records.map((r, k) => [String(r.rid), { r, k }]));
  const numOf = new Map(data.records.map((r, k) => [r.i, k + 1]));
  const ol = h("ol", { class: "duel" });
  for (const r of recs) {
    if (m.battle && r.author && !sides.has(r.author)) sides.set(r.author, sides.size % 2 ? "right" : "left");
    const side = m.battle ? sides.get(r.author) || "left" : "left";
    const filtered = !(r.status === "kept" || r.status === "duplicate");
    const parent = r.reply_to != null ? byRid.get(String(r.reply_to)) : null;
    ol.append(h("li", { class: `turn ${side}${filtered ? " filtered" : ""}`, id: `r${r.i}` },
      h("div", { class: "turn-head" },
        h("span", { class: "num" }, `#${numOf.get(r.i)}`),
        r.author ? h("span", { class: "who" }, r.author) : null,
        r.date ? h("span", {}, fmtDate(r.date)) : null,
        r.start != null ? h("span", {}, `${Math.floor(r.start / 60)}:${String(Math.floor(r.start % 60)).padStart(2, "0")}`) : null,
        parent ? h("a", { class: "answers", href: `#r${parent.r.i}`, onclick: (e) => { e.preventDefault(); flash(parent.r.i); } }, `answers #${parent.k + 1}${parent.r.author ? ` (${parent.r.author})` : ""}`) : null,
        !parent && r.reply_to_user ? h("span", { class: "answers" }, `to ${r.reply_to_user}`) : null),
      h("div", { class: "verse" }, verseLines(r)),
      h("div", { class: "turn-foot" },
        h("span", { class: filtered ? "status-filtered" : "status-kept" }, STATUS_LABEL[r.status] || r.status),
        r.cid ? h("span", {}, r.cid) : null,
        r.lines ? h("span", {}, `${r.meter_ok ? "meter ✓" : "meter ✗"} · ${r.rhyme_ok ? "rhyme ✓" : "rhyme ✗"}`) : null,
        r.kind && r.kind !== "couplet" ? h("span", {}, r.kind) : null)));
  }
  main.append(ol);
  if (focus != null) requestAnimationFrame(() => flash(Number(focus)));
}

function flash(i) {
  const el = document.getElementById(`r${i}`);
  if (!el) return;
  el.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
}

async function viewPairs() {
  setTitle("Call → reply pairs");
  main.append(h("p", { class: "summary" }, "Each pair is a couplet and the couplet that answered it — the training signal for replying like a ποιητάρης. ",
    "“Explicit” pairs come from recorded reply links; “consecutive” pairs are successive turns by different poets in one thread."));
  const holder = h("div");
  main.append(holder);
  const load = async (p) => {
    const data = await api("pairs", { page: p, good: state.showAll ? "" : "" });
    holder.replaceChildren(...data.items.map((p) => h("div", { class: "pair" },
      h("div", { class: "verse" }, h("div", { class: "who" }, "Call"), verseLines({ lines: p.call_text.split("\n") })),
      h("span", { class: "arrow", "aria-label": "answered by" }, "→"),
      h("div", { class: "verse" }, h("div", { class: "who" }, `Reply · ${sourceName(p.source)} · ${p.link}${p.good ? " · strict ✓" : ""}`), verseLines({ lines: p.reply_text.split("\n") })))),
      pager(data, load));
  };
  await load(0);
}

async function viewSearch(q) {
  setTitle(q ? `“${q}”` : "All couplets");
  $("#q").value = q;
  const holder = h("div");
  main.append(holder);
  const load = async (p) => {
    holder.replaceChildren(h("div", { class: "loading" }, "Searching…"));
    const data = await api("search", { q, page: p, all: state.showAll ? 1 : "" });
    holder.replaceChildren(h("p", { class: "summary" }, `${fmt(data.total)} ${state.showAll ? "stored records" : "couplets"}${q ? " match" : ""}.`));
    if (!data.total) return holder.append(h("p", { class: "empty" }, "No verses match. Search ignores accents, so try a single word."));
    holder.append(h("ul", { class: "list" }, data.items.map((r) => h("li", {},
      h("a", { class: "row", href: `#/thread/${r.src}/${enc(r.thread)}/${r.i}` },
        h("div", { class: "row-main" },
          h("div", { class: "row-title", style: "font-family: var(--verse); white-space: normal" }, (r.text || "").split("\n").join(" / ")),
          h("div", { class: "row-sub" }, [r.author, sourceName(r.src), r.status !== "kept" ? STATUS_LABEL[r.status] : null].filter(Boolean).join(" · "))),
        h("span", { class: "chev", "aria-hidden": "true" }, "›"))))), pager(data, load));
  };
  await load(0);
}

// ---------- Router ----------
async function route() {
  closeSidebar();
  main.replaceChildren();
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  const [view, a, b, c] = parts;
  renderNav(view || "battles");
  try {
    if (view === "source" && a) await viewSource(a);
    else if (view === "thread" && a && b != null) await viewThread(a, decodeURIComponent(b), c);
    else if (view === "pairs") await viewPairs();
    else if (view === "training") await window.MougiaTraining.viewTraining(main, { api, setTitle, verseLines, fmt });
    else if (view === "couplets") await viewSearch("");
    else if (view === "search") await viewSearch(decodeURIComponent(a || ""));
    else await viewBattles();
  } catch (err) {
    main.replaceChildren(h("p", { class: "empty" }, `Couldn’t load this view (${err.message}). Is the server still running?`));
  }
  main.focus({ preventScroll: true });
}

function setShowAll(v) {
  state.showAll = v;
  $("#show-kept").setAttribute("aria-pressed", String(!v));
  $("#show-all").setAttribute("aria-pressed", String(v));
  try { localStorage.setItem("mougia.showAll", v ? "1" : ""); } catch {}
  route();
}

function openSidebar() { $("#sidebar").classList.add("open"); $("#scrim").hidden = false; $("#menu-btn").setAttribute("aria-expanded", "true"); }
function closeSidebar() { $("#sidebar").classList.remove("open"); $("#scrim").hidden = true; $("#menu-btn").setAttribute("aria-expanded", "false"); }

$("#show-kept").onclick = () => setShowAll(false);
$("#show-all").onclick = () => setShowAll(true);
$("#menu-btn").onclick = openSidebar;
$("#scrim").onclick = closeSidebar;
$("#back").onclick = () => (history.length > 1 ? history.back() : (location.hash = "#/battles"));
$("#search-form").onsubmit = (e) => { e.preventDefault(); location.hash = `#/search/${enc($("#q").value.trim())}`; };
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== $("#q")) { e.preventDefault(); $("#q").focus(); }
  if (e.key === "Escape") closeSidebar();
});
window.addEventListener("hashchange", route);

(async () => {
  try { state.showAll = localStorage.getItem("mougia.showAll") === "1"; } catch {}
  $("#show-kept").setAttribute("aria-pressed", String(!state.showAll));
  $("#show-all").setAttribute("aria-pressed", String(state.showAll));
  main.append(h("div", { class: "loading" }, "Loading dataset…"));
  state.overview = await api("overview");
  route();
})();
