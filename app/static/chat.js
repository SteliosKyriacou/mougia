// Chat tab: send a couplet to any checkpoint of any run and get a reply couplet back.
// Each message is an independent call → reply (that is how the poets were trained); the thread is kept on screen
// so you can duel back and forth.
(() => {
  const ui = { model: null, mode: "verse", style: "duel", temperature: 0.8, thread: [] };
  try { Object.assign(ui, JSON.parse(localStorage.getItem("mougia.chat") || "{}"), { thread: [] }); } catch {}
  const save = () => { try { localStorage.setItem("mougia.chat", JSON.stringify({ model: ui.model, mode: ui.mode, style: ui.style, temperature: ui.temperature })); } catch {} };

  async function viewChat(main, ctx) {
    const { api, setTitle, verseLines } = ctx;
    setTitle("Chat");
    const models = (await api("chat/models")).models;
    if (!models.length) {
      main.append(h("div", { class: "empty" }, h("p", {}, "No checkpoints yet."),
        h("p", { class: "summary", style: "margin: 8px auto; text-align: center" },
          "A run saves its first checkpoint 30 minutes into training. It will appear here automatically.")));
      return;
    }
    if (!models.find((m) => m.id === ui.model)) ui.model = models[0].id;

    const select = h("select", { class: "select", "aria-label": "Model to chat with", onchange: (e) => { ui.model = e.target.value; save(); } });
    const groups = {};
    for (const m of models) (groups[m.group] ||= []).push(m);
    for (const [g, ms] of Object.entries(groups)) select.append(h("optgroup", { label: g },
      ms.map((m) => h("option", { value: m.id, selected: m.id === ui.model }, `${m.label} · ${m.time}`))));
    const seg = (label, key, opts) => {
      const el = h("div", { class: "segmented", role: "group", "aria-label": label });
      const sync = () => el.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(String(ui[key]) === b.dataset.v)));
      opts.forEach(([v, l, title]) => el.append(h("button", { "data-v": String(v), title, onclick: () => { ui[key] = v; save(); sync(); } }, l)));
      sync();
      return el;
    };
    main.append(h("div", { class: "filter-row" }, select,
      seg("Decoding", "mode", [["verse", "Verse mode", "Two lines of ~15 syllables, best of 16 by rhyme"], ["raw", "Raw", "One unconstrained sample"]]),
      seg("Reply style", "style", [["duel", "Duel", "Answer like a rival ποιητάρης"], ["cont", "Continue", "Continue the song"]]),
      seg("Temperature", "temperature", [[0.6, "Calm"], [0.8, "Normal"], [1.0, "Wild"]])));

    const log = h("div", { class: "chat-log", "aria-live": "polite" });
    const input = h("textarea", { class: "chat-input", rows: 2, placeholder: "Γράψε ένα τσιαττιστό: δύο στίχοι των 15 συλλαβών…", "aria-label": "Your couplet" });
    const send = h("button", { class: "send-btn", onclick: () => submit() }, "Send");
    const hint = h("div", { class: "chat-hint" }, "Enter sends, Shift+Enter starts the second line.");
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } });
    const prompts = (await api("run_samples", { size: 50 })).items.map((s) => s.prompt).filter((v, i, a) => v && a.indexOf(v) === i).slice(0, 4);
    const examples = h("div", { class: "run-legend chat-examples" }, h("span", { class: "chip-state", style: "align-self:center" }, "Try:"),
      prompts.map((p) => h("button", { class: "run-chip", title: p, onclick: () => { input.value = p; input.focus(); } }, p.split("\n")[0].slice(0, 42) + "…")));
    main.append(log, h("div", { class: "composer" }, input, send), hint, prompts.length ? examples : null);
    render();

    function render() {
      log.replaceChildren(...(ui.thread.length ? ui.thread.map(bubble) : [h("p", { class: "empty" }, "Write a couplet and the fly will answer.")]));
      log.lastElementChild?.scrollIntoView({ block: "nearest" });
    }

    function bubble(m) {
      if (m.role === "user") return h("div", { class: "msg user" }, h("div", { class: "verse" }, verseLines({ lines: m.text.split("\n") })));
      if (m.pending) return h("div", { class: "msg bot" }, h("div", { class: "who" }, `${m.poet || "The fly"} is thinking…`), h("div", { class: "verse typing" }, "· · ·"));
      if (m.error) return h("div", { class: "msg bot" }, h("div", { class: "verse error" }, m.error));
      const s = m.scores;
      const others = (m.candidates || []).slice(1);
      return h("div", { class: "msg bot" },
        h("div", { class: "who" }, `${m.poet} · ${m.label} · step ${Number(m.step || 0).toLocaleString("en-US")}`),
        h("div", { class: "verse" }, verseLines({ lines: m.reply.split("\n"), syl: s.syllables, rhyme_ok: s.rhyme_ok })),
        h("div", { class: "turn-foot" }, h("span", {}, s.meter_ok ? "meter ✓" : "meter ✗"), h("span", {}, s.rhyme_ok ? "rhyme ✓" : "rhyme ✗"),
          h("span", {}, `Cypriot ${s.cy_score}`), h("span", {}, `${m.mode === "verse" ? "verse mode" : "raw"} · ${m.seconds}s`)),
        others.length ? h("details", { class: "cands" }, h("summary", {}, `${others.length} other candidates`),
          others.map((c) => h("div", { class: "cand" }, h("p", { class: "raw" }, c.reply),
            h("span", { class: "tip-label" }, `${c.scores.meter_ok ? "meter ✓" : "meter ✗"} · ${c.scores.rhyme_ok ? "rhyme ✓" : "rhyme ✗"} · Cypriot ${c.scores.cy_score} · logp ${c.logp}`)))) : null);
    }

    async function submit() {
      const text = input.value.trim();
      if (!text || send.disabled) return;
      const model = models.find((m) => m.id === ui.model);
      const poet = model.group.split(" ")[0];
      ui.thread.push({ role: "user", text });
      const pending = { role: "bot", pending: true, poet };
      ui.thread.push(pending);
      input.value = "";
      send.disabled = true;
      render();
      try {
        const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: ui.model, text, mode: ui.mode, style: ui.style, temperature: ui.temperature }) });
        const data = await res.json();
        Object.assign(pending, data.error ? { pending: false, error: data.error } : { ...data, pending: false, label: `${model.group.replace(/^[^·]*· ?/, "")} ${model.label}`.trim() });
      } catch (err) {
        Object.assign(pending, { pending: false, error: `Couldn’t reach the server (${err.message}).` });
      }
      send.disabled = false;
      render();
      input.focus();
    }
  }

  window.MougiaChat = { viewChat };
})();
