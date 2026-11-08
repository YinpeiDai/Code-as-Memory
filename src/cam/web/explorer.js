"use strict";
// Annotated rollout explorer and chronological development log.
// Both open as full-window views. Overlays use the rollout's own trace
// (observations.h5 / actions.jsonl / decisions.jsonl): frame 0 is the reset
// observation and frame i ≥ 1 is the observation produced by action i−1.

const X = {
  esc: (x) =>
    String(x ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]),
  el(html) {
    const t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  },
  fmt(v) {
    if (v == null) return "—";
    if (typeof v === "number") return Math.abs(v) >= 1000 || Number.isInteger(v) ? String(v) : v.toFixed(3);
    if (Array.isArray(v)) return "[" + v.map(X.fmt).join(", ") + "]";
    if (typeof v === "object") return "{" + Object.entries(v).map(([k, x]) => k + ": " + X.fmt(x)).join(", ") + "}";
    return String(v);
  },
  args(a, limit = 140) {
    const text = Object.entries(a || {}).map(([k, v]) => `${k}=${X.fmt(v)}`).join(", ");
    return text.length > limit ? text.slice(0, limit) + "…" : text;
  },
  outcomeClass: (s) => (s === "success" ? "ok" : s === "infrastructure_error" ? "infra" : s ? "bad" : "none"),
  duration(s) {
    if (s == null) return "";
    s = Math.round(s);
    return s >= 3600 ? `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m` : s >= 60 ? `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s` : s + "s";
  },
  color(name) {
    let h = 0;
    for (const c of String(name)) h = (h * 31 + c.charCodeAt(0)) % 360;
    return `hsl(${h} 70% 55%)`;
  },
  POINT_COLORS: { TCP: "#ffd400", "left EE": "#00e5ff", "right EE": "#ff4fd8" },
  stage: (name) => String(name || "Observe").replace(/^act_/, "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
};

// Minimal Markdown: code fences, inline code, bold, headings and lists.
function markdown(text) {
  const parts = String(text || "").split(/```[^\n]*\n?/);
  return parts
    .map((part, i) => {
      if (i % 2) return `<pre class="md-code">${X.esc(part.replace(/\n$/, ""))}</pre>`;
      return X.esc(part)
        .replace(/`([^`\n]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
        .replace(/^#{1,6} (.*)$/gm, '<span class="md-h">$1</span>')
        .replace(/^(\s*)[-*] /gm, "$1• ");
    })
    .join("");
}

/* ------------------------------------------------------------------ */
/* Rollout explorer                                                    */
/* ------------------------------------------------------------------ */

const Explorer = {
  items: [],
  index: 0,
  trace: null,
  frame: -1,
  root: null,
  traces: new Map(),

  open(items, index) {
    this.items = items;
    this.ensure();
    this.embedded = false;
    this.root.classList.remove("embedded");
    document.body.append(this.root);
    this.root.hidden = false;
    document.body.classList.add("modal-open");
    this.show(index);
  },
  embed(host) {
    this.homeHost = host;
    this.embedded = true;
    this.root.classList.add("embedded");
    host.replaceChildren(this.root);
    this.root.hidden = false;
    if (!DevLog.root || DevLog.root.hidden) document.body.classList.remove("modal-open");
    requestAnimationFrame(() => this.layout());
  },
  mount(items, index, host) {
    this.ensure();
    this.items = items;
    this.embed(host);
    this.show(index);
  },
  expand() {
    if (!this.root || this.root.hidden) return;
    this.embedded = false;
    this.root.classList.remove("embedded");
    document.body.append(this.root);
    document.body.classList.add("modal-open");
    requestAnimationFrame(() => this.layout());
  },
  close() {
    if (!this.root) return;
    this.video.pause();
    if (this.homeHost?.isConnected) {
      this.embed(this.homeHost);
      window.dispatchEvent(new Event("cam:player-return"));
    } else this.root.hidden = true;
    if (location.hash.startsWith("#rollout=")) history.replaceState(null, "", location.pathname + location.search);
    if (!DevLog.root || DevLog.root.hidden) document.body.classList.remove("modal-open");
  },
  ensure() {
    if (this.root) return;
    this.root = X.el(`
      <div class="fullview" id="explorer" hidden>
        <div class="fv-head">
          <div class="fv-title"></div>
          <div class="fv-actions">
            <button data-x="prev" title="Previous episode ([)">◀ Prev</button>
            <button data-x="next" title="Next episode (])">Next ▶</button>
            <button data-x="close" title="Close (Esc)">✕ Close</button>
          </div>
        </div>
        <div class="ex-body">
          <div class="ex-main">
            <div class="ex-goal" hidden></div>
            <div class="ex-current" aria-live="polite"></div>
            <div class="ex-stage-wrap">
              <div class="ex-stage">
                <video class="ex-src" playsinline muted preload="auto"></video>
                <div class="ex-tiles"></div>
                <div class="ex-banner" hidden></div>
                <button class="ex-play-overlay" data-x="play" aria-label="Play rollout video">▶<span>Play rollout</span></button>
              </div>
            </div>
            <div class="ex-hud"></div>
            <div class="ex-transport">
              <button data-x="play" title="Play/pause (space)" aria-label="Play or pause video">▶ Play</button>
              <button data-x="dprev" title="Previous decision (shift+←)">⏮</button>
              <button data-x="fprev" title="Previous frame (←)">−1</button>
              <button data-x="fnext" title="Next frame (→)">+1</button>
              <button data-x="dnext" title="Next decision (shift+→)">⏭</button>
              <span class="ex-clock"></span>
              <label>Speed <select data-x="speed">${[0.1, 0.25, 0.5, 1, 2].map((r) => `<option value="${r}">${r}×</option>`).join("")}</select></label>
              <label><input type="checkbox" data-x="markers" checked> Markers</label>
              <label><input type="checkbox" data-x="trail" checked> Trail</label>
              <label><input type="checkbox" data-x="hud" checked> Frame labels</label>
            </div>
            <input class="ex-scrubber" type="range" min="0" max="1" value="0" aria-label="Video frame">
            <div class="ex-timeline" title="Decisions over time; click to seek"></div>
            <div class="ex-step-labels" aria-label="Action stages"></div>
            <div class="ex-panels">
              <details class="ex-panel"><summary>Actions &amp; robot state at this frame</summary><div class="ex-data"></div></details>
              <details class="ex-panel"><summary>Episode result &amp; downloads</summary><div class="ex-result"></div></details>
            </div>
          </div>
          <div class="ex-side">
            <h4>Policy decisions</h4><p class="muted small">Click a stage to jump to that moment.</p><div class="muted small legend ex-dcount"></div>
            <div class="ex-decisions"></div>
          </div>
        </div>
        <div class="ex-strip"></div>
      </div>`);
    document.body.append(this.root);
    const q = (s) => this.root.querySelector(s);
    Object.assign(this, {
      video: q("video"), tilesBox: q(".ex-tiles"), hud: q(".ex-hud"), banner: q(".ex-banner"), stage: q(".ex-stage"), stageWrap: q(".ex-stage-wrap"),
      timeline: q(".ex-timeline"), decisionsBox: q(".ex-decisions"), data: q(".ex-data"), result: q(".ex-result"),
      strip: q(".ex-strip"), clock: q(".ex-clock"), title: q(".fv-title"), playButton: q('.ex-transport [data-x="play"]'),
      speed: q('[data-x="speed"]'),
      current: q(".ex-current"), labels: q(".ex-step-labels"), scrubber: q(".ex-scrubber"), playOverlay: q(".ex-play-overlay"),
      goalBox: q(".ex-goal"),
    });
    this.root.addEventListener("click", (e) => {
      const tile = e.target.closest(".ex-tile");
      if (tile) {
        this.focus = this.focus === tile.dataset.t ? null : tile.dataset.t;
        return this.layout();
      }
      const b = e.target.closest("[data-x],[data-seek],[data-ep]");
      if (!b) return;
      if (b.dataset.seek != null) return this.seek(+b.dataset.seek, true);
      if (b.dataset.ep != null) return this.show(+b.dataset.ep);
      const x = b.dataset.x;
      if (x === "close") this.close();
      else if (x === "prev") this.show(this.index - 1);
      else if (x === "next") this.show(this.index + 1);
      else if (x === "play") this.video.paused ? this.video.play() : this.video.pause();
      else if (x === "fprev") this.seek(this.frame - 1, true);
      else if (x === "fnext") this.seek(this.frame + 1, true);
      else if (x === "dprev" || x === "dnext") this.jumpDecision(x === "dnext" ? 1 : -1);
    });
    this.root.addEventListener("change", (e) => {
      if (e.target === this.speed) this.video.playbackRate = +this.speed.value;
      else this.draw(true);
    });
    this.scrubber.addEventListener("input", () => this.seek(+this.scrubber.value, true));
    this.timeline.addEventListener("pointerdown", (e) => {
      if (!this.trace?.frames) return;
      const seekAt = (ev) => {
        const r = this.timeline.getBoundingClientRect();
        this.seek(Math.floor(((ev.clientX - r.left) / r.width) * this.trace.frames), true);
      };
      seekAt(e);
      const move = (ev) => seekAt(ev);
      const up = () => (removeEventListener("pointermove", move), removeEventListener("pointerup", up));
      addEventListener("pointermove", move);
      addEventListener("pointerup", up);
    });
    this.video.addEventListener("play", () => { this.playButton.textContent = "❚❚ Pause"; this.playOverlay.hidden = true; });
    this.video.addEventListener("pause", () => { this.playButton.textContent = "▶ Play"; this.playOverlay.hidden = false; });
    this.video.addEventListener("error", () => { this.hud.textContent = "Video could not load. Open the recording directly from Episode result & downloads."; });
    this.video.addEventListener("seeked", () => (this.paint(), this.sync()));
    this.video.addEventListener("loadeddata", () => (this.paint(), this.sync()));
    const tick = (now, meta) => {
      this.paint();
      this.sync(meta && meta.mediaTime);
      this.video.requestVideoFrameCallback ? this.video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
    };
    this.video.requestVideoFrameCallback ? this.video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
    new ResizeObserver(() => this.layout()).observe(this.stageWrap);
    document.addEventListener("keydown", (e) => {
      if (this.root.hidden || !this.root.getClientRects().length || e.target.matches("input,select,textarea") || (this.embedded && DevLog.root && !DevLog.root.hidden)) return;
      const keys = {
        " ": () => (this.video.paused ? this.video.play() : this.video.pause()),
        ArrowLeft: () => (e.shiftKey ? this.jumpDecision(-1) : this.seek(this.frame - 1, true)),
        ArrowRight: () => (e.shiftKey ? this.jumpDecision(1) : this.seek(this.frame + 1, true)),
        "[": () => this.show(this.index - 1),
        "]": () => this.show(this.index + 1),
        Escape: () => { if (!this.embedded) this.close(); },
      };
      if (keys[e.key]) (e.preventDefault(), keys[e.key]());
    });
  },
  async show(index) {
    if (index < 0 || index >= this.items.length) return;
    this.index = index;
    const request = this.request = (this.request || 0) + 1;
    const item = this.items[index];
    if (this.embedded && new URLSearchParams(location.search).has("task") && item.attempt != null) {
      const params = new URLSearchParams(location.search);
      params.set("attempt", item.attempt);
      history.replaceState(null, "", "?" + params + "#development");
    } else if (item.key) history.replaceState(null, "", "#rollout=" + encodeURIComponent(item.key));
    window.dispatchEvent(new CustomEvent("cam:rolloutchange", { detail: item }));
    this.trace = null;
    this.frame = -1;
    this.title.innerHTML =
      `<span class="badge ${X.outcomeClass(item.status)}">${X.esc(item.status || "—")}</span> ` +
      `<strong>${X.esc(item.task)}</strong> · ${X.esc(item.kind)} ${X.esc(item.label)} <span class="muted">· ${X.esc(item.run)} · ${index + 1}/${this.items.length}</span>`;
    this.strip.innerHTML = this.items
      .map((it, i) => `<button class="chip ${X.outcomeClass(it.status)}${i === index ? " current" : ""}" data-ep="${i}" title="${X.esc(it.kind + " " + it.label + ": " + it.status)}">${X.esc(it.short ?? it.label)}</button>`)
      .join("");
    if (!this.embedded) this.strip.querySelector(".current")?.scrollIntoView({ block: "nearest", inline: "center" });
    this.tilesBox.innerHTML = "";
    this.tiles = [];
    this.hud.textContent = "Loading…";
    this.goalBox.hidden = true;
    this.current.innerHTML = '<span class="muted">Loading recorded video and action labels…</span>';
    this.labels.innerHTML = "";
    this.currentDecision = null;
    this.banner.hidden = true;
    this.decisionsBox.innerHTML = "";
    this.timeline.innerHTML = "";
    this.data.innerHTML = "";
    this.result.innerHTML = this.resultHtml(item, null);
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.poster = item.poster || "";
    this.video.controls = false;
    this.root.classList.remove("video-only");
    if (item.video) this.video.src = item.video;
    else this.hud.textContent = "No video recorded for this rollout.";
    if (!item.trace) {
      this.hud.textContent = item.video ? "No aligned trace for this rollout (video only)." : this.hud.textContent;
      this.fallbackVideo();
      return;
    }
    let trace = this.traces.get(item.trace);
    if (!trace) {
      try {
        trace = await fetch(item.trace).then((r) => r.json());
      } catch (e) {
        trace = { error: e.message };
      }
      this.traces.set(item.trace, trace);
    }
    if (this.request !== request) return;
    if (trace.error) {
      this.hud.textContent = "Trace unavailable: " + trace.error;
      this.fallbackVideo();
      return;
    }
    this.trace = trace;
    this.goalBox.hidden = !trace.result?.goal;
    this.goalBox.innerHTML = trace.result?.goal ? `<span>Task goal</span><p>${X.esc(trace.result.goal)}</p>` : "";
    this.scrubber.max = Math.max(0, trace.frames - 1);
    this.buildTiles(trace);
    const seconds = trace.frames / trace.fps;
    const rate = [1, 0.5, 0.25, 0.1].find((r) => seconds / r >= 8) ?? 0.1; // at least ~8 s on screen
    this.speed.value = String(rate);
    this.video.playbackRate = rate;
    this.renderTimeline();
    this.renderDecisions();
    this.result.innerHTML = this.resultHtml(item, trace);
    this.seek(0, true);
  },
  fallbackVideo() {
    this.root.classList.add("video-only");
    this.video.controls = true;
    this.current.textContent = "Recorded video · action labels unavailable";
    this.playOverlay.hidden = true;
  },
  buildTiles(trace) {
    const tiles = trace.tiles.length
      ? trace.tiles
      : [{ name: "video", label: "", x: 0, width: trace.video_width || 256, height: trace.video_height || 256 }];
    if (this.focus && !tiles.some((t) => t.name === this.focus)) this.focus = null;
    this.tilesBox.innerHTML = tiles
      .map((t) => `<div class="ex-tile" data-t="${X.esc(t.name)}" title="Click to enlarge / restore"><canvas width="${t.width}" height="${t.height}"></canvas><svg viewBox="0 0 ${t.width} ${t.height}" preserveAspectRatio="none"></svg></div>`)
      .join("");
    this.tiles = tiles.map((t) => {
      const el = this.tilesBox.querySelector(`.ex-tile[data-t="${CSS.escape(t.name)}"]`);
      return { ...t, el, canvas: el.querySelector("canvas"), ctx: el.querySelector("canvas").getContext("2d"), svg: el.querySelector("svg") };
    });
    this.layout();
  },
  layout() {
    if (!this.tiles?.length) return;
    const shown = this.tiles.filter((t) => !this.focus || t.name === this.focus);
    const gap = 6;
    const width = this.stageWrap.clientWidth - 12;
    const height = this.embedded ? Math.max(230, Math.min(480, innerHeight * 0.48)) : Math.max(160, innerWidth <= 980 ? innerHeight * 0.5 : innerHeight - 340);
    const aspect = shown[0].width / shown[0].height;
    let best = { w: 0, c: 1 };
    for (let c = 1; c <= shown.length; c++) {
      const r = Math.ceil(shown.length / c);
      const w = Math.min((width - gap * (c - 1)) / c, ((height - gap * (r - 1)) / r) * aspect);
      if (w > best.w) best = { w, c };
    }
    this.tilesBox.style.gridTemplateColumns = `repeat(${best.c}, ${Math.floor(best.w)}px)`;
    for (const t of this.tiles) {
      t.el.hidden = !shown.includes(t);
      t.el.style.width = Math.floor(best.w) + "px";
      t.el.style.height = Math.floor((best.w * t.height) / t.width) + "px";
      t.canvas.classList.toggle("pixelated", best.w / t.width >= 1.6);
    }
    this.draw(true);
  },
  paint() {
    if (!this.tiles?.length || this.video.readyState < 2) return;
    for (const t of this.tiles) if (!t.el.hidden) t.ctx.drawImage(this.video, t.x, 0, t.width, t.height, 0, 0, t.width, t.height);
  },
  resultHtml(item, trace) {
    const r = (trace && trace.result) || {};
    const rows = [
      ["Outcome", `<span class="badge ${X.outcomeClass(item.status)}">${X.esc(item.status)}</span>`],
      ["Episode", X.esc(item.episode)],
      ["Native steps", X.esc(r.steps ?? item.steps)],
      ["Decisions", X.esc(r.decisions ?? "—")],
      ["Seconds", r.seconds != null ? r.seconds.toFixed(1) : "—"],
    ];
    if (r.difficulty) rows.push(["Difficulty", X.esc(r.difficulty)]);
    if (r.error) rows.push(["Error", `<span class="bad-text">${X.esc(r.error)}</span>`]);
    if (r.native_metrics && Object.keys(r.native_metrics).length) rows.push(["Native metrics", X.esc(X.fmt(r.native_metrics))]);
    const downloads = Object.entries(item.downloads || {}).filter(([, u]) => u).map(([n, u]) => `<a href="${X.esc(u)}" download>${X.esc(n)}</a>`).join(" · ");
    return (
      `<table class="kv">${rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("")}</table>` +
      (downloads ? `<p class="downloads">Downloads: ${downloads}</p>` : "")
    );
  },
  renderTimeline() {
    const t = this.trace, n = Math.max(1, t.frames);
    const pct = (f) => (100 * f) / n;
    let html = "";
    // Shade the task-provided video prefix (phase 0) separately from execution.
    const first = t.phase.indexOf(1);
    if (first > 0) html += `<div class="tl-demo" style="left:0;width:${pct(first)}%" title="task video (phase 0)"></div>`;
    t.decisions.forEach((d, k) => {
      const end = k + 1 < t.decisions.length ? t.decisions[k + 1].frame : n;
      html += `<div class="tl-seg" data-seek="${d.frame}" style="left:${pct(d.frame)}%;width:${Math.max(0.15, pct(end - d.frame))}%;background:${X.color(d.tool)}" title="#${k + 1} ${X.esc(d.tool)}(${X.esc(X.args(d.args, 80))}) · frame ${d.frame}">${pct(end - d.frame) > 7 ? X.esc(X.stage(d.tool)) : ""}</div>`;
    });
    html += '<div class="tl-head"></div>';
    this.timeline.innerHTML = html;
    this.labels.innerHTML = t.decisions.map((d, k) => `<button data-seek="${d.frame}" data-stage="${k}" style="--stage-color:${X.color(d.tool)}"><b>${k + 1}</b><span>${X.esc(X.stage(d.tool))}</span><small>step ${d.step}</small></button>`).join("");
    const tools = [...new Set(t.decisions.map((d) => d.tool))];
    this.timeline.insertAdjacentHTML("afterend", "");
    this.root.querySelector(".ex-dcount").innerHTML =
      `${t.decisions.length} · ` + tools.map((x) => `<span class="swatch" style="background:${X.color(x)}"></span>${X.esc(x)}`).join(" ");
  },
  renderDecisions() {
    const t = this.trace;
    this.decisionsBox.innerHTML =
      t.decisions
        .map(
          (d, k) => `
        <div class="dec" data-k="${k}">
          <button class="dec-head" data-seek="${d.frame}"><span class="swatch" style="background:${X.color(d.tool)}"></span><b>${k + 1}. ${X.esc(X.stage(d.tool))}</b><span class="muted">step ${d.step}</span></button>
          <div class="dec-raw"><code>${X.esc(d.tool)}</code></div>
          <div class="dec-args">${X.esc(X.args(d.args, 400))}</div>
          ${d.memory?.length ? `<table class="mem">${d.memory.map(([k2, v]) => `<tr><th>${X.esc(k2)}</th><td>${X.esc(v)}</td></tr>`).join("")}</table>` : ""}
          ${d.outcome ? `<details><summary>Outcome</summary><pre>${X.esc(d.outcome)}</pre></details>` : ""}
        </div>`,
        )
        .join("") || '<div class="muted">No decisions were recorded.</div>';
  },
  jumpDecision(direction) {
    const t = this.trace;
    if (!t?.decisions.length) return;
    const k = t.decision[Math.max(0, this.frame)] ?? -1;
    const target = Math.min(t.decisions.length - 1, Math.max(0, direction > 0 ? k + 1 : this.frame > (t.decisions[k]?.frame ?? 0) ? k : k - 1));
    this.seek(t.decisions[target].frame, true);
  },
  seek(frame, pause) {
    const t = this.trace;
    if (!t) return;
    frame = Math.max(0, Math.min(t.frames - 1, frame));
    if (pause) this.video.pause();
    if (this.video.readyState >= 1) this.video.currentTime = (frame + 0.5) / t.fps;
    this.setFrame(frame);
  },
  sync(mediaTime) {
    const t = this.trace;
    if (!t) return;
    const time = mediaTime ?? this.video.currentTime;
    this.setFrame(Math.max(0, Math.min(t.frames - 1, Math.floor(time * t.fps + 1e-3))));
  },
  setFrame(frame) {
    if (frame === this.frame) return;
    this.frame = frame;
    this.draw();
  },
  draw(force) {
    const t = this.trace, f = this.frame;
    if (!t || f < 0) return;
    const show = (name) => this.root.querySelector(`[data-x="${name}"]`).checked;
    for (const tile of this.tiles || []) {
      if (tile.el.hidden) continue;
      const unit = tile.width / Math.max(1, tile.el.clientWidth); // overlay units per screen pixel
      let svg = tile.label
        ? `<text x="${6 * unit}" y="${17 * unit}" class="ov-label" font-size="${13 * unit}" stroke-width="${2.4 * unit}">${X.esc(tile.label)}</text>`
        : "";
      if (show("markers"))
        for (const p of t.points[tile.name] || []) {
          const color = X.POINT_COLORS[p.label] || "#fff";
          const inside = (uv) => uv && uv[0] >= 0 && uv[0] < tile.width && uv[1] >= 0 && uv[1] < tile.height;
          if (show("trail")) {
            let run = [];
            const lines = [];
            for (let j = Math.max(0, f - 25); j <= f; j++) {
              if (inside(p.uv[j])) run.push(`${p.uv[j][0]},${p.uv[j][1]}`);
              else if (run.length) (lines.push(run), (run = []));
            }
            if (run.length) lines.push(run);
            for (const pts of lines)
              if (pts.length > 1) svg += `<polyline points="${pts.join(" ")}" fill="none" stroke="${color}" stroke-opacity="0.8" stroke-width="${2.2 * unit}" stroke-linejoin="round" stroke-linecap="round"/>`;
          }
          const uv = p.uv[f];
          if (inside(uv)) {
            const [x, y] = uv, r = 7 * unit;
            svg +=
              `<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="#000" stroke-width="${4 * unit}"/>` +
              `<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="${color}" stroke-width="${2.2 * unit}"/>` +
              `<circle cx="${x}" cy="${y}" r="${1.6 * unit}" fill="${color}"/>` +
              `<text x="${x + r * 1.3}" y="${y - r * 1.1}" class="ov-point" fill="${color}" font-size="${12 * unit}" stroke-width="${3 * unit}">${X.esc(p.label)}</text>`;
          }
        }
      tile.svg.innerHTML = svg;
    }
    const k = t.decision[f];
    const d = k >= 0 ? t.decisions[k] : null;
    this.scrubber.value = f;
    if (this.currentDecision !== k) {
      this.currentDecision = k;
      const labels = d?.labels || {};
      const states = Object.entries(labels).filter(([key, value]) => ["phase", "stage", "state", "mode", "target"].includes(key) && value !== "");
      this.current.innerHTML = `<div class="current-action"><span class="eyebrow">${d ? `DECISION ${k + 1} / ${t.decisions.length}` : "OBSERVATION"}</span><strong>${X.esc(d ? X.stage(d.tool) : "Observe scene")}</strong>${d ? `<code>${X.esc(d.tool)}</code>` : ""}</div><div class="current-labels">${states.map(([key, value]) => `<span>${X.esc(key)} <b>${X.esc(value)}</b></span>`).join("")}${labels.note || labels.reason ? `<p>${X.esc(labels.note || labels.reason)}</p>` : ""}</div>`;
      this.labels.querySelectorAll("[data-stage]").forEach((button) => { button.classList.toggle("active", +button.dataset.stage === k); button.setAttribute("aria-current", +button.dataset.stage === k ? "step" : "false"); });
    }
    this.hud.hidden = !show("hud");
    this.hud.innerHTML =
      `<span>frame <b>${f}</b>/${t.frames - 1}</span><span>step <b>${t.step[f]}</b></span><span class="${t.phase[f] ? "" : "demo"}">${t.phase[f] ? "execution" : "task video"}</span>` +
      (d ? `<span class="hud-act"><span class="swatch" style="background:${X.color(d.tool)}"></span>#${k + 1}/${t.decisions.length} <b>${X.esc(d.tool)}</b>(${X.esc(X.args(d.args, 90))})</span>` : "");
    const last = f === t.frames - 1;
    const status = this.items[this.index]?.status;
    this.banner.hidden = !last || !show("hud");
    this.banner.className = "ex-banner " + X.outcomeClass(status);
    this.banner.textContent = status === "success" ? "SUCCESS" : String(status || "").replace(/_/g, " ").toUpperCase();
    this.clock.textContent = `frame ${f} / ${t.frames - 1} · ${(f / t.fps).toFixed(2)} s`;
    this.timeline.querySelector(".tl-head")?.style.setProperty("left", `${(100 * (f + 0.5)) / t.frames}%`);
    // Aligned data: the command that produced this frame, then the robot state in it.
    const action = f >= 1 ? t.actions[f - 1] : null;
    const names = t.action_names || [];
    let data =
      `<div class="muted small">${f === 0 ? "Reset observation (no command yet)." : `Frame ${f} is the observation after command ${f - 1}.`}</div>` +
      (action
        ? `<div class="cmd-title">command ${f - 1} <span class="muted">(robot.step arguments)</span></div><div class="cmd-cells">${action.map((v, i) => `<div><span>${X.esc(names[i] ?? i)}</span>${X.fmt(v)}</div>`).join("")}</div>`
        : "");
    data += `<div class="cmd-title">robot state in frame ${f}</div><table class="vec">${Object.entries(t.state)
      .map(([name, rows]) => `<tr><th>${X.esc(name)}</th>${(rows[f] || []).map((v) => `<td>${X.fmt(v)}</td>`).join("")}</tr>`)
      .join("")}</table>`;
    this.data.innerHTML = data;
    if (force !== true || k !== this.activeDecision) {
      this.activeDecision = k;
      this.decisionsBox.querySelectorAll(".dec.active").forEach((x) => x.classList.remove("active"));
      const card = this.decisionsBox.querySelector(`.dec[data-k="${k}"]`);
      if (card) {
        card.classList.add("active");
        const box = this.decisionsBox;
        if (card.offsetTop < box.scrollTop || card.offsetTop + card.offsetHeight > box.scrollTop + box.clientHeight)
          box.scrollTop = card.offsetTop - box.clientHeight / 3;
      }
    }
  },
};

/* ------------------------------------------------------------------ */
/* Development log                                                     */
/* ------------------------------------------------------------------ */

const DevLog = {
  root: null,
  data: null,
  task: null,
  run: null,

  async open(url, task, run) {
    this.ensure();
    this.root.hidden = false;
    document.body.classList.add("modal-open");
    this.task = task;
    this.run = run;
    this.url = url;
    this.root.querySelector(".fv-title").innerHTML = `<strong>${X.esc(task?.task || "")}</strong> <span class="muted">· ${X.esc(run?.name || "")} · loading…</span>`;
    this.body.innerHTML = '<div class="muted">Loading development log…</div>';
    this.data = await fetch(url, { cache: "no-store" }).then((r) => r.json());
    this.task = { ...task, rollouts: this.data.rollouts || task?.rollouts || [] };
    this.renderHead();
    this.tab(this.current || "timeline");
  },
  close() {
    if (this.root) this.root.hidden = true;
    if (!Explorer.root || Explorer.root.hidden || Explorer.embedded) document.body.classList.remove("modal-open");
  },
  ensure() {
    if (this.root) return;
    this.root = X.el(`
      <div class="fullview" id="devlog" hidden>
        <div class="fv-head">
          <div class="fv-title"></div>
          <div class="fv-actions"><button data-d="refresh">Refresh log</button><button data-d="close" title="Close (Esc)">✕ Close</button></div>
        </div>
        <div class="dl-stats"></div>
        <div class="dl-tabs">
          <button data-tab="timeline">Timeline</button><button data-tab="prompt">Prompt</button><button data-tab="code">Code versions</button><button data-tab="rollouts">Training rollouts</button>
          <span class="dl-filters">
            <label><input type="checkbox" data-f="thinking" checked> Thinking</label>
            <label><input type="checkbox" data-f="output" checked> Tool output</label>
            <label><input type="checkbox" data-f="milestones"> Milestones only</label>
            <input type="search" data-f="search" placeholder="Search log…">
          </span>
        </div>
        <div class="dl-body"></div>
      </div>`);
    document.body.append(this.root);
    this.body = this.root.querySelector(".dl-body");
    this.root.addEventListener("click", (e) => {
      const b = e.target.closest("[data-d],[data-tab],[data-rollout],[data-code-attempt],[data-zoom]");
      if (!b) return;
      if (b.dataset.d === "close") this.close();
      if (b.dataset.d === "refresh") this.open(this.url, this.task, this.run);
      if (b.dataset.tab) this.tab(b.dataset.tab);
      if (b.dataset.rollout != null) this.openRollout(+b.dataset.rollout);
      if (b.dataset.codeAttempt != null) {
        this.selectedCodeAttempt = +b.dataset.codeAttempt;
        this.tab("code");
      }
      if (b.dataset.zoom) window.open(b.dataset.zoom, "_blank");
    });
    this.root.addEventListener("input", (e) => e.target.dataset.f && this.current === "timeline" && this.renderTimeline());
    this.root.addEventListener("change", (e) => {
      if (e.target.matches("[data-code-version]")) this.showCodeVersion(+e.target.value);
    });
    this.root.addEventListener("toggle", (e) => {
      const d = e.target;
      if (d.matches?.("details[data-diff]") && d.open && !d.dataset.done) {
        d.dataset.done = "1";
        d.querySelector(".diff").innerHTML = diffHtml(this.diffs[+d.dataset.diff]);
      }
    }, true);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !this.root.hidden && (!Explorer.root || Explorer.root.hidden || Explorer.embedded)) this.close();
    });
  },
  renderHead() {
    const t = this.task || {}, d = this.data, tl = d.timeline || [];
    const tools = tl.filter((e) => e.kind === "tool");
    const changed = tl.flatMap((e) => (e.kind === "revision" ? [e] : e.revisions || [])).filter((r) => r.changed);
    const summary = tl.filter((e) => e.kind === "summary");
    const cost = summary.reduce((s, e) => s + (e.cost || 0), 0);
    const seconds = summary.reduce((s, e) => s + (e.seconds || 0), 0);
    this.root.querySelector(".fv-title").innerHTML =
      `<span class="badge">${X.esc(t.status || "")}</span> <strong>${X.esc(d.task)}</strong> <span class="muted">· ${X.esc(this.run?.name || "")} · ${X.esc(this.run?.learning_setting || this.run?.config?.mode || "")} · ${X.esc(this.run?.config?.coding?.model || "")}</span>`;
    const stat = (label, value) => `<div><span>${label}</span><b>${value}</b></div>`;
    this.root.querySelector(".dl-stats").innerHTML =
      stat("Sessions", tl.filter((e) => e.kind === "session").length) +
      stat("Tool calls", tools.length) +
      stat("Commands", tools.filter((e) => e.name === "command").length) +
      stat("Images viewed", tools.filter((e) => e.name === "read_image").length) +
      stat("Code revisions", changed.length) +
      stat("Training rollouts", tools.filter((e) => e.name === "rollout").length) +
      stat("Wall time", X.duration(seconds) || "—") +
      stat("Reported cost", summary.length ? "$" + cost.toFixed(2) : "—");
  },
  tab(name) {
    this.current = name;
    this.root.querySelectorAll("[data-tab]").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
    this.root.querySelector(".dl-filters").hidden = name !== "timeline";
    const d = this.data;
    if (name === "timeline") return this.renderTimeline();
    if (name === "prompt") this.body.innerHTML = `<p class="muted">The exact prompt given to the coding model (also in the workspace as PROTOCOL.md).</p><pre class="prompt">${X.esc(d.prompt)}</pre>`;
    if (name === "code") {
      this.codeVersions = [];
      if (Object.keys(d.final_files || {}).length) this.codeVersions.push({ label: "Final frozen policy", files: d.final_files });
      const latest = (d.revisions || []).at(-1);
      if (latest) this.codeVersions.push({ label: `Latest saved revision #${latest.id}`, files: latest.files });
      for (const r of this.task?.rollouts || []) if (Object.keys(r.policy_files || {}).length) {
        this.codeVersions.push({ label: `R${r.attempt} · ${r.status} · episode ${r.episode} (executed policy)`, files: r.policy_files, attempt: r.attempt });
      }
      this.body.innerHTML = this.codeVersions.length
        ? `<label>Version <select data-code-version>${this.codeVersions.map((v, i) => `<option value="${i}">${X.esc(v.label)}</option>`).join("")}</select></label><div class="dl-files"></div>`
        : '<p class="muted">No saved code yet.</p>';
      const selected = this.codeVersions.findIndex((v) => v.attempt === this.selectedCodeAttempt);
      this.selectedCodeAttempt = null;
      if (this.codeVersions.length) this.showCodeVersion(Math.max(0, selected));
    }
    if (name === "rollouts") {
      const rows = this.task?.rollouts || [];
      this.body.innerHTML = rows.length
        ? `<div class="chips">${rows.map((r, i) => `<span><button class="chip big ${X.outcomeClass(r.status)}" data-rollout="${i}">R${String(r.attempt).padStart(3, "0")} · ep ${X.esc(r.episode)}<br><small>${X.esc(r.status)} · ${X.esc(r.steps)} steps</small></button>${Object.keys(r.policy_files || {}).length ? `<button data-code-attempt="${r.attempt}">Code</button>` : ""}</span>`).join("")}</div>`
        : '<p class="muted">No completed training rollouts yet.</p>';
    }
  },
  showCodeVersion(index) {
    const version = this.codeVersions[index];
    this.body.querySelector("[data-code-version]").value = String(index);
    const box = this.body.querySelector(".dl-files");
    box.innerHTML = "";
    for (const [file, url] of Object.entries(version.files || {})) {
      const card = X.el(`<details class="file"${/^tools\/|task\.json$/.test(file) ? " open" : ""}><summary><code>${X.esc(file)}</code> <a href="${X.esc(url)}" download>download</a></summary><pre>Loading…</pre></details>`);
      box.append(card);
      fetch(url).then((r) => r.text()).then((text) => (card.querySelector("pre").textContent = text)).catch(() => {});
    }
  },
  openRollout(i) {
    Explorer.open(trainItems(this.task, this.run), i);
  },
  renderTimeline() {
    const tl = this.data.timeline || [];
    const f = (n) => this.root.querySelector(`[data-f="${n}"]`);
    const showThinking = f("thinking").checked, showOutput = f("output").checked, milestones = f("milestones").checked;
    const search = f("search").value.trim().toLowerCase();
    const t0 = tl.find((e) => e.started)?.started;
    this.diffs = [];
    const rolloutIndex = new Map((this.task?.rollouts || []).map((r, i) => [Number(r.attempt), i]));
    const out = [];
    let lastStart = t0;
    for (const e of tl) {
      if (e.started) lastStart = e.started;
      const changedRevisions = (e.kind === "revision" ? [e] : e.revisions || []).filter((r) => r.changed);
      const milestone = e.kind === "text" || e.kind === "summary" || e.kind === "session" || e.name === "rollout" || e.name === "read_image" || changedRevisions.length;
      if (milestones && !milestone) continue;
      if (e.kind === "thinking" && !showThinking) continue;
      if (search && !JSON.stringify(e).toLowerCase().includes(search)) continue;
      const when = (t) => (t0 && t ? `<span class="when">+${X.duration(t - t0)}</span>` : "");
      const revisionHtml = (rs) =>
        rs
          .map((r) => {
            const files = [...r.diff.matchAll(/^\+\+\+ (?:after\/)?(\S+)/gm)].map((m) => m[1]);
            this.diffs.push(r.diff);
            return `<details class="rev" data-diff="${this.diffs.length - 1}"><summary>✎ <b>Code revision #${r.id}</b> <span class="muted">(${X.esc(r.reason)}, after ${r.attempts} attempt${r.attempts === 1 ? "" : "s"})</span> ${files.map((x) => `<code>${X.esc(x)}</code>`).join(" ")} ${r.diff_url ? `<a href="${X.esc(r.diff_url)}" download>diff</a>` : ""}</summary><div class="diff"></div></details>`;
          })
          .join("");
      if (e.kind === "session") out.push(`<div class="tl-session">${X.esc(e.session.replace("_", " "))}</div>`);
      else if (e.kind === "text") out.push(`<div class="card agent"><div class="who">Agent</div><div class="md">${markdown(e.text)}</div></div>`);
      else if (e.kind === "thinking")
        out.push(e.redacted ? `<div class="thinking-mark">thinking (content not recorded by the client)</div>` : `<details class="card thinking"><summary>Thinking <span class="muted">· ${e.text.length} chars · ${X.esc(e.text.slice(0, 110).replace(/\s+/g, " "))}…</span></summary><div class="md">${markdown(e.text)}</div></details>`);
      else if (e.kind === "summary")
        out.push(`<div class="card summary"><div class="who">Session finished · ${X.esc(e.status || "")}${e.turns ? ` · ${e.turns} turns` : ""}${e.seconds ? ` · ${X.duration(e.seconds)}` : ""}${e.cost != null ? ` · $${e.cost.toFixed(2)}` : ""}</div><div class="md">${markdown(e.text)}</div></div>`);
      else if (e.kind === "revision") {
        if (e.changed) out.push(`<div class="card revision">${revisionHtml([e])}</div>`);
        else if (e.reason === "checkpoint") out.push(`<div class="marker">⚑ checkpoint R${String(e.attempts).padStart(3, "0")} frozen (policy after ${e.attempts} attempt${e.attempts === 1 ? "" : "s"})</div>`);
        else if (e.reason === "final") out.push(`<div class="marker final">■ final policy frozen after ${e.attempts} attempt${e.attempts === 1 ? "" : "s"}</div>`);
      }
      else if (e.kind === "tool") {
        const dur = e.started && e.finished ? ` · ${(e.finished - e.started).toFixed(1)} s` : "";
        const err = e.is_error || (e.returncode != null && e.returncode !== 0);
        let head = `<span class="tool-name">${X.esc(e.name)}</span> ${when(e.started)}<span class="muted">${X.esc(e.id || "")}${dur}</span>`;
        let body = "";
        const outputBlock = (label, text, open, cls = "") =>
          text && showOutput ? `<details class="out ${cls}"${open ? " open" : ""}><summary>${label} <span class="muted">· ${text.split("\n").length} lines</span></summary><pre>${X.esc(text)}</pre></details>` : "";
        if (e.name === "command") {
          head += e.returncode != null ? ` <span class="badge ${e.returncode ? "bad" : "ok"}">exit ${e.returncode}</span>` : "";
          const cmd = e.input?.command ?? JSON.stringify(e.input);
          body += `<pre class="cmd">${X.esc(cmd)}</pre>`;
          body += outputBlock("Output", e.output, (e.output || "").split("\n").length <= 25);
          body += outputBlock("stderr", e.error_output, false, "stderr");
        } else if (e.name === "read_image") {
          body += `<div class="muted small">${X.esc(e.input?.path || "")}</div>`;
          body += e.image ? `<img class="viewed" loading="lazy" src="${X.esc(e.image)}" data-zoom="${X.esc(e.image)}" title="Click to open full size">` : '<div class="muted">Image not archived.</div>';
        } else if (e.name === "rollout") {
          const r = e.rollout || {};
          const i = rolloutIndex.get(Number(r.attempt));
          head += ` <span class="badge ${X.outcomeClass(r.status)}">R${String(r.attempt ?? "?").padStart(3, "0")} · episode ${X.esc(r.episode)} · ${X.esc(r.status)} · ${X.esc(r.steps)} steps</span>`;
          body += (i != null ? `<button class="primary" data-rollout="${i}">▶ Open annotated rollout</button> ` : "") + (r.remaining != null ? `<span class="muted">${r.remaining} attempts remaining</span>` : "");
          body += outputBlock("Tool response", e.output, false);
        } else {
          if (e.input && Object.keys(e.input).length) body += `<pre class="cmd">${X.esc(JSON.stringify(e.input, null, 1))}</pre>`;
          body += outputBlock("Response", e.output, (e.output || "").length < 400);
        }
        body += revisionHtml(changedRevisions);
        out.push(`<div class="card tool t-${X.esc(e.name)}${err ? " error" : ""}"><div class="tool-head">${head}</div>${body}</div>`);
      }
    }
    this.body.innerHTML = out.join("") || '<div class="muted">No matching entries.</div>';
  },
};

function diffHtml(text) {
  return String(text || "")
    .split("\n")
    .map((line) => {
      const cls = line.startsWith("+++") || line.startsWith("---") ? "f" : line.startsWith("@@") ? "h" : line.startsWith("+") ? "a" : line.startsWith("-") ? "d" : line.startsWith("diff ") ? "f" : "";
      return `<span class="${cls}">${X.esc(line)}</span>`;
    })
    .join("\n");
}

function trainItems(task, run) {
  return (task?.rollouts || []).map((r) => ({
    key: `train|${task.development_url}|${r.attempt}`,
    kind: "training",
    label: `R${String(r.attempt).padStart(3, "0")} · ep ${r.episode}`,
    short: `R${r.attempt}`,
    task: task.task,
    run: run?.name || "",
    episode: r.episode,
    attempt: r.attempt,
    status: r.status,
    steps: r.steps,
    video: r.video,
    poster: r.poster,
    trace: r.trace,
    downloads: { video: r.video, observations: r.observations, actions: r.actions, decisions: r.decisions_log, result: r.result_file, stderr: r.policy_stderr },
  }));
}

function testItems(run, episodes) {
  return episodes.map((e) => ({
    key: `test|${run.id}|${e.task}|${e.maximum_rollouts}|${e.episode}`,
    kind: "test",
    label: `episode ${e.episode} · R${e.maximum_rollouts}`,
    short: String(e.episode),
    task: e.task,
    run: run.name,
    episode: e.episode,
    status: e.status,
    steps: e.steps,
    video: e.video,
    trace: e.trace,
    downloads: { observations: e.observations, actions: e.actions, decisions: e.decisions_log, result: e.result_file },
  }));
}

window.CaMExplorer = { Explorer, DevLog, trainItems, testItems };
