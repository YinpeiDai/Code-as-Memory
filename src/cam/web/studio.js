"use strict";

// Visual task cards open independent replay pages. Each replay keeps its own
// selected attempt and playback position across automatic data refreshes.
const Studio = {
  rows: [], key: null, attempt: null,
  query: new URLSearchParams(location.search),
  url(row, attempt = null, view = null) {
    const params = new URLSearchParams({ task: `${row.benchmark}/${row.name}` });
    if (row.run) params.set("run", row.run.id);
    if (attempt != null) params.set("attempt", attempt);
    if (view) params.set("view", view);
    return "?" + params + "#development";
  },
  bench: (name) => ({ mikasa_robo: "MIKASA-Robo", mikasa_robo_v2: "MIKASA-Robo v2", rmbench: "RMBench", robomemarena: "RoboMemArena", robomme: "RoboMME" })[name] || name,
  short: (name) => name.replace(/-VLA-v\d+$/, ""),
  benchmarkLabel(row) {
    const backend = row.run?.config.physics_backend;
    return this.bench(row.benchmark) + (backend ? ` · ${backend.toUpperCase()} physics` : "");
  },
  evaluations(row) { return (row?.run?.episodes || []).filter((e) => e.task === row.name); },
  evaluationUrl(row) {
    const episodes = this.evaluations(row), episode = episodes.find((e) => e.video) || episodes[0];
    return episode ? "./#rollout=" + encodeURIComponent(`test|${row.run.id}|${row.name}|${episode.maximum_rollouts}|${episode.episode}`) : null;
  },
  status(row) {
    if (row.task?.frozen || row.task?.status === "finished") return "Policy frozen";
    if (row.task?.status === "running") return "Developing";
    if (row.batch?.status === "preparation_failed") return "Preparation blocked";
    if (row.batch?.status === "failed") return "Needs attention";
    if (row.batch?.status === "waiting_for_validated_pool") return "Waiting for RGB-D validation";
    return row.batch?.status === "extracting" ? "Preparing demos" : "Waiting to start";
  },
  init() {
    if (this.ready) return;
    this.ready = true;
    this.detail = this.query.has("task");
    this.demoView = this.query.get("view") === "demos";
    document.body.classList.toggle("task-detail", this.detail);
    document.body.classList.toggle("demo-detail", this.demoView);
    document.getElementById("gallery-search").addEventListener("input", () => this.renderGallery());
    document.getElementById("gallery-failures").addEventListener("change", () => this.renderGallery());
    document.getElementById("studio-outcome").addEventListener("change", () => this.renderAttempts());
    document.getElementById("development").addEventListener("click", (e) => {
      const button = e.target.closest("[data-studio-action]");
      if (!button) return;
      if (button.dataset.studioAction === "expand") window.CaMExplorer.Explorer.expand();
      if (button.dataset.studioAction === "log") window.open(this.url(this.current(), this.attempt, "log"), "_blank", "noopener");
      if (button.dataset.studioAction === "code") window.open(this.url(this.current(), this.attempt, "code"), "_blank", "noopener");
      if (button.dataset.studioAction === "demos") window.open(Demos.url(this.current()), "_blank", "noopener");
    });
    window.addEventListener("cam:rolloutchange", (e) => {
      const { Explorer } = window.CaMExplorer;
      if (!Explorer.embedded) return;
      const row = this.current();
      const match = row?.task?.rollouts.find((r) => e.detail.key === `train|${row.task.development_url}|${r.attempt}`);
      if (match) {
        this.attempt = match.attempt;
        this.renderAttempts();
        this.renderHeader();
      }
    });
    window.addEventListener("cam:player-return", () => {
      const row = this.current(), player = window.CaMExplorer.Explorer;
      if (!row?.task?.rollouts.length) return this.renderWorkspace(true);
      if (player.items[player.index]?.key !== `train|${row.task.development_url}|${this.attempt}`) this.play(this.attempt);
    });
  },
  current() { return this.rows.find((r) => r.key === this.key); },
  update(data, runs, benchmark, runFilter) {
    this.init();
    const batch = data.batch?.tasks || [];
    this.rows = runs.flatMap((run) => run.development.map((task) => ({
      key: task.development_url, name: task.task, benchmark: run.benchmark, task, run,
      batch: batch.find((r) => r.benchmark === run.benchmark && r.task === task.task),
    })));
    if (!runFilter) for (const row of batch) {
      if (benchmark && row.benchmark !== benchmark) continue;
      if (!this.rows.some((r) => r.benchmark === row.benchmark && r.name === row.task))
        this.rows.push({ key: `${row.benchmark}/${row.task}`, name: row.task, benchmark: row.benchmark, batch: row });
    }
    const frozen = this.rows.filter((r) => r.task?.frozen).length;
    const active = this.rows.filter((r) => r.task?.status === "running").length;
    const videos = this.rows.reduce((n, r) => n + (r.task?.rollouts.filter((a) => a.video).length || 0), 0);
    const testVideos = this.rows.reduce((n, r) => n + this.evaluations(r).filter((e) => e.video).length, 0);
    document.getElementById("studio-summary").innerHTML = [
      [this.rows.length, "tasks"], [frozen, "policies frozen"], [active, "developing"], [videos, "training videos"], [testVideos, "evaluation videos"],
    ].map(([n, label]) => `<div><b>${n}</b><span>${label}</span></div>`).join("") +
      '<p><span class="status-dot"></span>Training development <span class="muted">· refreshes automatically</span></p>';
    Demos.library(this.rows);
    if (!this.detail) { this.renderGallery(); return; }
    let reset = !this.current();
    if (reset) {
      const chosen = this.rows.find((r) => `${r.benchmark}/${r.name}` === this.query.get("task") && (!this.query.has("run") || r.run?.id === this.query.get("run")));
      this.key = chosen?.key;
      this.attempt = this.query.has("attempt") ? Number(this.query.get("attempt")) : null;
    }
    if (this.demoView) { this.renderHeader(); Demos.show(this.current()); return; }
    this.renderWorkspace(reset);
    if (!this.openedView && this.current()?.task && ["code", "log"].includes(this.query.get("view"))) {
      this.openedView = true;
      this.log(this.query.get("view") === "code");
    }
  },
  renderGallery() {
    const query = document.getElementById("gallery-search").value.toLowerCase();
    const failures = document.getElementById("gallery-failures").checked;
    const rows = this.rows.filter((r) => `${r.name} ${r.benchmark}`.toLowerCase().includes(query) && (!failures || [...(r.task?.rollouts || []), ...this.evaluations(r)].some((a) => a.status !== "success")));
    document.getElementById("studio-gallery").innerHTML = rows.map((row) => {
      const attempts = row.task?.rollouts || [], latest = [...attempts].reverse().find((a) => a.video);
      const evaluations = this.evaluations(row), demoOnly = row.run?.config.mode === "demos";
      const success = attempts.filter((a) => a.status === "success").length;
      const goal = latest?.goal || row.task?.reference_demos?.[0]?.goal;
      return `<article class="replay-card">
        <div class="replay-preview">${latest?.poster ? `<img src="${X.esc(latest.poster)}" alt="Recorded camera views for ${X.esc(row.name)}" loading="lazy">` : `<div class="preview-placeholder">◇<span>${demoOnly ? "Demo-only development" : "No recorded training rollout yet"}</span></div>`}
        ${latest ? `<span class="preview-play">▶</span><span class="preview-attempt">R${latest.attempt} · ${latest.steps} steps</span><span class="preview-status ${X.outcomeClass(latest.status)}">${X.esc(latest.status.replace(/_/g, " "))}</span>` : ""}</div>
        <div class="replay-card-body"><div class="eyebrow">${X.esc(this.benchmarkLabel(row))}<span>${X.esc(this.status(row))}</span></div>
        <h3>${X.esc(this.short(row.name))}</h3><p>${attempts.length ? `<b>${attempts.length}</b> training attempts · <span class="good">${success} successful</span> · <span class="failure-text">${attempts.length - success} failed</span>` : demoOnly ? "Demo-only · no training rollouts" : "Development video will appear here."}</p>
        ${goal ? `<p class="card-goal" title="${X.esc(goal)}"><b>Task goal</b> ${X.esc(goal)}</p>` : ""}
        ${evaluations.length ? `<div class="card-evaluation"><a href="${X.esc(this.evaluationUrl(row))}" target="_blank" rel="noopener">Watch evaluation (${evaluations.length} episodes) ↗</a><span>${evaluations.filter((e) => e.status === "success").length} successful</span></div>` : ""}
        ${row.task?.reference_demos?.length ? `<div class="card-reference"><a href="${X.esc(Demos.url(row))}" target="_blank" rel="noopener">${row.task.reference_demos.length} reference demos ↗</a></div>` : ""}
        <a class="card-open" href="${X.esc(this.url(row, latest?.attempt))}" target="_blank" rel="noopener" aria-label="Open ${X.esc(row.name)} in a new tab"><span>${latest ? "Watch rollout" : "Task details"} ↗</span></a>
        <div class="card-attempts" aria-label="Recorded attempts">${attempts.map((r) => `<a class="mini-attempt ${X.outcomeClass(r.status)}" href="${X.esc(this.url(row, r.attempt))}" target="_blank" rel="noopener" title="R${r.attempt}: ${X.esc(r.status)} · ${r.steps} steps">R${r.attempt}</a>`).join("")}</div></div></article>`;
    }).join("") || '<div class="empty">No tasks match these filters.</div>';
  },
  renderHeader() {
    const row = this.current();
    if (!row) { document.getElementById("studio-task-header").innerHTML = ""; return; }
    const selected = row.task?.rollouts.find((a) => a.attempt === this.attempt);
    const hasCode = Object.keys(selected?.policy_files || {}).length;
    const evaluations = this.evaluations(row);
    document.title = `${this.short(row.name)} · ${this.demoView ? "Reference demos" : this.query.get("view") === "log" ? "Agent log" : this.query.get("view") === "code" ? "Code" : "R" + (this.attempt ?? "—")} · CaM`;
    document.getElementById("studio-task-header").innerHTML = `<div class="studio-task-title"><a class="back-link" href="./#development">← Rollout library</a><div class="eyebrow">${X.esc(this.benchmarkLabel(row))} <span> / ${X.esc(this.demoView ? "Reference demonstrations" : this.status(row))}</span></div><h2 title="${X.esc(row.name)}">${X.esc(this.short(row.name))}</h2><p>${row.task ? this.demoView ? `${row.task.reference_demos?.length || 0} assigned training demonstrations · reference behavior` : `${row.task.demonstrations} offline demos · ${row.task.attempts.length}/${row.run.config.budget} interactions` : "Demonstrations are being prepared for this task."}</p></div>
      <div class="studio-actions">${evaluations.length ? `<a class="button evaluation-link" href="${X.esc(this.evaluationUrl(row))}" target="_blank" rel="noopener">Evaluation (${evaluations.length} episodes) ↗</a>` : ""}${this.demoView ? `<a class="button" href="${X.esc(this.url(row))}" target="_blank" rel="noopener">Policy rollouts ↗</a>` : `<button class="reference-link" data-studio-action="demos"${row.task?.reference_demos?.length ? "" : " disabled"}>Reference demos (${row.task?.reference_demos?.length || 0}) ↗</button><button data-studio-action="log"${row.task ? "" : " disabled"}>Agent log</button><button data-studio-action="code"${hasCode ? "" : " disabled"}>Executed code${selected ? ` · R${selected.attempt}` : ""}</button><button data-studio-action="expand"${selected?.video ? "" : " disabled"} aria-label="Expand video player">⛶ Expand</button>`}</div>`;
  },
  renderWorkspace(reset) {
    const row = this.current(), attempts = row?.task?.rollouts || [];
    if (!attempts.some((a) => a.attempt === this.attempt)) {
      this.attempt = ([...attempts].reverse().find((a) => a.video) || attempts.at(-1))?.attempt;
      reset = true;
    }
    this.renderHeader();
    this.renderAttempts();
    Demos.references(row);
    if (!attempts.length) {
      const player = window.CaMExplorer.Explorer;
      if (player.embedded && player.root) { player.video.pause(); player.root.hidden = true; player.root.remove(); }
      const demoOnly = row?.run?.config.mode === "demos", evaluations = this.evaluations(row);
      document.getElementById("studio-player").innerHTML = `<div class="studio-empty"><span>▷</span><h3>${row?.batch?.status === "preparation_failed" ? "Demo preparation needs attention" : demoOnly ? "Demo-only development" : "Waiting for the first rollout"}</h3><p>${demoOnly ? "This run uses demonstrations without training rollouts. The agent log and reference demos are available above." : row?.task ? "The agent's development log is available above. A video will appear here when its first attempt finishes." : "This task has no development rollout yet."}</p>${evaluations.length ? `<a class="button evaluation-link" href="${X.esc(this.evaluationUrl(row))}" target="_blank" rel="noopener">Watch evaluation (${evaluations.length} episodes) ↗</a>` : ""}${row?.batch?.error ? `<details><summary>Preparation details</summary><pre>${X.esc(row.batch.error)}</pre></details>` : ""}</div>`;
    } else if (reset) this.play(this.attempt);
    else {
      const player = window.CaMExplorer.Explorer;
      if (player.embedded) {
        player.items = window.CaMExplorer.trainItems(row.task, row.run);
        player.index = attempts.findIndex((a) => a.attempt === this.attempt);
      }
    }
  },
  renderAttempts() {
    const rows = this.current()?.task?.rollouts || [], filter = document.getElementById("studio-outcome").value;
    const visible = rows.filter((r) => filter === "all" || (filter === "success" ? r.status === "success" : r.status !== "success"));
    document.getElementById("studio-attempts").innerHTML = visible.map((r) => `<a class="attempt-card ${X.outcomeClass(r.status)}${r.attempt === this.attempt ? " selected" : ""}" href="${X.esc(this.url(this.current(), r.attempt))}" target="_blank" rel="noopener" aria-label="Open R${r.attempt}: ${X.esc(r.status)} in a new tab"><b>R${String(r.attempt).padStart(2, "0")}</b><span>${X.esc(r.status.replace(/_/g, " "))}</span><small>${r.steps} steps ↗</small></a>`).join("") || '<p class="muted">No completed attempts for this selection.</p>';
  },
  play(attempt) {
    const row = this.current();
    if (!row?.task) return;
    const index = row.task.rollouts.findIndex((a) => a.attempt === attempt);
    if (index < 0) return;
    this.attempt = attempt;
    this.renderHeader();
    this.renderAttempts();
    window.CaMExplorer.Explorer.mount(window.CaMExplorer.trainItems(row.task, row.run), index, document.getElementById("studio-player"));
  },
  async log(code) {
    const row = this.current();
    if (!row?.task) return;
    const { DevLog, Explorer } = window.CaMExplorer;
    Explorer.video?.pause();
    DevLog.current = code ? "code" : "timeline";
    DevLog.selectedCodeAttempt = code ? this.attempt : null;
    try { await DevLog.open(row.task.development_url, row.task, row.run); }
    catch (e) { DevLog.body.textContent = "Unable to load the log: " + e.message; }
  },

};
window.CaMStudio = Studio;
