"use strict";
(() => {
  const $ = (s) => document.querySelector(s), esc = X.esc;
  const fmt = (n) => Number(n).toLocaleString(), pct = (n) => n == null ? "Pending" : `${(n * 100).toFixed(2)}%`;
  const pretty = (s) => s.replace(/^T\d+_/, "").replace(/([a-z])([A-Z])/g, "$1 $2").replace(/_/g, " ");
  const short = (s) => s.split("_")[0];
  const url = (page, args = {}) => `#${page}${Object.keys(args).length ? "?" + new URLSearchParams(Object.entries(args).filter(([, v]) => v != null && v !== "")) : ""}`;
  let data, archive, runs, state, library = [], visible = [], opener;
  const main = $("#arena-main"), player = $("#raw-player"), pageSize = 18;
  const experiment = (id) => archive.experiments.find((e) => e.id === id);
  const label = (id) => experiment(id)?.label || "All experiments";
  const badge = (r) => r.kind === "demo" ? '<span class="arena-badge demo">Expert demo</span>' :
    `<span class="arena-badge ${r.scorable === false ? "infra" : r.success ? "success" : "failure"}">${r.scorable === false ? "Infrastructure error" : r.success ? "Success" : "Failure"}</span>`;
  const taskOptions = (value = "") => `<option value="">All 26 tasks</option>${archive.tasks.map((t) => `<option value="${esc(t)}"${value === t ? " selected" : ""}>${short(t)} · ${esc(pretty(t))}</option>`).join("")}`;
  const settingOptions = (value = "") => `<option value="">All experiments</option>${archive.experiments.map((e) => `<option value="${e.id}"${value === e.id ? " selected" : ""}>${esc(e.label)}</option>`).join("")}`;
  const hero = (eyebrow, title, text) => `<div class="arena-hero"><span class="arena-eyebrow">${eyebrow}</span><h1>${title}</h1><p>${text}</p></div>`;
  const download = (href, text) => href ? `<a href="${esc(href)}" target="_blank" rel="noopener">${text} ↗</a>` : "";

  function indexRecordings() {
    library = [];
    for (const run of runs) for (const dev of run.development) {
      for (const r of dev.reference_demos || []) library.push({ ...r, task: dev.task, kind: "demo", run, dev, setting: run.experiment, success: true });
      for (const r of dev.rollouts || []) library.push({ ...r, task: dev.task, kind: "train", run, dev, setting: run.experiment });
      for (const r of run.episodes.filter((e) => e.task === dev.task && e.maximum_rollouts === run.config.budget))
        library.push({ ...r, kind: "test", run, dev, setting: run.experiment });
    }
  }
  function route() {
    if (!archive) return;
    closeRaw();
    const [name, query = ""] = location.hash.slice(1).split("?");
    const args = Object.fromEntries(new URLSearchParams(query));
    state = { page: ["videos", "development"].includes(name) ? name : "results", kind: "test", setting: "", task: "", outcome: "all", q: "", p: 1, ...args };
    if (name === "demonstrations") Object.assign(state, { page: "videos", kind: "demo" });
    if (name === "rollouts") state.page = "videos";
    document.querySelectorAll(".arena-header nav a").forEach((a) => {
      const active = a.hash === `#${state.page}`;
      a.classList.toggle("active", active);
      if (active) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    document.title = `${state.page === "results" ? "Results" : state.page === "videos" ? "Video gallery" : "Development logs"} · RoboMemArena`;
    if (state.page === "results") renderResults();
    if (state.page === "videos") renderGallery();
    if (state.page === "development") renderDevelopment();
    if (name.startsWith("rollout=")) legacyRollout(decodeURIComponent(name.slice(8)));
  }
  function go(page, args) { location.hash = url(page, args); }
  function selections() { return { kind: state.kind, setting: state.setting, task: state.task, outcome: state.outcome, q: state.q }; }

  function renderResults() {
    const total = archive.experiments.reduce((s, e) => s + e.scored, 0);
    const baseline = experiment("demo_only_50"), interaction = experiment("demo_interaction_50");
    const delta = baseline?.complete && interaction?.complete ? (interaction.tsr - baseline.tsr) * 100 : null;
    main.innerHTML = hero("THE COMPLETE COMPARISON", "From demonstrations to reliable behavior.", "26 tasks. Three learning settings. The same 50 held-out episodes per task. Official task success rate (TSR).") +
      `<div class="arena-coverage"><span><i></i>${fmt(total)} held-out episodes scored</span><span>Claude Code / Opus 5.5</span><span>256 × 256 RGB-D</span></div>` +
      `<div class="result-cards">${archive.experiments.map((e, i) => `<article class="result-card arm-${i}"><div class="result-card-top"><span class="arm-dot"></span><h2>${esc(e.label)}</h2><span class="arena-badge demo">${e.complete ? "Complete" : "In progress"}</span></div><div class="result-rate">${pct(e.tsr)}</div><p>${fmt(e.successes)} successful / ${fmt(e.expected)} expected</p><div class="score-bar"><span style="width:${100 * (e.tsr || 0)}%"></span></div><div class="result-facts"><span><b>${e.perfect_tasks}/26</b> perfect tasks</span><span><b>${e.failures}</b> failed episodes</span><span><b>${fmt(e.demos)}</b> assigned demos</span><span><b>${fmt(e.training_attempts)}</b> training attempts</span></div><a class="arena-text-link" href="${url("videos", { kind: "test", setting: e.id })}">Explore evaluation videos <span>→</span></a></article>`).join("")}</div>` +
      (delta == null ? "" : `<div class="comparison-callout"><strong>+${delta.toFixed(2)} percentage points</strong><p>50 demos + interaction recorded <b>${interaction.successes - baseline.successes} more successes</b> than 50-demo only on the same held-out episodes. The prompt also changed; this is not an interaction-only ablation.</p></div>`) +
      `<section class="arena-section"><div class="section-heading"><div><span class="arena-eyebrow">TASK BY TASK</span><h2>Where each solution succeeds</h2><p>Each cell shows TSR and successful episodes out of 50. Select a cell to watch its rollouts.</p></div><a class="arena-button secondary" id="download-results" download="robomemarena-results.csv">Download CSV ↓</a></div><div class="result-controls"><label><input id="results-failures" type="checkbox"> Only tasks with failures</label><label>Order <select id="results-sort"><option value="task">Task number</option><option value="hardest">Lowest interaction success first</option><option value="gain">Largest gain from interaction</option></select></label></div><div id="task-matrix"></div></section>` +
      `<section class="arena-section"><div class="section-heading"><div><span class="arena-eyebrow">FAILURE COVERAGE</span><h2>All failures remain in the archive</h2><p>Recorded termination statuses describe how episodes ended, not their diagnosed causes.</p></div></div><div class="failure-cards">${archive.experiments.map((e) => `<article><h3>${esc(e.label)}</h3><strong>${e.failures} <small>failed episodes</small></strong><div class="failure-statuses">${Object.entries(e.failure_statuses).sort((a, b) => b[1] - a[1]).map(([s, n]) => `<span><span>${esc(s.replace(/_/g, " "))}</span><b>${n}</b></span>`).join("") || "No scored failures"}</div><p>${e.infrastructure} infrastructure errors · ${e.scored}/${e.expected} scored</p><a href="${url("videos", { kind: "test", setting: e.id, outcome: "failure" })}">Watch failures →</a></article>`).join("")}</div></section>` +
      `<section class="protocol-note"><h2>How to read this comparison</h2><p>${esc(archive.comparison_note)}</p><p>TSR requires completion of the whole task. Training outcomes reflect changing policies and are kept separate from held-out results. The 25-demo subset contains 650 episodes from the same 1,282-demo RGB-D pool. T04 has 32 demos in the larger assignments and uses 32 interactions; every other task uses 50. Every final evaluation has 50 episodes.</p></section>`;
    renderMatrix();
    $("#results-failures").onchange = renderMatrix;
    $("#results-sort").onchange = renderMatrix;
    const lines = [["Task", ...archive.experiments.flatMap((e) => [`${e.label}: successes`, `${e.label}: scored`, `${e.label}: expected`, `${e.label}: TSR`])], ...archive.tasks.map((t) => [t, ...archive.experiments.flatMap((e) => { const r = e.tasks[t]; return [r?.successes ?? "", r?.scored ?? "", r?.expected ?? "", r?.tsr ?? ""]; })])];
    $("#download-results").href = URL.createObjectURL(new Blob([lines.map((row) => row.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n")], { type: "text/csv" }));
  }
  function renderMatrix() {
    let tasks = archive.tasks.filter((t) => !$("#results-failures").checked || archive.experiments.some((e) => e.tasks[t]?.failures));
    const score = (t, arm) => experiment(arm)?.tasks[t]?.tsr ?? -1;
    if ($("#results-sort").value === "hardest") tasks.sort((a, b) => score(a, "demo_interaction_50") - score(b, "demo_interaction_50"));
    if ($("#results-sort").value === "gain") tasks.sort((a, b) => (score(b, "demo_interaction_50") - score(b, "demo_only_50")) - (score(a, "demo_interaction_50") - score(a, "demo_only_50")));
    $("#task-matrix").innerHTML = `<div class="matrix-wrap"><table class="task-matrix"><caption class="sr-only">Per-task held-out success across the three experiments</caption><thead><tr><th scope="col">Task</th>${archive.experiments.map((e) => `<th scope="col">${esc(e.label)}</th>`).join("")}<th scope="col">Interaction gain<small>vs. 50-demo only</small></th></tr></thead><tbody>${tasks.map((t) => {
      const a = score(t, "demo_only_50"), b = score(t, "demo_interaction_50"), gain = a < 0 || b < 0 ? null : (b - a) * 100;
      return `<tr><th scope="row"><span class="task-number">${short(t)}</span>${esc(pretty(t))}</th>${archive.experiments.map((e) => { const r = e.tasks[t]; return r ? `<td><a class="score-cell ${r.tsr === 1 ? "perfect" : r.tsr == null ? "pending" : r.tsr < .6 ? "low" : "mixed"}" href="${url("videos", { kind: "test", setting: e.id, task: t, outcome: r.failures ? "failure" : "all" })}" aria-label="${esc(t)} ${esc(e.label)}: ${r.successes} of ${r.expected} successes. View ${r.failures ? "failures" : "videos"}."><b>${pct(r.tsr)}</b><span>${r.successes}/${r.expected}${r.complete ? "" : ` · ${r.scored} scored`}</span></a></td>` : "<td>Unavailable</td>"; }).join("")}<td class="gain ${gain > 0 ? "positive" : ""}">${gain == null ? "—" : `${gain > 0 ? "+" : ""}${gain.toFixed(0)} pp`}</td></tr>`;
    }).join("")}</tbody></table></div>`;
  }

  function renderGallery() {
    if (!["demo", "train", "test"].includes(state.kind)) state.kind = "test";
    main.innerHTML = hero("THE RECORDINGS", "Watch the behavior.", "Browse expert demonstrations, development attempts, and held-out evaluations. Open any recording for unannotated camera playback.") +
      `<div class="gallery-tabs" role="group" aria-label="Recording type">${[["test", "Held-out rollouts", "3,900"], ["train", "Training rollouts", "1,282"], ["demo", "Demo videos", "1,282 unique"]].map(([id, title, count]) => `<a href="${url("videos", { kind: id, task: state.task })}" class="${id === state.kind ? "active" : ""}"${id === state.kind ? ' aria-current="page"' : ""}>${title}<span>${count}</span></a>`).join("")}</div>` +
      `<div class="gallery-filters"><label>Task<select id="video-task">${taskOptions(state.task)}</select></label><label>${state.kind === "demo" ? "Demo assignment" : "Experiment"}<select id="video-setting">${settingOptions(state.setting)}</select></label><label${state.kind === "demo" ? " hidden" : ""}>Outcome<select id="video-outcome"><option value="all">All outcomes</option><option value="success"${state.outcome === "success" ? " selected" : ""}>Successes</option><option value="failure"${state.outcome === "failure" ? " selected" : ""}>Failures</option><option value="infra"${state.outcome === "infra" ? " selected" : ""}>Infrastructure errors</option></select></label><label class="search-label">Search<input id="video-search" type="search" placeholder="Task name or episode…" value="${esc(state.q)}"></label></div>` +
      `<div class="gallery-note">${state.kind === "demo" ? "Recorded RGB frames from replay-derived RGB-D demos · front and wrist cameras · no policy annotations. Shared demos are shown once when all experiments are selected." : state.kind === "train" ? "Development attempts on assigned demo seeds. Policies may change between attempts; these outcomes are not held-out performance. Demo-only runs have no training rollouts." : "Held-out execution of frozen final policies. All recorded successes and failures are available."}</div><div id="gallery-content"></div>`;
    for (const [id, key] of [["video-task", "task"], ["video-setting", "setting"], ["video-outcome", "outcome"]]) $("#" + id).onchange = (e) => go("videos", { ...selections(), [key]: e.target.value, p: 1 });
    $("#video-search").oninput = (e) => { state.q = e.target.value; state.p = 1; history.replaceState(null, "", url("videos", selections())); renderGalleryCards(); };
    renderGalleryCards();
  }
  function filteredRecordings() {
    let rows = library.filter((r) => r.kind === state.kind && (!state.setting || r.setting === state.setting) && (!state.task || r.task === state.task));
    if (state.kind === "demo") {
      const seen = new Set(); rows = rows.filter((r) => { const k = `${r.task}:${r.episode}`; if (seen.has(k)) return false; seen.add(k); return true; });
    } else rows = rows.filter((r) => state.outcome === "all" || (state.outcome === "infra" ? r.scorable === false : r.scorable !== false && (state.outcome === "success" ? r.success : !r.success)));
    const query = state.q.trim().toLowerCase();
    const order = (s) => archive.experiments.findIndex((e) => e.id === s);
    return rows.filter((r) => !query || `${r.task} ${pretty(r.task)} episode ${r.episode} ${r.attempt ? "R" + r.attempt : ""}`.toLowerCase().includes(query)).sort((a, b) => a.task.localeCompare(b.task) || (a.attempt ?? a.episode) - (b.attempt ?? b.episode) || order(a.setting) - order(b.setting));
  }
  function recordingTitle(r) { return r.kind === "train" ? `Attempt R${String(r.attempt).padStart(3, "0")}` : `${r.kind === "demo" ? "Demo" : "Episode"} ${String(r.episode).padStart(3, "0")}`; }
  function renderGalleryCards() {
    const rows = filteredRecordings();
    if (!state.task && rows.length) return renderTaskGallery(rows);
    const pages = Math.max(1, Math.ceil(rows.length / pageSize));
    state.p = Math.min(pages, Math.max(1, Number(state.p) || 1));
    const start = (state.p - 1) * pageSize; visible = rows.slice(start, start + pageSize);
    $("#gallery-content").innerHTML = `<div class="gallery-count"><strong>${fmt(rows.length)} recordings</strong><span>${rows.length ? `${start + 1}–${Math.min(start + pageSize, rows.length)} shown` : "No matching recordings"}</span></div><div class="video-gallery">${visible.map((r, i) => `<article class="video-card"><button class="video-thumbnail" data-watch="${i}" aria-label="Watch ${esc(r.task)} ${esc(recordingTitle(r))}">${r.poster ? `<img src="${esc(r.poster)}" alt="Front and wrist camera views" loading="lazy">` : ""}<span class="play-icon" aria-hidden="true">▶</span><span class="thumbnail-task">${short(r.task)}</span>${badge(r)}</button><div class="video-card-body"><p class="video-context">${r.kind === "demo" && !state.setting ? "Assigned expert demonstration" : esc(label(r.setting))}</p><h2>${esc(pretty(r.task))}</h2><div class="video-details"><strong>${recordingTitle(r)}</strong><span>${fmt(r.steps ?? r.frames)} ${r.kind === "demo" ? "frames" : "steps"}</span></div><div class="video-card-actions"><button data-watch="${i}">Watch raw video ↗</button>${r.kind !== "demo" ? `<button data-inspect="${i}" aria-label="Inspect decisions for ${esc(recordingTitle(r))}">Decisions &amp; state</button>` : ""}</div></div></article>`).join("")}</div>${!rows.length ? `<div class="arena-empty"><h2>No recordings match these filters.</h2><p>${state.kind === "train" ? "Only the demo + interaction experiment has training rollouts." : "Try another task, experiment, or outcome."}</p><a class="arena-button" href="${url("videos", { kind: state.kind })}">Reset filters</a></div>` : ""}<div class="gallery-pagination"><button data-page="${state.p - 1}"${state.p === 1 ? " disabled" : ""}>← Previous</button><span>Page ${state.p} of ${pages}</span><button data-page="${state.p + 1}"${state.p === pages ? " disabled" : ""}>Next →</button></div>`;
  }
  function renderTaskGallery(rows) {
    const groups = new Map();
    for (const r of rows) { if (!groups.has(r.task)) groups.set(r.task, []); groups.get(r.task).push(r); }
    const tasks = [...groups.keys()], pages = Math.max(1, Math.ceil(tasks.length / pageSize));
    state.p = Math.min(pages, Math.max(1, Number(state.p) || 1));
    const start = (state.p - 1) * pageSize;
    visible = tasks.slice(start, start + pageSize).map((t) => groups.get(t).find((r) => r.setting === 'demo_interaction_50') || groups.get(t)[0]);
    $("#gallery-content").innerHTML = `<div class="gallery-count"><strong>${fmt(rows.length)} recordings · ${tasks.length} tasks</strong><span>Choose a task to browse its videos</span></div><div class="video-gallery">${visible.map((r, i) => {
      const group = groups.get(r.task), success = group.filter((r) => r.success).length;
      return `<article class="video-card"><button class="video-thumbnail" data-watch="${i}" aria-label="Play preview for ${esc(r.task)}"><img src="${esc(r.poster)}" alt="${esc(pretty(r.task))}: recorded camera views" loading="lazy"><span class="play-icon" aria-hidden="true">▶</span><span class="thumbnail-task">${short(r.task)}</span><span class="arena-badge demo">${group.length} videos</span></button><div class="video-card-body"><p class="video-context">${state.setting ? esc(label(state.setting)) : state.kind === 'demo' ? 'Assigned expert demonstrations' : `${new Set(group.map((r) => r.setting)).size} experiment${new Set(group.map((r) => r.setting)).size > 1 ? 's' : ''}`}</p><h2>${esc(pretty(r.task))}</h2><div class="video-details"><span>${state.kind === 'demo' ? `${group.length} unique demos` : `${success} successful · ${group.length - success} unsuccessful`}</span></div><div class="video-card-actions"><button data-group="${esc(r.task)}">Browse ${group.length} videos →</button><button data-watch="${i}">Play preview</button></div></div></article>`;
    }).join('')}</div><div class="gallery-pagination"><button data-page="${state.p - 1}"${state.p === 1 ? ' disabled' : ''}>← Previous</button><span>Page ${state.p} of ${pages}</span><button data-page="${state.p + 1}"${state.p === pages ? ' disabled' : ''}>Next →</button></div>`;
  }
  function openRaw(r, trigger) {
    opener = trigger;
    player.innerHTML = `<div class="raw-heading"><div><span class="arena-eyebrow">${r.kind === "demo" ? "EXPERT DEMONSTRATION" : "RAW CAMERA RECORDING"}</span><h2>${short(r.task)} · ${esc(pretty(r.task))}</h2><p>${recordingTitle(r)} · ${esc(label(r.setting))}</p></div><button class="raw-close" data-close-raw aria-label="Close video">✕</button></div><div class="raw-video-stage"><video controls playsinline preload="auto" poster="${esc(r.poster || "")}" src="${esc(r.video)}"></video></div><div class="raw-video-status" role="status">${r.kind === "demo" ? "Preparing the video from recorded RGB frames. The first playback may take a few seconds…" : "Loading recording…"}</div><div class="raw-foot"><span>${badge(r)} Front camera · wrist camera</span><div>${download(r.video, "Open video")}${r.kind !== "demo" ? '<button data-raw-inspect>Decisions &amp; state →</button>' : ""}</div></div><p class="raw-goal">${esc(r.goal || "")}</p>`;
    const video = player.querySelector("video"), status = player.querySelector(".raw-video-status");
    video.addEventListener("loadeddata", () => { status.textContent = "Original camera playback · no overlays"; });
    video.addEventListener("error", () => { status.textContent = "Video could not load. Close and reopen to retry, or use Open video."; });
    player.querySelector("[data-raw-inspect]")?.addEventListener("click", () => { closeRaw(); inspect(r); });
    player.showModal(); video.play().catch(() => {});
  }
  function closeRaw() { player.querySelector("video")?.pause(); if (player.open) player.close(); }
  player.addEventListener("close", () => { player.querySelector("video")?.pause(); opener?.focus(); });
  player.addEventListener("cancel", (e) => { e.preventDefault(); closeRaw(); });
  player.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeRaw(); } });
  player.addEventListener("click", (e) => { if (e.target === player || e.target.closest("[data-close-raw]")) closeRaw(); });
  function inspect(r) {
    const { Explorer, trainItems, testItems } = window.CaMExplorer;
    const items = r.kind === "train" ? trainItems(r.dev, r.run) : testItems(r.run, r.run.episodes.filter((e) => e.task === r.task && e.maximum_rollouts === r.run.config.budget));
    const index = items.findIndex((x) => r.kind === "train" ? x.attempt === r.attempt : x.episode === r.episode);
    Explorer.open(items, Math.max(0, index));
  }
  async function openLog(run, code = false) {
    const { DevLog } = window.CaMExplorer;
    DevLog.current = code ? "code" : "timeline"; DevLog.selectedCodeAttempt = null;
    try { await DevLog.open(run.development[0].development_url, run.development[0], { ...run, name: run.display_name }); }
    catch (error) { DevLog.body.textContent = `Unable to load the development log: ${error.message}`; }
  }
  function renderDevelopment() {
    main.innerHTML = hero("HOW THE SOLUTIONS WERE BUILT", "Follow the development.", "Read the coding agent’s chronological log, inspect its exact prompt and code revisions, and watch the training attempts it used to verify its solution.") +
      `<div class="gallery-filters"><label>Task<select id="dev-task">${taskOptions(state.task)}</select></label><label>Experiment<select id="dev-setting">${settingOptions(state.setting)}</select></label></div><div class="development-note"><strong>Learning is separate from evaluation.</strong> Demo-only runs have no training rollouts. Interaction logs contain all assigned attempts; held-out videos belong to the Video gallery.</div><div class="development-grid">${runs.filter((r) => (!state.setting || r.experiment === state.setting) && (!state.task || r.tasks.includes(state.task))).sort((a, b) => a.tasks[0].localeCompare(b.tasks[0]) || a.experiment.localeCompare(b.experiment)).map((r) => {
        const d = r.development[0], t = d.task, count = d.rollouts.length;
        return `<article class="development-card"><div class="development-card-top"><span class="task-number">${short(t)}</span><span class="arena-badge demo">Frozen policy</span></div><p class="video-context">${esc(label(r.experiment))}</p><h2>${esc(pretty(t))}</h2><p class="development-counts"><span><b>${d.demonstrations}</b> demos</span><span><b>${count}</b> training attempts</span></p><button class="arena-button" data-log="${r.id}">Read development log →</button><div class="development-actions"><button data-code="${r.id}">Final code &amp; revisions</button><a href="${url("videos", { kind: count ? "train" : "demo", task: t, setting: r.experiment })}">${count ? "Training videos" : "Assigned demos"} ↗</a></div></article>`;
      }).join("")}</div>`;
    $("#dev-task").onchange = (e) => go("development", { task: e.target.value, setting: state.setting });
    $("#dev-setting").onchange = (e) => go("development", { task: state.task, setting: e.target.value });
    if (state.log) { const r = runs.find((r) => r.id === state.log); if (r) openLog(r, state.view === "code"); }
  }
  function legacyRollout(key) {
    const parts = key.split("|");
    const r = library.find((r) => parts[0] === "test" ? r.kind === "test" && r.run.id === parts[1] && r.task === parts[2] && String(r.episode) === parts[4] : r.kind === "train" && r.dev.development_url === parts[1] && String(r.attempt) === parts[2]);
    if (r) inspect(r);
  }
  main.addEventListener("click", (e) => {
    const b = e.target.closest("[data-watch],[data-inspect],[data-page],[data-log],[data-code],[data-group]");
    if (!b) return;
    if (b.dataset.watch != null) openRaw(visible[+b.dataset.watch], b);
    if (b.dataset.inspect != null) inspect(visible[+b.dataset.inspect]);
    if (b.dataset.group) go("videos", { ...selections(), task: b.dataset.group, p: 1 });
    if (b.dataset.page) { go("videos", { ...selections(), p: b.dataset.page }); window.scrollTo({ top: 180, behavior: "smooth" }); }
    if (b.dataset.log != null || b.dataset.code != null) {
      const id = b.dataset.log ?? b.dataset.code;
      history.replaceState(null, "", url("development", { setting: state.setting, task: state.task, log: id, ...(b.dataset.code != null ? { view: "code" } : {}) }));
      openLog(runs.find((r) => r.id === id), b.dataset.code != null);
    }
  });
  window.addEventListener("hashchange", route);
  window.CaMArena = {
    trainingGallery(log) {
      const rows = log.task?.rollouts || [];
      const run = runs.find((r) => r.id === log.run.id);
      if (!rows.length) { log.body.innerHTML = '<div class="arena-empty"><h2>No training rollouts</h2><p>This demo-only learner developed its policy from the assigned demonstrations.</p></div>'; return; }
      log.body.innerHTML = `<div class="video-gallery">${rows.map((r, i) => `<article class="video-card"><button class="video-thumbnail" data-log-watch="${i}" aria-label="Watch training attempt ${r.attempt}"><img loading="lazy" src="${esc(r.poster)}" alt="Recorded camera views"><span class="play-icon">▶</span>${badge({ ...r, kind: "train" })}</button><div class="video-card-body"><h2>Attempt R${String(r.attempt).padStart(3, "0")}</h2><div class="video-details"><span>Episode ${r.episode}</span><span>${fmt(r.steps)} steps</span></div><div class="video-card-actions"><button data-log-watch="${i}">Watch raw video ↗</button><button data-rollout="${i}">Decisions &amp; state</button></div></div></article>`).join("")}</div>`;
      log.body.onclick = (e) => {
        const b = e.target.closest('[data-log-watch]');
        if (b) openRaw({ ...rows[+b.dataset.logWatch], kind: 'train', task: log.task.task, dev: log.task, run, setting: run.experiment }, b);
      };
    },
  };
  fetch("data.json").then((r) => { if (!r.ok) throw Error("Archive data could not load"); return r.json(); }).then((value) => {
    data = value; archive = data.archive; if (!archive) throw Error("Archive presentation is unavailable");
    runs = data.runs.filter((r) => r.experiment); indexRecordings();
    const old = new URLSearchParams(location.search);
    if (old.has("task")) {
      const task = old.get("task").split("/").at(-1), run = runs.find((r) => r.id === old.get("run"));
      const args = { task, ...(run ? { setting: run.experiment } : {}) };
      history.replaceState(null, "", location.pathname + url(old.get("view") === "demos" ? "videos" : "development", { ...args, ...(old.get("view") === "demos" ? { kind: "demo" } : run && ["log", "code"].includes(old.get("view")) ? { log: run.id, view: old.get("view") } : {}) }));
    }
    route();
  }).catch((error) => { main.innerHTML = `<div class="arena-empty"><h1>Unable to load the archive</h1><p>${esc(error.message)}</p><button onclick="location.reload()">Retry</button></div>`; });
})();
