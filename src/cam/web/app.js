"use strict";
let DATA = { runs: [], datasets: [] };
const $ = (id) => document.getElementById(id);
const esc = (x) =>
  String(x ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const num = (x) =>
  x == null
    ? "—"
    : Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 });
const rate = (x) => (x == null ? "—" : (100 * x).toFixed(2) + "%");
const table = (headers, rows) =>
  rows.length
    ? '<div class="table-wrap"><table><thead><tr>' +
      headers.map((x) => "<th>" + esc(x) + "</th>").join("") +
      "</tr></thead><tbody>" +
      rows
        .map(
          (row) =>
            "<tr>" + row.map((x) => "<td>" + x + "</td>").join("") + "</tr>",
        )
        .join("") +
      "</tbody></table></div>"
    : '<div class="empty">No records yet.</div>';
const link = (url, label) =>
  url
    ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)}</a>`
    : "—";
const shape = (spec) =>
  (spec.shape || []).map((d) => (d == null ? "N" : d)).join("×") || "scalar";
const code = (x) => `<code>${esc(x)}</code>`;
function interfaceBlock(t, open) {
  const i = t.interface;
  const r = t.recorded;
  if (!i && !r) return "";
  let html = `<details class="interface"${open ? " open" : ""}><summary>Robot state &amp; action space</summary>`;
  if (i) {
    html +=
      "<h4>Policy observation per frame</h4>" +
      table(
        ["Sensor", "dtype", "Shape", "Meaning"],
        Object.entries(i.sensors).map(([n, sp]) => [
          code(n),
          esc(sp.dtype),
          esc(shape(sp)),
          esc(sp.description),
        ]),
      );
    const a = i.action;
    html +=
      `<h4>Action space: ${code("robot.step(" + a.names.join(", ") + ")")}</h4>` +
      `<p class="muted">${esc(a.description)}</p>` +
      table(
        ["Component", "Low", "High", "Note"],
        a.names.map((n, k) => [
          code(n),
          a.low[k] == null ? "−∞" : esc(a.low[k]),
          a.high[k] == null ? "∞" : esc(a.high[k]),
          a.ignored.includes(n) ? "ignored (sent as 0)" : "",
        ]),
      );
  }
  if (r) {
    const rows = (fields, perFrame) =>
      Object.entries(fields || {}).map(([n, sp]) => [
        code(n),
        esc(sp.dtype),
        esc(shape(sp)) + (perFrame ? " per frame" : " per episode"),
        esc(sp.description),
      ]);
    html +=
      "<h4>Recorded demonstration fields</h4>" +
      table(
        ["Field", "dtype", "Shape", "Meaning"],
        [...rows(r.frame, true), ...rows(r.static, false)],
      );
  }
  return html + "</details>";
}
const selected = () =>
  DATA.runs.filter(
    (r) =>
      (!$("benchmark").value || r.benchmark === $("benchmark").value) &&
      ($("run").value ? r.id === $("run").value : !r.superseded),
  );
const settingLabel = (r) => r.learning_setting || r.config.mode;
const label = (r) => (r.superseded ? "Historical · " : "") + r.name + " · " + settingLabel(r) + " · " + r.config.coding.model;
const runSelectionLabel = () => DATA.runs.some((r) => r.superseded) ? "Current runs" : "All runs";
function options(id, values, first) {
  const previous = $(id).value;
  $(id).innerHTML =
    '<option value="">' +
    esc(first) +
    "</option>" +
    values
      .map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`)
      .join("");
  $(id).value = values.some(([v]) => v === previous) ? previous : "";
}
function plot(id, runs, key, title) {
  const series = runs
    .map((r) => ({
      name: label(r),
      points: r.settings
        .filter((s) => s.complete && s[key] != null)
        .map((s) => [s.budget, key === "success_rate" ? 100 * s[key] : s[key]]),
    }))
    .filter((s) => s.points.length);
  if (!series.length) {
    $(id).innerHTML =
      '<div class="empty">Points appear when each evaluation is complete.</div>';
    return;
  }
  const W = 1000,
    H = 360,
    L = 65,
    R = 20,
    T = 24,
    B = 55,
    colors = ["#3267cf", "#c15468", "#29987b", "#9265c9", "#e09734"];
  const xmax = Math.max(1, ...series.flatMap((s) => s.points.map((p) => p[0]))),
    ymax =
      key === "success_rate"
        ? 100
        : Math.max(1, ...series.flatMap((s) => s.points.map((p) => p[1]))) *
          1.1;
  const x = (v) => L + (v / xmax) * (W - L - R),
    y = (v) => H - B - (v / ymax) * (H - T - B);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">`;
  for (let i = 0; i <= 4; i++) {
    const v = (ymax * i) / 4;
    svg += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#e4e9f1"/><text x="${L - 10}" y="${y(v) + 5}" text-anchor="end" fill="#728198" font-size="12">${num(v)}</text>`;
  }
  [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))]
    .sort((a, b) => a - b)
    .forEach(
      (v) =>
        (svg += `<text x="${x(v)}" y="${H - B + 25}" text-anchor="middle" fill="#728198" font-size="12">R${v}</text>`),
    );
  series.forEach((s, i) => {
    const c = colors[i % colors.length];
    svg += `<polyline points="${s.points.map((p) => `${x(p[0])},${y(p[1])}`).join(" ")}" fill="none" stroke="${c}" stroke-width="2.5"/>`;
    s.points.forEach(
      (p) =>
        (svg += `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="5" fill="${c}"><title>${esc(s.name)}: R${p[0]} = ${num(p[1])}</title></circle>`),
    );
  });
  svg += `<text x="${W / 2}" y="${H - 8}" text-anchor="middle" fill="#728198" font-size="12">Maximum training interactions</text></svg>`;
  $(id).innerHTML =
    svg +
    '<div class="legend">' +
    series
      .map(
        (s, i) =>
          `<span style="--series:${colors[i % colors.length]}">${esc(s.name)}</span>`,
      )
      .join("") +
    "</div>";
  const a = document.createElement("a");
  a.textContent = "Download SVG";
  a.className = "button";
  a.download = id + ".svg";
  a.href = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
  $(id).append(a);
}
function render() {
  const runs = selected();
  $("updated").textContent =
    "Updated " + new Date(DATA.updated * 1000).toLocaleString();
  // A site built only from datasets is a demonstration browser: show it first.
  const demosOnly = !DATA.runs.length && DATA.datasets.length > 0;
  for (const id of ["results", "success", "efficiency", "tasks", "behavior", "development", "rollouts"])
    $(id).hidden = demosOnly;
  $("run").closest("label").hidden = demosOnly;
  $("notice").innerHTML = runs.length
    ? runs.some((r) => r.superseded)
      ? `<div class="note">Historical run. ${runs.filter((r) => r.superseded).map((r) =>
          esc(r.superseded_reason) + (r.replacement_run_id != null
            ? " " + link(`?run=${encodeURIComponent(r.replacement_run_id)}#results`, "Open current run") : "")
        ).join(" ")}</div>`
      : DATA.runs.some((r) => r.superseded)
        ? '<div class="note">Showing current runs. Earlier development logs and scored rollouts remain available under Historical entries in the Run selector.</div>'
        : ""
    : demosOnly
      ? `<div class="note">Demonstration browser: ${DATA.datasets.length} extracted dataset(s) across ${new Set(DATA.datasets.map((d) => d.benchmark)).size} benchmark(s). No runs are linked to this site.</div>`
      : '<div class="note">Fresh CaM workspace. No historical results are imported. Extract raw training data, prepare a run, then develop and evaluate policies.</div>';
  if (DATA.site_note) $("notice").innerHTML += `<div class="note">${esc(DATA.site_note)}</div>`;
  $("result-table").innerHTML = table(
    [
      "Run / model",
      "Mode",
      "Demos/task",
      "Budget",
      "Progress",
      "Success",
      "Development cost",
      "Downloads",
    ],
    runs.flatMap((r) =>
      r.settings.map((s) => [
        esc(label(r)),
        esc(settingLabel(r)),
        esc([...new Set(r.development.map((t) => t.demonstrations))].join("/")),
        "R" + s.budget,
        `${s.scored}/${s.expected}${s.infrastructure_errors ? " · " + s.infrastructure_errors + " infrastructure errors" : ""}`,
        s.complete
          ? `<strong class="good">${rate(s.success_rate)}</strong> (${s.successes}/${s.expected})`
          : s.scored
            ? `<span class="partial">${rate(s.observed_success_rate)} partial</span> (${s.successes}/${s.scored} scored)`
            : "Not evaluated",
        r.cost.reported_cost_usd == null
          ? "Not reported"
          : `$${num(r.cost.reported_cost_usd)}${r.cost.complete ? "" : " known subtotal"} · once per run`,
        link(r.downloads.episodes, "CSV") +
          " · " +
          link(r.downloads.summary, "JSON"),
      ]),
    ),
  );
  plot("success-plot", runs, "success_rate", "Exact success rate (%)");
  $("success-plot").innerHTML += (DATA.figures || [])
    .map((p) => link(p, p.split("/").pop()))
    .join(" · ");
  plot(
    "efficiency-plot",
    runs,
    "mean_success_steps",
    "Mean steps on successful episodes",
  );
  const names = new Set(runs.map((r) => r.name));
  $("paired").innerHTML = table(
    [
      "Left setting",
      "Right setting",
      "Matched successes",
      "Left mean steps",
      "Right mean steps",
    ],
    (DATA.paired_comparisons || [])
      .filter(
        (p) =>
          names.has(p.left.split(" / R")[0]) &&
          names.has(p.right.split(" / R")[0]),
      )
      .map((p) => [
        esc(p.left),
        esc(p.right),
        num(p.matched_successes),
        num(p.left_mean_steps),
        num(p.right_mean_steps),
      ]),
  );
  $("task-table").innerHTML = table(
    [
      "Run",
      "Task",
      "Budget",
      "Scored / expected",
      "Success",
      "Successful-episode steps",
    ],
    runs.flatMap((r) =>
      r.settings.flatMap((s) =>
        Object.entries(s.tasks).map(([task, m]) => [
          esc(r.name),
          esc(task),
          "R" + s.budget,
          `${m.scored}/${m.expected}`,
          rate(m.complete ? m.success_rate : m.observed_success_rate) +
            (m.complete ? "" : " partial"),
          num(m.mean_success_steps),
        ]),
      ),
    ),
  );
  $("difficulty-table").innerHTML = table(
    ["Run", "Budget", "Difficulty", "Success", "Scored / expected"],
    runs.flatMap((r) =>
      r.settings.flatMap((s) =>
        Object.entries(s.difficulty).map(([d, m]) => [
          esc(r.name),
          "R" + s.budget,
          d,
          rate(m.complete ? m.success_rate : m.observed_success_rate) +
            (m.complete ? "" : " partial"),
          `${m.scored}/${m.expected}`,
        ]),
      ),
    ),
  );
  $("behavior-table").innerHTML = table(
    ["Run", "Budget", "Mean decisions", "Outcome counts"],
    runs.flatMap((r) =>
      r.settings.map((s) => [
        esc(r.name),
        "R" + s.budget,
        num(s.mean_decisions),
        esc(JSON.stringify(s.statuses)),
      ]),
    ),
  );
  const batch = DATA.batch;
  if ($("batch-status")) $("batch-status").innerHTML = batch
    ? `<p><strong>${batch.tasks.length} selected tasks · ${esc(batch.workers)} coding slots</strong></p><details><summary>Batch queue and data preparation</summary>${table(["Benchmark", "Task", "Batch status", "Policy", "Details"], batch.tasks.map((t) => [esc(t.benchmark), esc(t.task), esc(t.status), t.policy_frozen ? "Frozen" : "Not frozen", esc(t.error || "")]))}</details>`
    : "";
  $("development-table").innerHTML = table(
    [
      "Run",
      "Task",
      "Status",
      "Attempts",
      "Visited / allowed seeds",
      "First success",
      "Cost",
      "Logs",
    ],
    runs.flatMap((r) =>
      r.development.map((t) => [
        esc(r.name),
        esc(t.task),
        `<span class="badge">${esc(t.status)}</span>`,
        `${t.attempts.length}/${r.config.budget}`,
        `${t.train_episodes_visited}/${t.train_episodes_allowed}`,
        t.first_success_attempt == null
          ? "Not observed"
          : "R" + t.first_success_attempt,
        t.usage.reported_cost_usd == null
          ? "Not reported"
          : "$" + num(t.usage.reported_cost_usd),
        `<button data-dev="${esc(t.development_url)}">Development log</button>` +
          (t.rollouts && t.rollouts.length
            ? ` <button data-train="${esc(t.development_url)}">Training rollouts (${t.rollouts.length})</button>`
            : ""),
      ]),
    ),
  );
  window.CaMStudio.update(DATA, runs, $("benchmark").value, $("run").value);
  route();
  options(
    "rollout-task",
    [...new Set(runs.flatMap((r) => r.tasks))].map((x) => [x, x]),
    "All tasks",
  );
  options(
    "rollout-budget",
    [
      ...new Set(runs.flatMap((r) => r.settings.map((s) => String(s.budget)))),
    ].map((x) => [x, "R" + x]),
    "All budgets",
  );
  renderRollouts();
  renderDemoGallery();
}
const DEMO_THUMBS = 3; // thumbnails per task card before "+N more"
let DEMO_KEY = null;
function renderDemoGallery(force) {
  // Gallery: one compact card per task with lazy video thumbnails (front | wrist).
  const search = ($("demo-search")?.value || "").trim().toLowerCase();
  const datasets = DATA.datasets.filter(
    (d) => !$("benchmark").value || d.benchmark === $("benchmark").value,
  );
  const key = JSON.stringify([search, datasets.map((d) => [d.name, d.tasks.length, d.in_progress, d.tasks.reduce((n, t) => n + t.episodes.length, 0)])]);
  if (!force && key === DEMO_KEY) return; // periodic refreshes must not restart playing videos
  DEMO_KEY = key;
  const open = new Set([...document.querySelectorAll("#demo-list details.dataset[open]")].map((d) => d.dataset.name));
  const card = (t) => {
    const goals = new Set(t.episodes.map((e) => e.goal || ""));
    const goal = goals.size === 1 ? [...goals][0] : "";
    const tiles = t.episodes
      .map(
        (e, i) =>
          `<figure class="demo-thumb${i >= DEMO_THUMBS ? " extra" : ""}" data-big="${esc(e.video || "")}" title="Click to enlarge">` +
          (e.video
            ? `<video muted loop playsinline preload="none" data-src="${esc(e.video)}"></video>`
            : '<div class="novideo">no video</div>') +
          `<figcaption>ep ${esc(e.episode)} · ${esc(e.frames)} f${e.difficulty ? " · " + esc(e.difficulty) : ""}</figcaption><span class="cam-tag">front</span></figure>`,
      )
      .join("");
    const more = t.episodes.length - DEMO_THUMBS;
    const total = t.episodes_total ?? t.episodes.length;
    return (
      `<article class="demo-card"><header><h4 title="${esc(t.task)}">${esc(t.task)}</h4>` +
      `<span class="muted">${t.episodes.length < total ? `${t.episodes.length} of ${total}` : total} demos · ${link(t.h5, "H5")}</span></header>` +
      (goal ? `<p class="demo-goal" title="${esc(goal)}">${esc(goal)}</p>` : "") +
      `<div class="demo-thumbs">${tiles}</div>` +
      (more > 0 ? `<button class="demo-more" data-more>+${more} more</button>` : "") +
      `</article>`
    );
  };
  $("demo-list").innerHTML =
    (DATA.datasets.length
      ? `<div class="demo-tools"><input id="demo-search" type="search" placeholder="Filter tasks…" value="${esc(search)}"></div>`
      : "") +
    datasets
      .map((d) => {
        const tasks = d.tasks.filter((t) => !search || t.task.toLowerCase().includes(search));
        const shown = d.shown_per_task ? ` · showing ${d.shown_per_task} per task` : "";
        const isOpen = open.has(d.name) || (!DATA.runs.length && datasets.length === 1) || (search && tasks.length);
        return (
          `<details class="dataset" data-name="${esc(d.name)}"${isOpen ? " open" : ""}><summary>${esc(d.benchmark)} · ${esc(d.name)} · ${d.depth ? "RGB-D" : "RGB"} · ${d.tasks.length} task(s), ${d.tasks.reduce((n, t) => n + (t.episodes_total ?? t.episodes.length), 0)} demos${shown}${d.in_progress ? " · build in progress" : ""}</summary>` +
          `<p class="muted">Selection: ${esc(d.selection && d.selection.count)} per task · ${esc(d.selection && d.selection.rule)}</p>` +
          (d.tasks[0] ? interfaceBlock(d.tasks[0], false) : "") +
          `<div class="demo-gallery">${tasks.map(card).join("") || '<div class="empty">No matching tasks.</div>'}</div></details>`
        );
      })
      .join("") ||
    (window.CaMStudio?.rows?.some((r) => r.task?.reference_demos?.length) ? "" : '<div class="empty">No extracted datasets linked yet.</div>');
  setupDemoThumbs();
}
let DEMO_OBSERVER = null;
function setupDemoThumbs() {
  DEMO_OBSERVER?.disconnect();
  DEMO_OBSERVER = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const v = entry.target;
        if (entry.isIntersecting && !v.src) {
          v.preload = "metadata";
          v.src = v.dataset.src + "#t=0.5"; // show a frame without playing
        } else if (!entry.isIntersecting && !v.paused) v.pause();
      }
    },
    { rootMargin: "300px" },
  );
  document.querySelectorAll("#demo-list video[data-src]").forEach((v) => DEMO_OBSERVER.observe(v));
}
document.addEventListener("input", (e) => {
  if (e.target.id === "demo-search") {
    const at = e.target.selectionStart;
    renderDemoGallery(true);
    const box = $("demo-search");
    box.focus();
    box.setSelectionRange(at, at);
  }
});
document.addEventListener("mouseover", (e) => {
  const v = e.target.closest?.(".demo-thumb")?.querySelector("video");
  if (v && v.src) v.play().catch(() => {});
});
document.addEventListener("mouseout", (e) => {
  const fig = e.target.closest?.(".demo-thumb");
  if (fig && !fig.contains(e.relatedTarget)) fig.querySelector("video")?.pause();
});
document.addEventListener("click", (e) => {
  const more = e.target.closest("[data-more]");
  if (more) {
    more.closest(".demo-card").classList.add("expanded");
    more.remove();
    setupDemoThumbs();
    return;
  }
  const fig = e.target.closest(".demo-thumb");
  if (fig && fig.dataset.big) {
    // Full-size overlay with both cameras (front | wrist); Esc or a click outside closes it.
    let overlay = document.getElementById("demo-overlay");
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = "demo-overlay";
      document.body.append(overlay);
    }
    const card = fig.closest(".demo-card");
    overlay.innerHTML =
      `<div class="demo-big"><div class="demo-big-head"><strong>${esc(card.querySelector("h4").textContent)}</strong> · ${esc(fig.querySelector("figcaption").textContent)} <span class="muted">front | wrist</span><button data-close-demo>✕ Close</button></div>` +
      (card.querySelector(".demo-goal") ? `<p class="demo-big-goal">${esc(card.querySelector(".demo-goal").textContent)}</p>` : "") +
      `<video controls autoplay loop playsinline src="${esc(fig.dataset.big)}"></video></div>`;
    overlay.hidden = false;
    return;
  }
  if (e.target.closest("[data-close-demo]") || e.target.id === "demo-overlay") {
    const overlay = document.getElementById("demo-overlay");
    overlay.querySelector("video")?.pause();
    overlay.hidden = true;
  }
});
document.addEventListener("keydown", (e) => {
  const overlay = document.getElementById("demo-overlay");
  if (e.key === "Escape" && overlay && !overlay.hidden) {
    overlay.querySelector("video")?.pause();
    overlay.hidden = true;
  }
});
let ROLLOUT_GROUPS = [];
function renderRollouts() {
  // One card per run × task × budget; each episode is a chip coloured by outcome.
  const groups = new Map();
  for (const r of selected())
    for (const e of r.episodes) {
      if ($("rollout-task").value && e.task !== $("rollout-task").value) continue;
      if ($("rollout-budget").value && String(e.maximum_rollouts) !== $("rollout-budget").value) continue;
      const key = r.id + "|" + e.task + "|" + e.maximum_rollouts;
      if (!groups.has(key)) groups.set(key, { run: r, task: e.task, budget: e.maximum_rollouts, episodes: [] });
      groups.get(key).episodes.push(e);
    }
  ROLLOUT_GROUPS = [...groups.values()];
  const cls = (s) => (s === "success" ? "ok" : s === "infrastructure_error" ? "infra" : s ? "bad" : "none");
  $("rollout-table").innerHTML =
    ROLLOUT_GROUPS.map((g, gi) => {
      const ok = g.episodes.filter((e) => e.status === "success").length;
      const statuses = {};
      for (const e of g.episodes) statuses[e.status] = (statuses[e.status] || 0) + 1;
      return (
        `<div class="rollout-group"><div class="rg-head"><strong>${esc(g.task)}</strong> <span class="muted">· ${esc(label(g.run))} · R${g.budget}</span>` +
        `<span class="rg-score"><b>${ok}/${g.episodes.length}</b> success · ${esc(Object.entries(statuses).map(([k, v]) => k + " " + v).join(", "))}</span></div>` +
        `<div class="chips">${g.episodes
          .map((e) => `<a class="chip ${cls(e.status)}" href="#rollout=${encodeURIComponent(`test|${g.run.id}|${e.task}|${e.maximum_rollouts}|${e.episode}`)}" target="_blank" rel="noopener" title="Open episode ${esc(e.episode)} · ${esc(e.status)} · ${esc(e.steps)} steps">${esc(e.episode)}</a>`)
          .join("")}</div></div>`
      );
    }).join("") || '<div class="empty">No evaluation episodes for this selection.</div>';
}
document.addEventListener("click", (event) => {
  const b = event.target.closest("[data-dev],[data-video],[data-explore],[data-train]");
  if (!b) return;
  const { Explorer, DevLog, trainItems, testItems } = window.CaMExplorer;
  const owner = (url) => {
    for (const r of DATA.runs) for (const t of r.development) if (t.development_url === url) return [t, r];
    return [null, null];
  };
  if (b.dataset.dev || b.dataset.train) {
    const [task, run] = owner(b.dataset.dev || b.dataset.train);
    if (task) window.open(window.CaMStudio.url({ task, run, name: task.task, benchmark: run.benchmark }, task.rollouts.at(-1)?.attempt, b.dataset.dev ? "log" : null), "_blank", "noopener");
  }
  if (b.dataset.explore) {
    const [gi, i] = b.dataset.explore.split(":").map(Number);
    const g = ROLLOUT_GROUPS[gi];
    Explorer.open(testItems(g.run, g.episodes), i);
  }
  if (b.dataset.video) {
    const player = b.closest("#demonstrations") ? $("demo-player") : $("player");
    player.innerHTML =
      `<video controls autoplay playsinline src="${esc(b.dataset.video)}"></video>`;
    player.scrollIntoView({ behavior: "smooth" });
  }
});
$("benchmark").addEventListener("change", () => {
  options(
    "run",
    DATA.runs
      .filter(
        (r) => !$("benchmark").value || r.benchmark === $("benchmark").value,
      )
      .map((r) => [r.id, label(r)]),
    runSelectionLabel(),
  );
  render();
});
$("run").addEventListener("change", render);
$("rollout-task").addEventListener("change", renderRollouts);
$("rollout-budget").addEventListener("change", renderRollouts);
async function refresh() {
  DATA = await fetch("data.json", { cache: "no-store" }).then((r) => {
    if (!r.ok) throw Error("Unable to load experiment data");
    return r.json();
  });
  options(
    "benchmark",
    [...new Set([...DATA.runs, ...DATA.datasets].map((r) => r.benchmark))].map(
      (x) => [x, x],
    ),
    "All benchmarks",
  );
  options(
    "run",
    DATA.runs.map((r) => [r.id, label(r)]),
    runSelectionLabel(),
  );
  if (!refresh.initialSelectionApplied) {
    const requested = new URLSearchParams(location.search).get("run");
    if (DATA.runs.some((r) => r.id === requested)) $("run").value = requested;
    refresh.initialSelectionApplied = true;
  }
  render();
}
function openDeepLink() {
  // #rollout=test|run|task|budget|episode or #rollout=train|development_url|attempt
  if (!location.hash.startsWith("#rollout=")) return;
  const key = decodeURIComponent(location.hash.slice(9));
  const { Explorer, trainItems, testItems } = window.CaMExplorer;
  const parts = key.split("|");
  if (parts[0] === "test") {
    const run = DATA.runs.find((r) => r.id === parts[1]);
    if (!run) return;
    if (run.superseded && $("run").value !== run.id) {
      $("run").value = run.id;
      render();
    }
    const episodes = run.episodes.filter((e) => e.task === parts[2] && String(e.maximum_rollouts) === parts[3]);
    const i = episodes.findIndex((e) => String(e.episode) === parts[4]);
    if (i >= 0) Explorer.open(testItems(run, episodes), i);
  } else if (parts[0] === "train") {
    for (const run of DATA.runs)
      for (const task of run.development)
        if (task.development_url === parts[1]) {
          const i = (task.rollouts || []).findIndex((r) => String(r.attempt) === parts[2]);
          if (i >= 0) location.replace(window.CaMStudio.url({ task, run, name: task.task, benchmark: run.benchmark }, Number(parts[2])));
        }
  }
}
function route() {
  const hash = location.hash;
  const demo = hash === "#demonstrations" || (!DATA.runs.length && DATA.datasets.length > 0 && !hash);
  const evaluation = ["#results", "#success", "#efficiency", "#tasks", "#behavior", "#rollouts"].includes(hash) || hash.startsWith("#rollout=test");
  $("development").hidden = demo || evaluation;
  $("demonstrations").hidden = !demo;
  $("evaluation-view").hidden = !evaluation;
  document.querySelector("h1").textContent = demo ? "Demonstration library" : evaluation ? "Held-out evaluation" : new URLSearchParams(location.search).has("task") ? "Rollout replay" : "Development studio";
  document.querySelectorAll("nav a").forEach((a) => {
    const current = a.hash === (demo ? "#demonstrations" : evaluation ? "#results" : "#development");
    a.classList.toggle("active", current);
    if (current) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  if (demo || evaluation) window.CaMExplorer.Explorer.video?.pause();
}
window.addEventListener("hashchange", () => { route(); openDeepLink(); });
refresh()
  .then(openDeepLink)
  .catch((e) => ($("notice").textContent = e.message));
setInterval(() => refresh().catch(() => {}), 30000);
