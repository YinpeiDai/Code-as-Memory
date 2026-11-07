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
const selected = () =>
  DATA.runs.filter(
    (r) =>
      (!$("benchmark").value || r.benchmark === $("benchmark").value) &&
      (!$("run").value || r.id === $("run").value),
  );
const label = (r) => r.name + " · " + r.config.coding.model;
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
  $("notice").innerHTML = runs.length
    ? ""
    : '<div class="note">Fresh CaM workspace. No historical results are imported. Extract raw training data, prepare a run, then develop and evaluate policies.</div>';
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
        esc(r.config.mode),
        esc([...new Set(r.development.map((t) => t.demonstrations))].join("/")),
        "R" + s.budget,
        `${s.scored}/${s.expected}${s.infrastructure_errors ? " · " + s.infrastructure_errors + " infrastructure errors" : ""}`,
        s.complete
          ? `<strong class="good">${rate(s.success_rate)}</strong> (${s.successes}/${s.expected})`
          : `<span class="partial">${rate(s.observed_success_rate)} partial</span> (${s.successes}/${s.scored} scored)`,
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
        `<button data-dev="${esc(t.development_url)}">Open development</button>`,
      ]),
    ),
  );
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
  $("demo-list").innerHTML =
    DATA.datasets
      .filter(
        (d) => !$("benchmark").value || d.benchmark === $("benchmark").value,
      )
      .map(
        (d) =>
          `<details><summary>${esc(d.name)} · ${d.depth ? "RGB-D" : "RGB"}</summary>` +
          d.tasks
            .map(
              (t) =>
                `<h3>${esc(t.task)} ${link(t.h5, "H5 download")}</h3>` +
                table(
                  [
                    "Episode",
                    "Difficulty",
                    "Frames",
                    "Native video frames",
                    "Rollout",
                  ],
                  t.episodes.map((e) => [
                    e.episode,
                    esc(e.difficulty),
                    e.frames,
                    e.demo_frames,
                    e.video
                      ? `<button data-video="${esc(e.video)}">Play</button>`
                      : "Not recorded",
                  ]),
                ),
            )
            .join("") +
          "</details>",
      )
      .join("") || '<div class="empty">No extracted datasets linked yet.</div>';
}
function renderRollouts() {
  const rows = selected().flatMap((r) =>
    r.episodes
      .filter(
        (e) =>
          (!$("rollout-task").value || e.task === $("rollout-task").value) &&
          (!$("rollout-budget").value ||
            String(e.maximum_rollouts) === $("rollout-budget").value),
      )
      .map((e) => [
        esc(r.name),
        esc(e.task),
        e.episode,
        "R" + e.maximum_rollouts,
        esc(e.status),
        e.steps,
        e.video ? `<button data-video="${esc(e.video)}">Play</button>` : "—",
        link(e.observations, "H5") +
          " · " +
          link(e.actions, "Actions") +
          " · " +
          link(e.decisions_log, "Decisions"),
      ]),
  );
  $("rollout-table").innerHTML = table(
    ["Run", "Task", "Episode", "Budget", "Outcome", "Steps", "Video", "Data"],
    rows,
  );
}
async function development(url) {
  const d = await fetch(url).then((r) => r.json());
  const detail = $("development-detail");
  detail.innerHTML =
    `<h3>${esc(d.task)}</h3><details><summary>Exact development prompt</summary><pre>${esc(d.prompt)}</pre></details><details><summary>Final code</summary>${Object.entries(
      d.final_files,
    )
      .map(([n, u]) => link(u, n))
      .join("<br>")}</details>` +
    d.messages
      .map(
        (m) =>
          `<details><summary>${esc(m.session)} · ${esc(m.type)}</summary><pre>${esc(m.text)}</pre></details>`,
      )
      .join("") +
    "<h3>Tool calls</h3>" +
    d.calls
      .map(
        (c) =>
          `<details><summary>${esc(c.id)} · ${esc(c.tool)}${c.is_error ? " · error" : ""}</summary><pre>${esc(JSON.stringify(c.arguments, null, 2))}</pre><pre>${esc(c.text)}</pre>${link(c.stdout, "Full stdout")} ${link(c.stderr, "Full stderr")}</details>`,
      )
      .join("") +
    "<h3>Code revisions</h3>" +
    d.revisions
      .map(
        (r) =>
          `<details><summary>#${r.id} · ${esc(r.reason)} · after ${r.attempts} attempts</summary>${link(r.diff, "Diff")}<br>${Object.entries(
            r.files,
          )
            .map(([n, u]) => link(u, n))
            .join("<br>")}</details>`,
      )
      .join("");
  const task = DATA.runs
    .flatMap((r) => r.development)
    .find((t) => t.development_url === url);
  detail.innerHTML +=
    "<h3>Training rollouts</h3>" +
    table(
      ["Attempt", "Episode", "Outcome", "Steps", "Video", "Data"],
      (task?.rollouts || []).map((e) => [
        e.attempt,
        e.episode,
        esc(e.status),
        e.steps,
        e.video ? `<button data-video="${esc(e.video)}">Play</button>` : "—",
        link(e.observations, "H5") + " · " + link(e.actions, "Actions"),
      ]),
    );
  detail.scrollIntoView({ behavior: "smooth" });
}
document.addEventListener("click", (event) => {
  const b = event.target.closest("[data-dev],[data-video]");
  if (!b) return;
  if (b.dataset.dev) development(b.dataset.dev).catch((e) => alert(e.message));
  if (b.dataset.video) {
    $("player").innerHTML =
      `<video controls autoplay playsinline src="${esc(b.dataset.video)}"></video>`;
    $("player").scrollIntoView({ behavior: "smooth" });
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
    "All runs",
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
    "All runs",
  );
  render();
}
refresh().catch((e) => ($("notice").textContent = e.message));
setInterval(() => refresh().catch(() => {}), 30000);
