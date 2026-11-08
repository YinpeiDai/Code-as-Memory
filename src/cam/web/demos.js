"use strict";

// Reference recordings stay distinct from learner rollouts and their labels.
const Demos = {
  url(row, episode = null) {
    const url = new URL(Studio.url(row, null, "demos"), location.href);
    if (episode != null) url.searchParams.set("demo", episode);
    return url.search + url.hash;
  },
  card(row, demo) {
    return `<a class="reference-card" href="${X.esc(this.url(row, demo.episode))}" target="_blank" rel="noopener">
      <div class="reference-preview"><img src="${X.esc(demo.poster)}" alt="Training demo ${demo.number} camera preview" loading="lazy"><span class="preview-play">▶</span><span class="reference-badge">TRAINING DEMO</span></div>
      <div><strong>Demo ${String(demo.number).padStart(2, "0")} <span>↗</span></strong><small>Episode ${demo.episode} · ${demo.frames} recorded frames · ${demo.cameras.length} cameras</small></div></a>`;
  },
  references(row) {
    const demos = row?.task?.reference_demos || [];
    const box = document.getElementById("studio-reference-demos");
    box.innerHTML = demos.length ? `<div class="reference-heading"><div><h3>Training demos for reference</h3><p>Examples from the ${demos.length} demonstrations assigned to this learner.</p></div><a href="${X.esc(this.url(row))}" target="_blank" rel="noopener">View all ${demos.length} demos ↗</a></div><div class="reference-grid">${demos.slice(0, 3).map((demo) => this.card(row, demo)).join("")}</div>` : "";
  },
  library(rows) {
    document.getElementById("assigned-demo-library").innerHTML = rows.filter((row) => row.task?.reference_demos?.length).map((row) => {
      const demos = row.task.reference_demos, first = demos[0];
      return `<a class="reference-card" href="${X.esc(this.url(row))}" target="_blank" rel="noopener"><div class="reference-preview"><img src="${X.esc(first.poster)}" alt="${X.esc(row.name)} reference preview" loading="lazy"><span class="preview-play">▶</span><span class="reference-badge">${demos.length} TRAINING DEMOS</span></div><div><div class="eyebrow">${X.esc(Studio.bench(row.benchmark))}</div><strong>${X.esc(Studio.short(row.name))} ↗</strong><small>Assigned reference behavior</small></div></a>`;
    }).join("");
  },
  show(row) {
    const demos = row?.task?.reference_demos || [], box = document.getElementById("studio-demo-view");
    box.hidden = false;
    if (!demos.length) { box.innerHTML = '<div class="empty">No assigned demonstration videos are available for this run.</div>'; return; }
    const query = new URLSearchParams(location.search);
    const demo = query.has("demo") ? demos.find((d) => String(d.episode) === query.get("demo")) : demos[0];
    if (!demo) { box.innerHTML = '<div class="empty">That episode is not one of this run’s assigned demonstrations.</div>'; return; }
    document.title = `${Studio.short(row.name)} · Demo ${demo.number} · CaM`;
    const key = `${row.key}:${demo.episode}:${demo.video}`;
    if (key === this.loaded) return; // background refresh must not restart playback
    this.loaded = key;
    const rows = Math.ceil(demo.cameras.length / demo.columns);
    box.innerHTML = `<div class="reference-player"><div class="reference-caption"><div><span class="reference-label">TRAINING DEMONSTRATION</span><h3>Demo ${String(demo.number).padStart(2, "0")} <span>· episode ${demo.episode}</span></h3></div><span class="reference-clock">Loading video…</span></div>
      <div class="reference-goal"><span>Task goal</span><p>${X.esc(demo.goal || "No goal recorded for this demonstration.")}</p></div>
      <div class="reference-video" style="max-width:${Math.min(1100, Math.round(540 * demo.width / demo.height))}px"><video controls playsinline preload="auto" poster="${X.esc(demo.poster)}" src="${X.esc(demo.video)}"></video><div class="reference-cameras">${demo.cameras.map((name, i) => `<span style="left:calc(${100 * (i % demo.columns) / demo.columns}% + 10px);top:calc(${100 * Math.floor(i / demo.columns) / rows}% + 10px)">${X.esc(name.replace(/_rgb$/, "").replace(/_/g, " "))}</span>`).join("")}</div></div>
      <div class="reference-controls"><button data-demo-step="-1">−1 frame</button><button data-demo-step="1">+1 frame</button><label>Speed <select id="reference-speed">${[0.25, 0.5, 1, 2].map((s) => `<option value="${s}"${s === 1 ? " selected" : ""}>${s}×</option>`).join("")}</select></label><span class="reference-loading">Loading reference video…</span><a href="${X.esc(demo.video)}" download>Download video</a></div>
      <p class="reference-provenance">${row.benchmark === "robomemarena" ? "Recorded expert-action replay." : "Assigned training demonstration."} ${row.benchmark === "rmbench" ? "Playback shows recorded samples; it is not a real-time timing reference." : ""} No coding-agent decision labels are attached to this recording.</p></div>
      <div class="reference-heading"><div><h3>All ${demos.length} assigned demos</h3><p>Open another demonstration in a separate tab to compare behavior.</p></div></div><div class="reference-grid all-demos">${demos.map((d) => this.card(row, d)).join("")}</div>`;
    const video = box.querySelector("video"), clock = box.querySelector(".reference-clock"), loading = box.querySelector(".reference-loading");
    const update = () => { clock.textContent = `Frame ${Math.min(demo.frames - 1, Math.floor(video.currentTime * demo.fps + 1e-3))} / ${demo.frames - 1} · ${demo.cameras.length} cameras`; };
    video.addEventListener("timeupdate", update);
    video.addEventListener("seeked", update);
    video.addEventListener("loadeddata", () => { loading.hidden = true; update(); });
    video.addEventListener("error", () => { loading.textContent = "Video could not load. Refresh to retry."; });
    box.querySelectorAll("[data-demo-step]").forEach((button) => button.addEventListener("click", () => {
      video.pause();
      const frame = Math.floor(video.currentTime * demo.fps + 1e-3) + Number(button.dataset.demoStep);
      video.currentTime = (Math.max(0, Math.min(demo.frames - 1, frame)) + 0.5) / demo.fps;
      update();
    }));
    box.querySelector("#reference-speed").addEventListener("change", (e) => { video.playbackRate = Number(e.target.value); });
  },
};
window.CaMDemos = Demos;
