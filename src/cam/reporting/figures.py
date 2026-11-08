"""Publication exports of observed values only; no error bands or invented repeats."""

from pathlib import Path
import hashlib
import json

from cam.io import read, write
from cam.config import learning_setting


def export_figures(reports, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "name": r["name"],
            "benchmark": r["benchmark"],
            "mode": r["config"]["mode"],
            "learning_setting": learning_setting(
                r["config"]["mode"], r["config"].get("both_protocol")
            ),
            "model": r["config"]["coding"]["model"],
            "settings": [s for s in r["settings"] if s["complete"]],
        }
        for r in reports
    ]
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    if (output / "manifest.json").exists() and read(output / "manifest.json")[
        "data_hash"
    ] == digest:
        return read(output / "manifest.json")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    for benchmark in sorted({r["benchmark"] for r in data}):
        selected = [r for r in data if r["benchmark"] == benchmark and r["settings"]]
        if not selected:
            continue
        for field, ylabel, scale in [
            ("success_rate", "Success rate (%)", 100),
            ("mean_success_steps", "Native execution steps (successful episodes)", 1),
            ("mean_decisions", "Decisions per scored episode", 1),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
            for run in selected:
                points = sorted(
                    (s["budget"], s[field] * scale)
                    for s in run["settings"]
                    if s[field] is not None
                )
                if points:
                    ax.plot(
                        *zip(*points),
                        marker="o",
                        linewidth=1.8,
                        label=f"{run['name']} · {run['learning_setting']} · {run['model']}",
                    )
            ax.set_xlabel("Maximum training interactions")
            ax.set_ylabel(ylabel)
            ax.set_xticks(
                sorted({s["budget"] for r in selected for s in r["settings"]})
            )
            if field == "success_rate":
                ax.set_ylim(0, 102)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=7, loc="best")
            for extension in ("svg", "png", "pdf"):
                path = output / f"{benchmark}_{field}.{extension}"
                fig.savefig(path, dpi=200)
                files.append(path.name)
            plt.close(fig)
    manifest = {
        "data_hash": digest,
        "files": files,
        "rule": "Exact complete-evaluation values; no confidence intervals",
    }
    write(output / "manifest.json", manifest)
    return manifest
