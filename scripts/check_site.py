"""Browser checks for the visual rollout library and independent replay pages."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("site", type=Path)
parser.add_argument("--screenshots", type=Path, required=True)
parser.add_argument("--url", help="Check an existing server instead of launching a temporary one")
args = parser.parse_args()
args.screenshots.mkdir(parents=True, exist_ok=True)
server = None
try:
    if args.url:
        address = args.url.rstrip("/") + "/"
    else:
        server = subprocess.Popen(
            [sys.executable, "-m", "cam.cli", "serve", str(args.site), "--port", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        address = server.stdout.readline().strip().split("CaM website: ")[-1]
    data = json.load(urlopen(address + "data.json"))
    choices = [(r, t, a) for r in data["runs"] for t in r["development"] for a in t["rollouts"]
               if a.get("video") and a.get("trace") and a.get("policy_files")]
    chosen = next((x for x in choices if x[2]["status"] != "success"), choices[0] if choices else None)
    errors = []
    checks = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1040})
        context.on("page", lambda page: page.on("pageerror", lambda error: errors.append(str(error))))
        page = context.new_page()
        page.goto(address, wait_until="networkidle")
        page.wait_for_function("document.querySelector('#updated').textContent.startsWith('Updated')")
        assert page.locator("nav a").count() == 3
        if choices:
            assert page.locator(".replay-card").count() >= len(data["runs"])
            page.wait_for_function("[...document.querySelectorAll('.replay-preview img')].some(i => i.complete && i.naturalWidth > 0)")
            assert page.locator("#evaluation-view").is_hidden()
            page.screenshot(path=str(args.screenshots / "library.png"), full_page=True)
            original_url = page.url
            with context.expect_page() as opened:
                page.locator("a.card-open").first.click()
            task_page = opened.value
            task_page.wait_for_selector("#explorer.embedded")
            assert page.url == original_url
            task_page.close()
            checks.append("whole task card opens a separate replay page")
            run, task, attempt = chosen
            query = urlencode({"task": run["benchmark"] + "/" + task["task"], "run": run["id"], "attempt": attempt["attempt"]})
            anchor = page.locator('a.mini-attempt[href="?' + query + '#development"]')
            anchor.scroll_into_view_if_needed()
            parent_url = page.url
            parent_scroll = page.evaluate("scrollY")
            with context.expect_page() as opened:
                anchor.click()
            replay = opened.value
            replay.wait_for_load_state("networkidle")
            replay.wait_for_function("document.querySelector('#explorer video')?.readyState >= 2 && !!window.CaMExplorer.Explorer.trace")
            assert page.url == parent_url and page.evaluate("scrollY") == parent_scroll
            assert replay.locator("#explorer.embedded").is_visible()
            assert attempt["status"] in replay.locator("#explorer .fv-title").inner_text()
            assert replay.evaluate("(() => { const c = document.querySelector('.ex-tile canvas'); const p = c.getContext('2d').getImageData(0,0,c.width,c.height).data; return new Set(Array.from(p).filter((_,i) => i % 4 !== 3)).size > 20; })()")
            trace = json.load(urlopen(address + attempt["trace"]))
            if trace["result"].get("goal"):
                assert replay.locator(".ex-goal p").inner_text() == trace["result"]["goal"]
                assert replay.locator(".ex-goal").bounding_box()["y"] < replay.locator(".ex-stage").bounding_box()["y"]
            if len(trace["decisions"]) > 1:
                decision = trace["decisions"][1]
                replay.locator('#explorer [data-stage="1"]').click()
                replay.wait_for_function("n => window.CaMExplorer.Explorer.frame === n", arg=decision["frame"])
                assert decision["tool"] in replay.locator(".ex-current").inner_text()
                for key, value in decision.get("labels", {}).items():
                    if key in ("phase", "stage", "state", "mode", "target"):
                        assert str(value) in replay.locator(".current-labels").inner_text()
                checks.append("stage labels and video seek match recorded decision frame")
            replay.locator('#explorer .ex-transport [data-x="play"]').click()
            before = replay.locator("#explorer video").evaluate("v => v.currentTime")
            replay.wait_for_function("t => document.querySelector('#explorer video').currentTime > t + 0.3", arg=before)
            replay.locator('#explorer .ex-transport [data-x="play"]').click()
            checks.append("recorded video renders and plays")
            paused = replay.locator("#explorer video").evaluate("v => v.currentTime")
            replay.evaluate("refresh()")
            assert abs(replay.locator("#explorer video").evaluate("v => v.currentTime") - paused) < 0.01
            checks.append("refresh preserves open replay position")
            replay.evaluate("scrollTo(0, 0)")
            replay.screenshot(path=str(args.screenshots / "replay.png"), full_page=True)
            with context.expect_page() as opened:
                replay.locator(".attempt-card").first.click()
            second = opened.value
            second.wait_for_load_state("networkidle")
            assert abs(replay.locator("#explorer video").evaluate("v => v.currentTime") - paused) < 0.01
            second.close()
            with context.expect_page() as opened:
                replay.locator('[data-studio-action="code"]').click()
            code = opened.value
            code.wait_for_selector("#devlog details.file")
            assert f'R{attempt["attempt"]} · {attempt["status"]}' in code.locator("[data-code-version] option:checked").inner_text()
            expected = urlopen(address + attempt["policy_files"]["tools/policy.py"]).read().decode()
            code.wait_for_function("document.querySelector('#devlog .dl-files').textContent.includes('def ')")
            assert code.locator('#devlog details.file:has(summary code:text-is("tools/policy.py")) pre').inner_text() == expected
            code.screenshot(path=str(args.screenshots / "executed_code.png"))
            code.close()
            with context.expect_page() as opened:
                replay.locator('[data-studio-action="log"]').click()
            log = opened.value
            log.wait_for_selector("#devlog .card.tool")
            log.screenshot(path=str(args.screenshots / "agent_log.png"))
            log.close()
            checks.append("attempt, code, and log cards open separate pages")
            replay.locator('[data-studio-action="expand"]').click()
            assert replay.locator("#explorer").is_visible() and replay.locator("#explorer.embedded").count() == 0
            replay.locator('#explorer [data-x="close"]').click()
            assert replay.locator("#explorer.embedded").is_visible()
            checks.append("expand and restore player")
            replay.set_viewport_size({"width": 390, "height": 844})
            replay.wait_for_function("document.documentElement.scrollWidth <= innerWidth")
            replay.screenshot(path=str(args.screenshots / "mobile_replay.png"), full_page=True)
            replay.close()
            page.locator("#gallery-failures").check()
            assert page.locator(".replay-card").count() > 0
            page.locator("#gallery-search").fill("no-matching-task-xyz")
            assert page.locator(".replay-card").count() == 0
            page.locator("#gallery-search").fill("")
            page.locator("#gallery-failures").uncheck()
            checks.append("task search and failure filters")
            # Every adapter supplies its own camera list and trace layout.
            for benchmark in sorted({r[0]["benchmark"] for r in choices}):
                candidates = [x for x in choices if x[0]["benchmark"] == benchmark]
                selected_run, selected_task, selected_attempt = next(
                    (x for x in candidates if any(d.get("preview_on_demand") for d in x[1].get("reference_demos", []))),
                    candidates[0],
                )
                params = urlencode({"task": benchmark + "/" + selected_task["task"], "run": selected_run["id"], "attempt": selected_attempt["attempt"]})
                sample = context.new_page()
                sample.goto(address + "?" + params + "#development", wait_until="networkidle")
                sample.wait_for_function("document.querySelector('#explorer video')?.readyState >= 2 && !!window.CaMExplorer.Explorer.trace")
                assert sample.evaluate("document.querySelectorAll('.ex-tile canvas').length === window.CaMExplorer.Explorer.trace.tiles.length")
                assert sample.locator(".ex-current").inner_text()
                sample.screenshot(path=str(args.screenshots / (benchmark + "_replay.png")))
                references = selected_task.get("reference_demos", [])
                if references:
                    original_url = sample.url
                    with context.expect_page() as opened:
                        sample.locator('[data-studio-action="demos"]').click()
                    demo = opened.value
                    demo.wait_for_function("document.querySelector('#studio-demo-view video')?.readyState >= 2", timeout=120000)
                    assert sample.url == original_url
                    assert demo.locator(".reference-goal p").inner_text() == references[0]["goal"]
                    assert demo.locator(".reference-goal").bounding_box()["y"] < demo.locator(".reference-video").bounding_box()["y"]
                    assert demo.locator("#studio-demo-view .reference-card").count() == len(references)
                    assert demo.locator(".reference-cameras span").count() == len(references[0]["cameras"])
                    demo.locator("#studio-demo-view video").evaluate("v => v.play()")
                    demo.wait_for_function("document.querySelector('#studio-demo-view video').currentTime > 0.4")
                    demo.locator("#studio-demo-view video").evaluate("v => v.pause()")
                    current = demo.locator("#studio-demo-view video").evaluate("v => v.currentTime")
                    demo.evaluate("refresh()")
                    assert abs(demo.locator("#studio-demo-view video").evaluate("v => v.currentTime") - current) < 0.01
                    demo.locator('[data-demo-step="1"]').click()
                    assert "Frame" in demo.locator(".reference-clock").inner_text()
                    demo.screenshot(path=str(args.screenshots / (benchmark + "_reference.png")))
                    if len(references) > 1:
                        with context.expect_page() as opened:
                            demo.locator("#studio-demo-view .reference-card").nth(1).click()
                        other = opened.value
                        other.wait_for_selector(".reference-caption")
                        assert f'episode {references[1]["episode"]}' in other.locator(".reference-caption").inner_text()
                        other.close()
                    demo.set_viewport_size({"width": 390, "height": 844})
                    assert demo.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    demo.screenshot(path=str(args.screenshots / (benchmark + "_reference_mobile.png")))
                    demo.close()
                sample.close()
            checks.append("camera playback and labels across all represented benchmarks")
            if any(t.get("reference_demos") for r in data["runs"] for t in r["development"]):
                checks.append("assigned demo playback, frame controls, independent tabs, refresh, and mobile layout")
                page.goto(address + "#demonstrations", wait_until="networkidle")
                assert page.locator("#assigned-demo-library .reference-card").count() > 0
                page.goto(address, wait_until="networkidle")
            waiting = page.locator(".replay-card").filter(has=page.locator(".preview-placeholder"))
            if waiting.count():
                with context.expect_page() as opened:
                    waiting.first.locator("a.card-open").click()
                pending = opened.value
                pending.wait_for_selector(".studio-empty")
                assert pending.locator(".studio-empty").inner_text()
                pending.close()
                checks.append("explicit empty state for tasks without rollouts")
        elif data["datasets"]:
            page.goto(address + "#demonstrations", wait_until="networkidle")
            assert page.locator("#demonstrations").is_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(address, wait_until="networkidle")
        page.wait_for_function("document.querySelector('#updated').textContent.startsWith('Updated')")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(args.screenshots / "mobile_library.png"), full_page=True)
        assert not errors, errors
        browser.close()
    registry = json.loads((args.site / ".artifacts.json").read_text())
    if registry:
        key = next(k for k, value in registry.items() if Path(value).stat().st_size >= 16)
        response = urlopen(Request(address + "artifact/" + key, headers={"Range": "bytes=0-15"}))
        assert response.status == 206 and len(response.read()) == 16
    for relative in (".sources.json", ".artifacts.json", ".demo_sources.json", "../.git/config", "artifact/not-registered", "demo-video/not-registered.mp4"):
        try:
            urlopen(address + relative)
            raise AssertionError("Private/unregistered URL was served: " + relative)
        except HTTPError as error:
            assert error.code == 404
    checks += ["desktop/mobile layout", "artifact ranges and private URL rejection"]
    result = {"passed": True, "checks": checks, "browser_errors": errors}
    (args.screenshots / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
finally:
    if server:
        server.terminate()
        server.wait(timeout=10)
