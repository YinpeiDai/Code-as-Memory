"""Browser smoke check for an exported site; requires optional Playwright."""

import argparse
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen, Request
from urllib.error import HTTPError

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("site", type=Path)
parser.add_argument("--screenshots", type=Path, required=True)
args = parser.parse_args()
args.screenshots.mkdir(parents=True, exist_ok=True)
server = subprocess.Popen(
    [sys.executable, "-m", "cam.cli", "serve", str(args.site), "--port", "0"],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
)
try:
    address = server.stdout.readline().strip().split("CaM website: ")[-1]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(address, wait_until="networkidle")
        page.wait_for_function(
            "document.querySelector('#updated').textContent.startsWith('Updated')"
        )
        assert page.locator("h1").inner_text() == "Code-as-Memory experiments"
        assert page.locator("nav a").count() == 6
        page.screenshot(path=str(args.screenshots / "desktop.png"), full_page=True)
        if page.locator("[data-dev]").count():
            page.locator("[data-dev]").first.click()
            page.wait_for_selector("#development-detail h3")
            assert (
                "Exact development prompt"
                in page.locator("#development-detail").inner_text()
            )
            page.locator("#run").select_option(index=1)
            page.locator("#rollout-task").select_option(index=1)
            assert page.locator("#rollout-table tbody tr").count() > 0
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(address, wait_until="networkidle")
        page.wait_for_function(
            "document.querySelector('#updated').textContent.startsWith('Updated')"
        )
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )
        page.screenshot(path=str(args.screenshots / "mobile.png"), full_page=True)
        assert not errors, errors
        browser.close()
    import json

    registry = json.loads((args.site / ".artifacts.json").read_text())
    if registry:
        key = next(
            k for k, value in registry.items() if Path(value).stat().st_size >= 16
        )
        response = urlopen(
            Request(address + "artifact/" + key, headers={"Range": "bytes=0-15"})
        )
        assert response.status == 206 and len(response.read()) == 16
    for relative in (
        ".sources.json",
        ".artifacts.json",
        "../.git/config",
        "artifact/not-registered",
    ):
        try:
            urlopen(address + relative)
            raise AssertionError("Private/unregistered URL was served: " + relative)
        except HTTPError as error:
            assert error.code == 404
    print(
        "Desktop/mobile, development navigation, filters, artifact ranges and private-path checks passed."
    )
finally:
    server.terminate()
    server.wait(timeout=10)
