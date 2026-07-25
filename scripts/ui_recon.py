from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    output = Path("artifacts/ui-initial.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.goto("http://127.0.0.1:8501")
        page.wait_for_load_state("networkidle")
        page.get_by_text("Term Trace", exact=False).first.wait_for()
        page.screenshot(path=str(output), full_page=True)
        print({"title": page.title(), "buttons": page.get_by_role("button").all_inner_texts()})
        browser.close()


if __name__ == "__main__":
    main()
