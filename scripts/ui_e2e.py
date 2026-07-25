from __future__ import annotations

from pathlib import Path

from playwright.sync_api import ConsoleMessage, sync_playwright


def main() -> None:
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    console_errors: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})

        def capture_console(message: ConsoleMessage) -> None:
            if message.type == "error":
                console_errors.append(message.text)

        page.on("console", capture_console)
        page.goto("http://127.0.0.1:8501")
        page.wait_for_load_state("networkidle")
        page.get_by_text("Term Trace", exact=False).first.wait_for()

        page.get_by_role("button", name="Find terms for review").click()
        page.get_by_text("Ranked dictionary candidates", exact=True).wait_for(timeout=15_000)
        metformin_cell = page.get_by_text("metformin", exact=True).first
        metformin_cell.wait_for(state="attached", timeout=15_000)
        assert metformin_cell.text_content() == "metformin"
        assert page.get_by_text("AUTO-COMMIT · LOCKED", exact=True).is_visible()
        assert page.get_by_text("dose adjacent", exact=False).is_visible()

        page.get_by_text("Keep original", exact=True).click()
        page.get_by_role("button", name="Record decision").click()
        page.get_by_text("Decision recorded", exact=False).wait_for(timeout=15_000)
        page.screenshot(path=str(output_dir / "ui-reviewed.png"), full_page=True)

        sidebar_toggle = page.get_by_role("button").filter(has_text="keyboard_double_arrow_right")
        sidebar_toggle.click()
        search = page.get_by_role("textbox", name="Term, brand, or abbreviation")
        search.fill("Coumadin")
        search.press("Enter")
        page.get_by_text("warfarin", exact=False).first.wait_for(timeout=15_000)

        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(500)
        page.screenshot(path=str(output_dir / "ui-mobile.png"), full_page=True)
        viewport_ok = page.evaluate(
            "document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"
        )
        assert viewport_ok, "The reviewer UI overflows horizontally on a 390px viewport"
        assert not console_errors, f"Browser console errors: {console_errors}"
        print(
            {
                "candidate": "metformin",
                "decision": "keep_original",
                "dictionary_search": "warfarin",
                "mobile_no_overflow": viewport_ok,
                "console_errors": len(console_errors),
            }
        )
        browser.close()


if __name__ == "__main__":
    main()
