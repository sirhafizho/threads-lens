"""Playwright browser manager with persistent Chrome profile and human-in-the-loop."""

import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, Page, BrowserContext
from rich.console import Console

console = Console()

# Dedicated Chrome profile for Threads Lens — stays logged in between sessions
PROFILE_DIR = Path.home() / ".threads-lens" / "chrome-profile"
SCREENSHOT_DIR = Path(__file__).parent.parent / "output" / "screenshots"  # resolved at import


class ThreadsBrowser:
    """Manages a Playwright browser session acting as the user's Threads account."""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.playwright = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def start(self) -> Page:
        """Launch browser with persistent profile. Returns the main page."""
        self.playwright = sync_playwright().start()
        self.context = self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=self.headless,
            viewport={"width": 1280, "height": 900},
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        return self.page

    def stop(self):
        """Clean shutdown."""
        if self.context:
            self.context.close()
        if self.playwright:
            self.playwright.stop()

    def goto(self, url: str, wait: float = 3.0) -> bool:
        """Navigate to a URL. Returns True if page loaded."""
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(wait)
            return True
        except Exception as e:
            console.print(f"  [red]Navigation failed: {e}[/red]")
            return False

    def screenshot(self, name: str = "screenshot") -> str:
        """Take a screenshot of the current page. Returns file path."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = SCREENSHOT_DIR / f"{name}_{ts}.png"
        self.page.screenshot(path=str(path), full_page=False)
        return str(path)

    def screenshot_post(self, name: str = "post") -> str:
        """Screenshot the viewport area showing a post."""
        ts = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = SCREENSHOT_DIR / f"{name}_{ts}.png"
        self.page.screenshot(path=str(path), full_page=False)
        return str(path)

    def scroll_down(self, amount: int = 800):
        """Scroll the page down."""
        self.page.mouse.wheel(0, amount)
        time.sleep(1.5)

    def get_post_links(self) -> list[str]:
        """Extract all post URLs currently visible on the page."""
        links = self.page.query_selector_all('a[href*="/post/"]')
        urls = []
        for link in links:
            href = link.get_attribute("href")
            if href:
                if href.startswith("/"):
                    href = f"https://www.threads.com{href}"
                if href not in urls:
                    urls.append(href)
        return urls

    def check_for_issues(self) -> str | None:
        """Check page for common issues. Returns issue description or None."""
        url = self.page.url
        content = self.page.content().lower()

        if "log in" in content and ("sign up" in content or "create account" in content):
            # Check if it's blocking the content
            login_elements = self.page.query_selector_all(
                'div[role="dialog"], div[class*="login"], div[class*="Login"]'
            )
            if login_elements:
                return "login_wall"

        if "captcha" in content or "verify" in content.lower():
            return "captcha"

        if "something went wrong" in content or "error" in url:
            return "error_page"

        if self.page.query_selector_all('div[role="dialog"]'):
            return "popup_dialog"

        return None

    def wait_for_human(self, issue: str, screenshot_path: str):
        """Pause and wait for the human to fix an issue."""
        console.print()
        console.print(
            f"[bold red]⚠ ISSUE DETECTED: {issue}[/bold red]"
        )
        console.print(f"  Screenshot saved: {screenshot_path}")

        if issue == "login_wall":
            console.print(
                "  [yellow]Threads is asking for login.[/yellow]\n"
                "  The browser window should be visible — please log in manually.\n"
                "  Once you're logged in, press [bold]Enter[/bold] here to continue."
            )
        elif issue == "captcha":
            console.print(
                "  [yellow]CAPTCHA detected.[/yellow]\n"
                "  Please solve it in the browser window, then press [bold]Enter[/bold]."
            )
        elif issue == "popup_dialog":
            console.print(
                "  [yellow]A popup dialog is blocking the view.[/yellow]\n"
                "  Dismiss it in the browser, then press [bold]Enter[/bold]."
            )
        else:
            console.print(
                "  [yellow]Something looks off.[/yellow]\n"
                "  Check the browser window, fix any issues, then press [bold]Enter[/bold]."
            )

        try:
            input("  [Press Enter when ready] > ")
        except (EOFError, KeyboardInterrupt):
            raise InterruptedError("User aborted")

        time.sleep(1)

    def ensure_healthy(self, max_retries: int = 3):
        """Check for issues and let the human fix them. Retries up to max_retries."""
        for attempt in range(max_retries):
            issue = self.check_for_issues()
            if not issue:
                return True

            screenshot_path = self.screenshot(f"issue_{issue}")
            self.wait_for_human(issue, screenshot_path)

        console.print("[red]Max retries reached. Continuing anyway...[/red]")
        return False


def get_browser(headless: bool = False) -> ThreadsBrowser:
    """Create and start a ThreadsBrowser instance."""
    browser = ThreadsBrowser(headless=headless)
    browser.start()
    return browser
