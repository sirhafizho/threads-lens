"""Playwright browser manager with persistent Chrome profile and human-in-the-loop.

Uses async Playwright API — works both in CLI (via asyncio.run) and MCP contexts.
"""

import asyncio
import os
import time
from pathlib import Path

from playwright.async_api import async_playwright, Page, BrowserContext
from rich.console import Console

console = Console()

# Dedicated Chrome profile for Threads Lens — stays logged in between sessions
PROFILE_DIR = Path.home() / ".threads-lens" / "chrome-profile"
SCREENSHOT_DIR = Path.home() / ".threads-lens" / "screenshots"


class ThreadsBrowser:
    """Manages a Playwright browser session acting as the user's Threads account."""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.playwright = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    async def start(self) -> Page:
        """Launch browser with persistent profile. Returns the main page.

        Uses your system Google Chrome (channel='chrome') instead of bundled
        Chromium, so existing login sessions carry over — no need to log in again.
        """
        self.playwright = await async_playwright().start()
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=self.headless,
            channel="chrome",
            viewport={"width": 1280, "height": 900},
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        return self.page

    async def stop(self):
        """Clean shutdown."""
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()

    async def goto(self, url: str, wait: float = 3.0) -> bool:
        """Navigate to a URL. Returns True if page loaded."""
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Human-like jitter — don't be a metronome
            import random
            await asyncio.sleep(wait + random.uniform(0.5, 2.0))
            return True
        except Exception as e:
            console.print(f"  [red]Navigation failed: {e}[/red]")
            return False

    async def screenshot(self, name: str = "screenshot") -> str:
        """Take a screenshot of the current page. Returns file path."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = SCREENSHOT_DIR / f"{name}_{ts}.png"
        await self.page.screenshot(path=str(path), full_page=False)
        return str(path)

    async def screenshot_post(self, name: str = "post") -> str:
        """Screenshot the viewport area showing a post."""
        ts = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = SCREENSHOT_DIR / f"{name}_{ts}.png"
        await self.page.screenshot(path=str(path), full_page=False)
        return str(path)

    async def scroll_down(self, amount: int = 800):
        """Scroll the page down with human-like variance."""
        import random
        jitter = int(amount * random.uniform(0.7, 1.3))
        await self.page.mouse.wheel(0, jitter)
        await asyncio.sleep(random.uniform(0.8, 2.5))

    async def scroll_for_replies(self, scrolls: int = 2):
        """Scroll down to reveal the replies section on a post page."""
        for _ in range(scrolls):
            await self.scroll_down(600)

    async def get_post_links(self) -> list[str]:
        """Extract all post URLs currently visible on the page."""
        links = await self.page.query_selector_all('a[href*="/post/"]')
        urls = []
        for link in links:
            href = await link.get_attribute("href")
            if href:
                if href.startswith("/"):
                    href = f"https://www.threads.com{href}"
                if href not in urls:
                    urls.append(href)
        return urls

    async def check_for_issues(self) -> str | None:
        """Check page for common issues. Returns issue description or None."""
        url = self.page.url
        content = (await self.page.content()).lower()

        # Real CAPTCHA iframes — not just "verified" badges
        captcha_iframes = await self.page.query_selector_all(
            'iframe[src*="captcha"], iframe[src*="recaptcha"], iframe[src*="hcaptcha"]'
        )
        if captcha_iframes:
            return "captcha"

        # Login wall — must be a blocking dialog, not just "log in" text in footer
        if "log in" in content and ("sign up" in content or "create account" in content):
            login_dialogs = await self.page.query_selector_all(
                'div[role="dialog"] a[href*="login"], div[role="dialog"] button'
            )
            if login_dialogs:
                return "login_wall"

        if "something went wrong" in content and "try again" in content:
            return "error_page"

        return None

    async def check_and_report_issue(self) -> dict | None:
        """Non-blocking issue check for MCP mode. Returns issue info dict or None."""
        issue = await self.check_for_issues()
        if not issue:
            return None
        screenshot_path = await self.screenshot(f"issue_{issue}")
        return {
            "issue": issue,
            "screenshot": screenshot_path,
            "url": self.page.url,
            "message": self._issue_message(issue),
        }

    def _issue_message(self, issue: str) -> str:
        """Human-readable message for each issue type."""
        messages = {
            "login_wall": "Threads is asking for login. A browser window is open — please log in, then ask me to retry.",
            "captcha": "CAPTCHA detected. Please solve it in the browser window, then ask me to retry.",
            "popup_dialog": "A popup is blocking the view. Dismiss it in the browser, then ask me to retry.",
            "error_page": "The page failed to load. Check the browser window, then ask me to retry.",
        }
        return messages.get(issue, f"Issue detected: {issue}. Check the browser and ask me to retry.")

    async def wait_for_human(self, issue: str, screenshot_path: str):
        """Pause and wait for the human to fix an issue (CLI mode only)."""
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
            # Run input() in executor to not block the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: input("  [Press Enter when ready] > "))
        except (EOFError, KeyboardInterrupt):
            raise InterruptedError("User aborted")

        await asyncio.sleep(1)

    async def ensure_healthy(self, max_retries: int = 3):
        """Check for issues and let the human fix them. Retries up to max_retries."""
        for attempt in range(max_retries):
            issue = await self.check_for_issues()
            if not issue:
                return True

            screenshot_path = await self.screenshot(f"issue_{issue}")
            await self.wait_for_human(issue, screenshot_path)

        console.print("[red]Max retries reached. Continuing anyway...[/red]")
        return False


async def get_browser(headless: bool = False) -> ThreadsBrowser:
    """Create and start a ThreadsBrowser instance."""
    browser = ThreadsBrowser(headless=headless)
    await browser.start()
    return browser
