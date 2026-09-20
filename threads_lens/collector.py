"""Post collector — navigates Threads, screenshots posts, extracts data via vision."""

import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .browser import ThreadsBrowser
from .vision import analyze_screenshot, check_page_type

console = Console()
OUTPUT_DIR = Path(__file__).parent.parent / "output"


class CollectionResult:
    """Stores all extracted post data for a session."""

    def __init__(self, session_name: str):
        self.session_name = session_name
        self.posts: list[dict] = []
        self.errors: list[dict] = []
        self.started_at = datetime.now().isoformat()
        self.session_dir = OUTPUT_DIR / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def add_post(self, data: dict, url: str, screenshot: str):
        data["_url"] = url
        data["_screenshot"] = screenshot
        data["_collected_at"] = datetime.now().isoformat()
        self.posts.append(data)

    def add_error(self, url: str, error: str):
        self.errors.append({"url": url, "error": error, "time": datetime.now().isoformat()})

    def save(self) -> str:
        """Save session data as markdown."""
        md_path = self.session_dir / "data.md"
        lines = [
            f"# Threads Lens — {self.session_name}",
            f"**Collected:** {self.started_at}",
            f"**Posts:** {len(self.posts)} | **Errors:** {len(self.errors)}",
            "",
            "---",
            "",
        ]

        for i, post in enumerate(self.posts, 1):
            lines.append(f"## Post {i}: @{post.get('author_username', 'unknown')}")
            lines.append(f"- **URL:** {post.get('_url', 'N/A')}")
            lines.append(f"- **Text:** {post.get('post_text_summary', post.get('post_text', 'N/A')[:200])}")
            lines.append(f"- **Language:** {post.get('language', '?')}")
            lines.append(f"- **Topic:** {post.get('topic_category', '?')}")

            metrics = []
            for key in ("views", "likes", "replies", "reposts", "shares"):
                val = post.get(key)
                if val is not None:
                    metrics.append(f"**{key.title()}:** {val:,}")
            if metrics:
                lines.append(f"- **Metrics:** {' | '.join(metrics)}")

            flags = []
            if post.get("is_controversial"):
                flags.append("controversial")
            if post.get("has_hook"):
                flags.append(f"hook:{post.get('hook_type', '?')}")
            if post.get("has_media"):
                flags.append(f"media:{post.get('media_type', '?')}")
            if post.get("has_hashtags"):
                flags.append("hashtags")
            if post.get("sentiment"):
                flags.append(f"sentiment:{post['sentiment']}")
            if flags:
                lines.append(f"- **Flags:** {', '.join(flags)}")

            lines.append(f"- **Screenshot:** `{post.get('_screenshot', 'N/A')}`")
            lines.append("")

        if self.errors:
            lines.append("---")
            lines.append("## Errors")
            for e in self.errors:
                lines.append(f"- `{e['url']}` — {e['error']}")

        md_path.write_text("\n".join(lines))
        return str(md_path)


def search_and_collect(
    browser: ThreadsBrowser,
    query: str,
    max_posts: int = 10,
    provider: str = "kimi",
    scroll_rounds: int = 3,
) -> CollectionResult:
    """Search Threads for a topic and collect post data.

    Flow: search → get post links → visit each → screenshot → vision extract
    """
    session = CollectionResult(f"search_{query.replace(' ', '_')[:30]}_{int(time.time())}")

    console.print(f"\n[bold]Searching Threads for: '{query}'[/bold]")

    # Navigate to search
    search_url = f"https://www.threads.com/search?q={query}&serp_type=default"
    if not browser.goto(search_url, wait=4):
        console.print("[red]Failed to load search page[/red]")
        return session

    # Human-in-the-loop: check for issues
    browser.ensure_healthy()

    # Scroll to load more posts
    all_urls = []
    for round_num in range(scroll_rounds):
        urls = browser.get_post_links()
        for u in urls:
            if u not in all_urls:
                all_urls.append(u)
        console.print(f"  Scroll {round_num + 1}: found {len(all_urls)} post URLs so far")
        if len(all_urls) >= max_posts:
            break
        browser.scroll_down()

    if not all_urls:
        console.print("[yellow]No post links found. The page might need login.[/yellow]")
        screenshot = browser.screenshot("no_results")
        page_type = check_page_type(screenshot, provider)
        console.print(f"  Page type detected: {page_type}")
        if page_type in ("login", "popup"):
            browser.ensure_healthy()
            # Retry after user fixes
            all_urls = browser.get_post_links()

    console.print(f"\n[bold]Collecting {min(len(all_urls), max_posts)} posts[/bold]")

    # Visit each post and analyze
    for i, url in enumerate(all_urls[:max_posts]):
        console.print(f"\n  [{i + 1}/{min(len(all_urls), max_posts)}] {url}")

        if not browser.goto(url, wait=3):
            session.add_error(url, "navigation_failed")
            continue

        # Check for issues
        issue = browser.check_for_issues()
        if issue:
            screenshot = browser.screenshot(f"issue_{issue}")
            try:
                browser.wait_for_human(issue, screenshot)
            except InterruptedError:
                break

        # Screenshot the post
        screenshot = browser.screenshot_post(f"post_{i + 1}")

        # Vision extraction
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("  Analyzing with vision model...", total=None)
            data = analyze_screenshot(screenshot, provider)
            progress.update(task, description="[green]Done!")

        if "error" in data or data.get("parse_error"):
            console.print(f"    [yellow]Extraction issue: {data.get('error', 'parse error')}[/yellow]")
            session.add_error(url, data.get("error", "parse_error"))
            if "raw_response" in data:
                session.add_post(data, url, screenshot)
        else:
            likes = data.get("likes", "?")
            replies = data.get("replies", "?")
            author = data.get("author_username", "?")
            console.print(f"    @{author} — {likes} likes, {replies} replies")
            session.add_post(data, url, screenshot)

    # Save session data
    md_path = session.save()
    console.print(f"\n[green]Session saved: {md_path}[/green]")
    return session


def browse_trending(
    browser: ThreadsBrowser,
    max_posts: int = 15,
    provider: str = "kimi",
) -> CollectionResult:
    """Browse the Threads feed and collect trending posts."""
    session = CollectionResult(f"trending_{int(time.time())}")

    console.print("\n[bold]Browsing Threads feed for trending posts[/bold]")

    if not browser.goto("https://www.threads.com/", wait=4):
        console.print("[red]Failed to load Threads[/red]")
        return session

    browser.ensure_healthy()

    # Scroll and collect post URLs from feed
    all_urls = []
    for round_num in range(5):
        urls = browser.get_post_links()
        for u in urls:
            if u not in all_urls:
                all_urls.append(u)
        console.print(f"  Scroll {round_num + 1}: {len(all_urls)} posts found")
        if len(all_urls) >= max_posts:
            break
        browser.scroll_down()

    if not all_urls:
        console.print("[yellow]No posts found in feed.[/yellow]")
        return session

    console.print(f"\n[bold]Analyzing {min(len(all_urls), max_posts)} posts[/bold]")

    for i, url in enumerate(all_urls[:max_posts]):
        console.print(f"\n  [{i + 1}/{min(len(all_urls), max_posts)}] {url}")

        if not browser.goto(url, wait=3):
            session.add_error(url, "navigation_failed")
            continue

        issue = browser.check_for_issues()
        if issue:
            screenshot = browser.screenshot(f"issue_{issue}")
            try:
                browser.wait_for_human(issue, screenshot)
            except InterruptedError:
                break

        screenshot = browser.screenshot_post(f"trending_{i + 1}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("  Analyzing...", total=None)
            data = analyze_screenshot(screenshot, provider)
            progress.update(task, description="[green]Done!")

        if "error" not in data:
            console.print(f"    @{data.get('author_username', '?')} — {data.get('likes', '?')} likes")
        session.add_post(data, url, screenshot)

    md_path = session.save()
    console.print(f"\n[green]Session saved: {md_path}[/green]")
    return session


def analyze_profile(
    browser: ThreadsBrowser,
    username: str,
    max_posts: int = 10,
    provider: str = "kimi",
) -> CollectionResult:
    """Analyze a specific user's recent posts."""
    username = username.lstrip("@")
    session = CollectionResult(f"profile_{username}_{int(time.time())}")

    console.print(f"\n[bold]Analyzing profile: @{username}[/bold]")

    profile_url = f"https://www.threads.com/@{username}"
    if not browser.goto(profile_url, wait=4):
        console.print("[red]Failed to load profile[/red]")
        return session

    browser.ensure_healthy()

    # Scroll profile to load posts
    all_urls = []
    for round_num in range(4):
        urls = browser.get_post_links()
        for u in urls:
            if u not in all_urls:
                all_urls.append(u)
        console.print(f"  Scroll {round_num + 1}: {len(all_urls)} posts found")
        if len(all_urls) >= max_posts:
            break
        browser.scroll_down()

    if not all_urls:
        console.print("[yellow]No posts found on this profile.[/yellow]")
        return session

    console.print(f"\n[bold]Analyzing {min(len(all_urls), max_posts)} posts[/bold]")

    for i, url in enumerate(all_urls[:max_posts]):
        console.print(f"\n  [{i + 1}/{min(len(all_urls), max_posts)}] {url}")

        if not browser.goto(url, wait=3):
            session.add_error(url, "navigation_failed")
            continue

        issue = browser.check_for_issues()
        if issue:
            screenshot = browser.screenshot(f"issue_{issue}")
            try:
                browser.wait_for_human(issue, screenshot)
            except InterruptedError:
                break

        screenshot = browser.screenshot_post(f"profile_{i + 1}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("  Analyzing...", total=None)
            data = analyze_screenshot(screenshot, provider)
            progress.update(task, description="[green]Done!")

        if "error" not in data:
            console.print(f"    {data.get('likes', '?')} likes, {data.get('replies', '?')} replies")
        session.add_post(data, url, screenshot)

    md_path = session.save()
    console.print(f"\n[green]Session saved: {md_path}[/green]")
    return session
