"""Post collector — navigates Threads, screenshots posts, extracts data via vision."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .browser import ThreadsBrowser
from .hybrid import extract_post
from .vision import check_page_type

console = Console()
OUTPUT_DIR = Path.home() / ".threads-lens" / "sessions"


class CollectionResult:
    """Stores all extracted post data for a session."""

    def __init__(self, session_name: str):
        self.session_name = session_name
        self.posts: list[dict] = []
        self.errors: list[dict] = []
        self.started_at = datetime.now().isoformat()
        self.session_dir = OUTPUT_DIR / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.pending_issue: dict | None = None  # set when human-in-the-loop is needed

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

            # Media description
            media_desc = post.get("media_description")
            if media_desc and post.get("media_type") != "text":
                lines.append(f"- **Media ({post.get('media_type', '?')}):** {media_desc}")

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

            # Comments / discussion
            comments = post.get("comments", [])
            if comments:
                lines.append(f"- **Discussion ({len(comments)} comments captured):**")
                for c in comments:
                    author = c.get("author", "?")
                    text = c.get("text", "")[:150]
                    likes = c.get("likes")
                    likes_str = f" ({likes:,} likes)" if likes else ""
                    lines.append(f"  - @{author}{likes_str}: {text}")

            comments_summary = post.get("comments_summary")
            if comments_summary:
                lines.append(f"- **Discussion Summary:** {comments_summary}")

            discussion_sentiment = post.get("discussion_sentiment")
            if discussion_sentiment and discussion_sentiment != "none":
                lines.append(f"- **Discussion Sentiment:** {discussion_sentiment}")

            reply_insight = post.get("reply_insight")
            if reply_insight:
                lines.append(f"- **Audience Insight:** {reply_insight}")

            lines.append(f"- **Screenshot:** `{post.get('_screenshot', 'N/A')}`")
            lines.append("")

        if self.errors:
            lines.append("---")
            lines.append("## Errors")
            for e in self.errors:
                lines.append(f"- `{e['url']}` — {e['error']}")

        md_path.write_text("\n".join(lines))
        return str(md_path)


async def _check_issue(browser: ThreadsBrowser, session: CollectionResult, interactive: bool) -> bool:
    """Check for browser issues. Returns True if OK to continue."""
    if interactive:
        await browser.ensure_healthy()
        return True
    issue = await browser.check_and_report_issue()
    if issue:
        session.pending_issue = issue
        return False
    return True


async def _handle_post_issue(browser: ThreadsBrowser, session: CollectionResult, interactive: bool) -> bool:
    """Handle per-post issue. Returns True if OK to continue."""
    issue = await browser.check_for_issues()
    if not issue:
        return True
    if interactive:
        screenshot = await browser.screenshot(f"issue_{issue}")
        try:
            await browser.wait_for_human(issue, screenshot)
            return True
        except InterruptedError:
            return False
    session.pending_issue = await browser.check_and_report_issue()
    return False


async def _capture_post(browser: ThreadsBrowser, name: str) -> tuple[str, str]:
    """Get DOM text + screenshot. Returns (page_text, screenshot_path)."""
    page_text = await browser.page.inner_text("body")
    screenshot = await browser.screenshot_post(name)
    return page_text, screenshot


async def _analyze(page_text: str, screenshot: str, provider: str, label: str = "Analyzing...") -> dict:
    """Run hybrid extraction (DOM + minimal vision) with a spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"  {label}", total=None)
        data = await extract_post(page_text, screenshot, provider)
        progress.update(task, description="[green]Done!")
    return data


async def search_and_collect(
    browser: ThreadsBrowser,
    query: str,
    max_posts: int = 10,
    provider: str = "kimi",
    scroll_rounds: int = 3,
    interactive: bool = True,
) -> CollectionResult:
    """Search Threads for a topic and collect post data.

    Flow: search → get post links → visit each → screenshot → vision extract
    """
    session = CollectionResult(f"search_{query.replace(' ', '_')[:30]}_{int(time.time())}")

    console.print(f"\n[bold]Searching Threads for: '{query}'[/bold]")

    search_url = f"https://www.threads.com/search?q={query}&serp_type=default"
    if not await browser.goto(search_url, wait=4):
        console.print("[red]Failed to load search page[/red]")
        return session

    if not await _check_issue(browser, session, interactive):
        return session

    all_urls = []
    for round_num in range(scroll_rounds):
        urls = await browser.get_post_links()
        for u in urls:
            if u not in all_urls:
                all_urls.append(u)
        console.print(f"  Scroll {round_num + 1}: found {len(all_urls)} post URLs so far")
        if len(all_urls) >= max_posts:
            break
        await browser.scroll_down()

    if not all_urls:
        console.print("[yellow]No post links found. The page might need login.[/yellow]")
        screenshot = await browser.screenshot("no_results")
        page_type = await asyncio.to_thread(check_page_type, screenshot, provider)
        console.print(f"  Page type detected: {page_type}")
        if page_type in ("login", "popup"):
            if interactive:
                await browser.ensure_healthy()
                all_urls = await browser.get_post_links()
            else:
                session.pending_issue = await browser.check_and_report_issue() or {
                    "issue": "no_posts_found",
                    "screenshot": screenshot,
                    "message": "No posts found — may need login. Check the browser window.",
                }
                return session

    console.print(f"\n[bold]Collecting {min(len(all_urls), max_posts)} posts[/bold]")

    import random
    for i, url in enumerate(all_urls[:max_posts]):
        console.print(f"\n  [{i + 1}/{min(len(all_urls), max_posts)}] {url}")

        if i > 0:
            # Human-like pause between posts (1-4s)
            await asyncio.sleep(random.uniform(1.0, 4.0))

        if not await browser.goto(url, wait=3):
            session.add_error(url, "navigation_failed")
            continue

        if not await _handle_post_issue(browser, session, interactive):
            break

        page_text, screenshot = await _capture_post(browser, f"post_{i + 1}")
        data = await _analyze(page_text, screenshot, provider, "Extracting...")

        if "error" in data or data.get("parse_error"):
            console.print(f"    [yellow]Extraction issue: {data.get('error', 'parse error')}[/yellow]")
            session.add_error(url, data.get("error", "parse_error"))
            if "raw_response" in data:
                session.add_post(data, url, screenshot)
        else:
            comments = len(data.get("comments", []))
            console.print(f"    @{data.get('author_username', '?')} — {data.get('likes', '?')} likes, {data.get('replies', '?')} replies, {comments} comments")
            session.add_post(data, url, screenshot)

    md_path = session.save()
    console.print(f"\n[green]Session saved: {md_path}[/green]")
    return session


async def browse_trending(
    browser: ThreadsBrowser,
    max_posts: int = 15,
    provider: str = "kimi",
    interactive: bool = True,
) -> CollectionResult:
    """Browse the Threads feed and collect trending posts."""
    session = CollectionResult(f"trending_{int(time.time())}")

    console.print("\n[bold]Browsing Threads feed for trending posts[/bold]")

    if not await browser.goto("https://www.threads.com/", wait=4):
        console.print("[red]Failed to load Threads[/red]")
        return session

    if not await _check_issue(browser, session, interactive):
        return session

    all_urls = []
    for round_num in range(5):
        urls = await browser.get_post_links()
        for u in urls:
            if u not in all_urls:
                all_urls.append(u)
        console.print(f"  Scroll {round_num + 1}: {len(all_urls)} posts found")
        if len(all_urls) >= max_posts:
            break
        await browser.scroll_down()

    if not all_urls:
        console.print("[yellow]No posts found in feed.[/yellow]")
        if not interactive:
            session.pending_issue = await browser.check_and_report_issue() or {
                "issue": "no_posts_found",
                "message": "No posts found in feed — may need login.",
            }
        return session

    console.print(f"\n[bold]Analyzing {min(len(all_urls), max_posts)} posts[/bold]")

    for i, url in enumerate(all_urls[:max_posts]):
        console.print(f"\n  [{i + 1}/{min(len(all_urls), max_posts)}] {url}")

        if not await browser.goto(url, wait=3):
            session.add_error(url, "navigation_failed")
            continue

        if not await _handle_post_issue(browser, session, interactive):
            break

        page_text, screenshot = await _capture_post(browser, f"trending_{i + 1}")
        data = await _analyze(page_text, screenshot, provider)

        if "error" not in data:
            comments = len(data.get("comments", []))
            console.print(f"    @{data.get('author_username', '?')} — {data.get('likes', '?')} likes, {comments} comments")
        session.add_post(data, url, screenshot)

    md_path = session.save()
    console.print(f"\n[green]Session saved: {md_path}[/green]")
    return session


async def analyze_profile(
    browser: ThreadsBrowser,
    username: str,
    max_posts: int = 10,
    provider: str = "kimi",
    interactive: bool = True,
) -> CollectionResult:
    """Analyze a specific user's recent posts."""
    username = username.lstrip("@")
    session = CollectionResult(f"profile_{username}_{int(time.time())}")

    console.print(f"\n[bold]Analyzing profile: @{username}[/bold]")

    profile_url = f"https://www.threads.com/@{username}"
    if not await browser.goto(profile_url, wait=4):
        console.print("[red]Failed to load profile[/red]")
        return session

    if not await _check_issue(browser, session, interactive):
        return session

    all_urls = []
    for round_num in range(4):
        urls = await browser.get_post_links()
        for u in urls:
            if u not in all_urls:
                all_urls.append(u)
        console.print(f"  Scroll {round_num + 1}: {len(all_urls)} posts found")
        if len(all_urls) >= max_posts:
            break
        await browser.scroll_down()

    if not all_urls:
        console.print("[yellow]No posts found on this profile.[/yellow]")
        if not interactive:
            session.pending_issue = await browser.check_and_report_issue() or {
                "issue": "no_posts_found",
                "message": f"No posts found on @{username}'s profile — may need login.",
            }
        return session

    console.print(f"\n[bold]Analyzing {min(len(all_urls), max_posts)} posts[/bold]")

    for i, url in enumerate(all_urls[:max_posts]):
        console.print(f"\n  [{i + 1}/{min(len(all_urls), max_posts)}] {url}")

        if not await browser.goto(url, wait=3):
            session.add_error(url, "navigation_failed")
            continue

        if not await _handle_post_issue(browser, session, interactive):
            break

        page_text, screenshot = await _capture_post(browser, f"profile_{i + 1}")
        data = await _analyze(page_text, screenshot, provider)

        if "error" not in data:
            comments = len(data.get("comments", []))
            console.print(f"    {data.get('likes', '?')} likes, {data.get('replies', '?')} replies, {comments} comments")
        session.add_post(data, url, screenshot)

    md_path = session.save()
    console.print(f"\n[green]Session saved: {md_path}[/green]")
    return session
