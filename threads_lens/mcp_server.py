"""Threads Lens MCP Server — exposes Threads analytics tools via MCP for Devin/opencode."""

import asyncio
import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .analyzer import generate_report, quick_stats
from .browser import ThreadsBrowser, get_browser
from .collector import analyze_profile, browse_trending, search_and_collect
from .vision import analyze_screenshot

mcp = MCPServer(
    "threads-lens",
    description=(
        "AI agent that reads Threads posts via Playwright screenshots + LLM vision. "
        "No Threads API needed. Opens a real browser, screenshots posts, and extracts "
        "engagement metrics (likes, replies, reposts, views) using vision AI. "
        "Human-in-the-loop: if a login wall or popup appears, the tool returns the "
        "issue — tell the user to fix it in the browser window, then retry."
    ),
)

# Global browser — persists across tool calls so login state is maintained
_browser: ThreadsBrowser | None = None


def _get_browser() -> ThreadsBrowser:
    """Get or create the shared browser instance."""
    global _browser
    if _browser is None or _browser.page is None or _browser.page.is_closed():
        if _browser:
            try:
                _browser.stop()
            except Exception:
                pass
        _browser = get_browser(headless=False)
    return _browser


def _format_session_result(session) -> str:
    """Format a CollectionResult into a readable markdown response."""
    if session.pending_issue:
        issue = session.pending_issue
        parts = [
            f"⚠ **Issue detected: {issue['issue']}**",
            f"",
            issue["message"],
            f"",
            f"Screenshot: `{issue.get('screenshot', 'N/A')}`",
            f"URL: `{issue.get('url', 'N/A')}`",
        ]
        if session.posts:
            parts.append(f"\n**Partial results:** {len(session.posts)} posts collected before the issue.")
            parts.extend(_format_posts(session.posts))
        return "\n".join(parts)

    if not session.posts:
        return "No posts collected. The browser may need to be logged into Threads first."

    parts = [
        f"## Collected {len(session.posts)} posts",
        f"Session saved: `{session.session_dir}`",
        "",
    ]
    parts.extend(_format_posts(session.posts))

    # Quick stats
    stats = quick_stats(session.posts)
    if stats:
        parts.extend([
            "",
            "---",
            f"**Totals:** {stats['total_likes']:,} likes | {stats['total_replies']:,} replies | {stats['total_views']:,} views",
            f"**Averages:** {stats['avg_likes']} likes/post | {stats['avg_replies']} replies/post",
            f"**Controversial:** {stats['controversial_count']}/{stats['total_posts']}",
        ])
        if stats.get("top_post"):
            parts.append(f"**Top post:** @{stats['top_post']} ({stats['top_likes']:,} likes)")

    if session.errors:
        parts.append(f"\n**Errors:** {len(session.errors)} posts failed")

    return "\n".join(parts)


def _format_posts(posts: list[dict]) -> list[str]:
    """Format post list as markdown lines."""
    lines = []
    for i, p in enumerate(posts, 1):
        author = p.get("author_username", "unknown")
        text = p.get("post_text_summary", p.get("post_text", "N/A")[:100])
        metrics = []
        for k in ("views", "likes", "replies", "reposts", "shares"):
            v = p.get(k)
            if v is not None:
                metrics.append(f"{k}: {v:,}")
        flags = []
        if p.get("is_controversial"):
            flags.append("controversial")
        if p.get("has_hook"):
            flags.append(f"hook:{p.get('hook_type', '?')}")
        if p.get("sentiment"):
            flags.append(p["sentiment"])

        lines.append(f"**{i}. @{author}** — {text}")
        if metrics:
            lines.append(f"   {' | '.join(metrics)}")
        if flags:
            lines.append(f"   [{', '.join(flags)}]")
        lines.append(f"   `{p.get('_url', '')}`")
        lines.append("")
    return lines


@mcp.tool()
async def threads_status() -> str:
    """Check if the Threads browser is open and logged in.

    Returns browser status, current URL, and whether any issues are detected.
    Use this to check if the user needs to log in before running other tools.
    """
    global _browser
    if _browser is None:
        return "Browser not started. Call any threads_* tool to launch it, or call threads_open to open it now."

    if _browser.page is None or _browser.page.is_closed():
        return "Browser was closed. Call any threads_* tool to relaunch it."

    issue = _browser.check_and_report_issue()
    status = f"Browser open at: {_browser.page.url}"
    if issue:
        status += f"\n⚠ Issue: {issue['issue']} — {issue['message']}"
    else:
        status += "\nNo issues detected."
    return status


@mcp.tool()
async def threads_open() -> str:
    """Open the Threads browser window so the user can log in.

    Opens a visible browser at threads.com. The browser stays open — the user
    can log in, dismiss popups, or navigate. Once done, call threads_status to
    verify, then use other tools.
    """
    browser = _get_browser()
    ok = browser.goto("https://www.threads.com/", wait=4)
    if not ok:
        return "Failed to open Threads. Check that Chrome is installed."

    issue = browser.check_and_report_issue()
    if issue:
        return f"Browser opened at {browser.page.url}\n⚠ {issue['message']}"
    return f"Browser opened at {browser.page.url}\nReady — no issues detected."


@mcp.tool()
async def threads_research(query: str, max_posts: int = 10) -> str:
    """Research a topic on Threads — search, screenshot posts, extract engagement data.

    Searches Threads for the given topic, visits each post, screenshots it,
    and uses LLM vision to extract metrics (likes, replies, reposts, views, shares)
    plus topic classification, sentiment, and hook analysis.

    If a login wall or popup blocks the page, returns the issue — tell the user
    to fix it in the browser window, then call this tool again.

    Args:
        query: Topic to search for (e.g. "malaysian politics", "AI news", "viral")
        max_posts: Max posts to analyze (default 10, keep small for speed)
    """
    browser = _get_browser()
    session = await asyncio.to_thread(
        search_and_collect, browser, query, max_posts, "kimi", 3, False
    )
    return _format_session_result(session)


@mcp.tool()
async def threads_trending(max_posts: int = 10) -> str:
    """Browse Threads feed and analyze trending posts.

    Opens the user's Threads feed, scrolls to find posts, screenshots each one,
    and extracts engagement metrics via vision AI.

    If a login wall appears, returns the issue — tell the user to log in
    in the browser window, then call this tool again.

    Args:
        max_posts: Max posts to analyze (default 10)
    """
    browser = _get_browser()
    session = await asyncio.to_thread(
        browse_trending, browser, max_posts, "kimi", False
    )
    return _format_session_result(session)


@mcp.tool()
async def threads_profile(username: str, max_posts: int = 10) -> str:
    """Analyze a Threads user's recent posts and engagement.

    Visits the user's profile, collects their recent posts, and extracts
    engagement metrics. Great for understanding what content works for
    a specific account.

    Args:
        username: Threads username (with or without @)
        max_posts: Max posts to analyze (default 10)
    """
    browser = _get_browser()
    session = await asyncio.to_thread(
        analyze_profile, browser, username, max_posts, "kimi", False
    )
    return _format_session_result(session)


@mcp.tool()
async def threads_post(url: str) -> str:
    """Analyze a single Threads post by URL.

    Screenshots the post and extracts all visible engagement metrics
    via vision AI.

    Args:
        url: Full Threads post URL (e.g. https://www.threads.com/@user/post/ABC123)
    """
    browser = _get_browser()
    ok = browser.goto(url, wait=3)
    if not ok:
        return "Failed to load the post URL."

    issue = browser.check_and_report_issue()
    if issue:
        return f"⚠ {issue['message']}\nScreenshot: `{issue['screenshot']}`"

    screenshot = browser.screenshot_post("single_post")
    data = await asyncio.to_thread(analyze_screenshot, screenshot, "kimi")

    if "error" in data:
        return f"Vision extraction error: {data['error']}"

    lines = [
        f"## @{data.get('author_username', 'unknown')}",
        f"**Text:** {data.get('post_text', 'N/A')[:500]}",
        f"**Topic:** {data.get('topic_category', '?')} | **Sentiment:** {data.get('sentiment', '?')}",
        f"**Hook:** {data.get('hook_type', '?')}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k in ("views", "likes", "replies", "reposts", "shares"):
        v = data.get(k)
        lines.append(f"| {k.title()} | {v if v is not None else '—'} |")

    lines.append(f"\nScreenshot: `{screenshot}`")
    return "\n".join(lines)


@mcp.tool()
async def threads_report(session_dir: str) -> str:
    """Generate a virality analysis report from a previous collection session.

    Reads the data.md file from a session directory and generates a detailed
    report on what makes posts go viral — topics, hooks, engagement patterns.

    Args:
        session_dir: Path to a session directory (e.g. output/search_topic_1234567890)
    """
    data_file = Path(session_dir) / "data.md"
    if not data_file.exists():
        # Try relative to output dir
        data_file = Path(__file__).parent.parent / "output" / session_dir / "data.md"
    if not data_file.exists():
        return f"No data.md found in {session_dir}. Run a collection first."

    # Parse the markdown to extract post data (simplified)
    content = data_file.read_text()
    # For the report, we just pass the raw markdown to the LLM
    from .vision import PROVIDERS, get_api_key
    from openai import OpenAI

    cfg = PROVIDERS["kimi"]
    client = OpenAI(api_key=get_api_key(), base_url=cfg["base_url"], timeout=120)

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {
                "role": "system",
                "content": "You are a Threads virality analyst. Analyze this collected post data and produce insights on what makes posts go viral — topics, hooks, engagement patterns, and recommendations.",
            },
            {"role": "user", "content": content},
        ],
        max_tokens=4000,
        temperature=0.3,
    )
    return resp.choices[0].message.content


@mcp.tool()
async def threads_close() -> str:
    """Close the Threads browser window.

    Call this when done to clean up. The login session persists
    in the Chrome profile for next time.
    """
    global _browser
    if _browser is None:
        return "No browser to close."
    try:
        _browser.stop()
    except Exception:
        pass
    _browser = None
    return "Browser closed. Login session saved for next time."


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
