"""Virality analyzer — generates insights and reports from collected Threads data."""

import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .vision import PROVIDERS, get_api_key

console = Console()
OUTPUT_DIR = Path(__file__).parent.parent / "output"

ANALYSIS_PROMPT = """You are a Threads virality analyst. You've been given structured data extracted from real Threads posts via screenshot analysis.

Analyze the data and produce a report with these sections:

## Engagement Overview
- Total posts analyzed, average metrics
- Distribution of engagement (likes, replies, reposts, views)

## What Makes Posts Go Viral
- Patterns in high-engagement posts vs low-engagement
- Which topics get the most traction
- Hook patterns that work (questions, bold claims, how-tos)
- Media type impact (text vs image vs video)
- Sentiment analysis — does controversy help?

## Threads Algorithm Signals
- Reply-to-like ratio analysis (replies matter more than likes)
- Engagement velocity indicators
- What separates viral (>100 likes) from mid posts

## Content Recommendations
- What topics to post about this week
- Best posting format/patterns to copy
- Specific hooks that are working
- Topics to avoid (low engagement)

## Data Table
A summary table of all posts with their key metrics.

Be specific and data-driven. Reference actual numbers from the data."""


def _posts_to_context(posts: list[dict]) -> str:
    """Convert post dicts to a structured text block for the LLM."""
    lines = []
    for i, p in enumerate(posts, 1):
        lines.append(f"POST {i}:")
        lines.append(f"  author: @{p.get('author_username', 'unknown')}")
        lines.append(f"  text: {p.get('post_text', 'N/A')[:300]}")
        lines.append(f"  summary: {p.get('post_text_summary', 'N/A')}")
        lines.append(f"  language: {p.get('language', '?')}")
        lines.append(f"  topic: {p.get('topic_category', '?')}")
        lines.append(f"  sentiment: {p.get('sentiment', '?')}")

        metrics = []
        for k in ("views", "likes", "replies", "reposts", "shares"):
            v = p.get(k)
            metrics.append(f"{k}={v}" if v is not None else f"{k}=?")
        lines.append(f"  metrics: {', '.join(metrics)}")

        flags = []
        if p.get("is_controversial"):
            flags.append("controversial")
        if p.get("has_hook"):
            flags.append(f"hook={p.get('hook_type', '?')}")
        if p.get("has_media"):
            flags.append(f"media={p.get('media_type', '?')}")
        if p.get("has_hashtags"):
            flags.append("hashtags")
        if flags:
            lines.append(f"  flags: {', '.join(flags)}")

        lines.append(f"  url: {p.get('_url', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def generate_report(
    posts: list[dict],
    session_name: str = "analysis",
    provider: str = "kimi",
) -> str:
    """Generate a virality analysis report from collected posts. Returns markdown."""
    if not posts:
        return "No posts to analyze."

    cfg = PROVIDERS.get(provider, PROVIDERS["kimi"])
    client = OpenAI(
        api_key=get_api_key(),
        base_url=cfg["base_url"],
        timeout=120,
    )

    context = _posts_to_context(posts)

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": f"Here is data from {len(posts)} Threads posts:\n\n{context}",
            },
        ],
        max_tokens=4000,
        temperature=0.3,
    )

    report = resp.choices[0].message.content

    # Save report as markdown
    report_path = OUTPUT_DIR / f"report_{session_name}_{int(datetime.now().timestamp())}.md"
    header = f"""# Threads Lens Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Posts analyzed:** {len(posts)}
**Model:** {cfg['name']}

---

"""
    report_path.write_text(header + report)
    console.print(f"\n[green]Report saved: {report_path}[/green]")

    return report


def quick_stats(posts: list[dict]) -> dict:
    """Compute quick stats from collected posts without LLM."""
    if not posts:
        return {}

    stats = {
        "total_posts": len(posts),
        "total_likes": 0,
        "total_replies": 0,
        "total_reposts": 0,
        "total_views": 0,
        "posts_with_likes": 0,
        "posts_with_views": 0,
        "top_post": None,
        "top_likes": 0,
        "topics": {},
        "hooks": {},
        "controversial_count": 0,
        "avg_likes": 0,
        "avg_replies": 0,
    }

    for p in posts:
        likes = p.get("likes") or 0
        replies = p.get("replies") or 0
        reposts = p.get("reposts") or 0
        views = p.get("views") or 0

        stats["total_likes"] += likes
        stats["total_replies"] += replies
        stats["total_reposts"] += reposts
        stats["total_views"] += views

        if likes:
            stats["posts_with_likes"] += 1
        if views:
            stats["posts_with_views"] += 1

        if likes > stats["top_likes"]:
            stats["top_likes"] = likes
            stats["top_post"] = p.get("author_username", "?")

        topic = p.get("topic_category", "other")
        if topic not in stats["topics"]:
            stats["topics"][topic] = {"count": 0, "total_likes": 0}
        stats["topics"][topic]["count"] += 1
        stats["topics"][topic]["total_likes"] += likes

        hook = p.get("hook_type", "none")
        stats["hooks"][hook] = stats["hooks"].get(hook, 0) + 1

        if p.get("is_controversial"):
            stats["controversial_count"] += 1

    if stats["posts_with_likes"]:
        stats["avg_likes"] = round(stats["total_likes"] / stats["posts_with_likes"])
    if stats["total_posts"]:
        stats["avg_replies"] = round(stats["total_replies"] / stats["total_posts"])

    return stats


def display_stats(stats: dict):
    """Display quick stats in terminal."""
    if not stats:
        console.print("[yellow]No data to analyze[/yellow]")
        return

    from rich.table import Table

    table = Table(title="Quick Stats", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Posts analyzed", str(stats["total_posts"]))
    table.add_row("Total likes", f"{stats['total_likes']:,}")
    table.add_row("Total replies", f"{stats['total_replies']:,}")
    table.add_row("Total reposts", f"{stats['total_reposts']:,}")
    table.add_row("Total views", f"{stats['total_views']:,}")
    table.add_row("Avg likes/post", str(stats["avg_likes"]))
    table.add_row("Avg replies/post", str(stats["avg_replies"]))
    table.add_row("Top post", f"@{stats['top_post']} ({stats['top_likes']:,} likes)")
    table.add_row("Controversial", f"{stats['controversial_count']}/{stats['total_posts']}")

    console.print(table)

    if stats.get("topics"):
        console.print("\n[bold]Topics by engagement:[/bold]")
        sorted_topics = sorted(
            stats["topics"].items(),
            key=lambda x: x[1]["total_likes"],
            reverse=True,
        )
        for topic, data in sorted_topics:
            avg = data["total_likes"] / data["count"] if data["count"] else 0
            console.print(f"  {topic}: {data['count']} posts, avg {avg:.0f} likes")

    if stats.get("hooks"):
        console.print("\n[bold]Hook patterns:[/bold]")
        for hook, count in sorted(stats["hooks"].items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {hook}: {count} posts")
