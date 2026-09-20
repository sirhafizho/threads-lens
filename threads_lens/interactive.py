"""Interactive mode — REPL for asking questions about collected Threads data."""

import asyncio
import json
import os
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .vision import PROVIDERS, get_api_key

console = Console()


def _get_llm(provider: str = "kimi") -> tuple[OpenAI, str]:
    cfg = PROVIDERS.get(provider, PROVIDERS["kimi"])
    return (
        OpenAI(api_key=get_api_key(), base_url=cfg["base_url"], timeout=120),
        cfg["model"],
    )


def _posts_to_context(posts: list[dict]) -> str:
    """Convert post dicts to structured context for the LLM."""
    lines = []
    for i, p in enumerate(posts, 1):
        lines.append(f"POST {i}: @{p.get('author_username', 'unknown')}")
        lines.append(f"  text: {p.get('post_text', 'N/A')[:400]}")
        lines.append(f"  topic: {p.get('topic_category', '?')}")
        lines.append(f"  sentiment: {p.get('sentiment', '?')}")

        if p.get("media_description"):
            lines.append(f"  media: {p.get('media_type', '?')} — {p['media_description'][:200]}")

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
        if flags:
            lines.append(f"  flags: {', '.join(flags)}")

        comments = p.get("comments", [])
        if comments:
            lines.append(f"  comments ({len(comments)}):")
            for c in comments:
                lines.append(f"    @{c.get('author', '?')}: {c.get('text', '')[:150]}")

        if p.get("comments_summary"):
            lines.append(f"  discussion: {p['comments_summary']}")
        if p.get("discussion_sentiment"):
            lines.append(f"  discussion_sentiment: {p['discussion_sentiment']}")
        if p.get("reply_insight"):
            lines.append(f"  reply_insight: {p['reply_insight']}")

        lines.append(f"  url: {p.get('_url', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are a Threads analytics expert. You have detailed data from real Threads posts including metrics, comments, and discussion analysis.

Answer questions about the data — be specific, cite post numbers, quote actual comments, and give actionable advice. When asked for content suggestions, base them on what actually worked in the data.

Commands the user can use:
- "#N" or "post N" — see details of a specific post
- "list" — see all posts numbered
- "suggest" — get content recommendations
- "quit" — exit

For anything else, just answer naturally using the data."""


def _show_index(posts: list[dict]):
    """Show numbered list of posts."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Author", style="green", width=20)
    table.add_column("Post", width=50)
    table.add_column("Likes", justify="right", width=8)
    table.add_column("Replies", justify="right", width=8)
    table.add_column("Topic", width=12)

    for i, p in enumerate(posts, 1):
        text = p.get("post_text_summary", p.get("post_text", ""))[:80]
        table.add_row(
            str(i),
            f"@{p.get('author_username', '?')}",
            text,
            str(p.get("likes", "—")),
            str(p.get("replies", "—")),
            p.get("topic_category", "?"),
        )
    console.print(table)


def _show_post_detail(post: dict, idx: int):
    """Show detailed info for a single post."""
    lines = [f"[bold]Post {idx}: @{post.get('author_username', 'unknown')}[/bold]\n"]
    lines.append(f"{post.get('post_text', 'N/A')}\n")

    if post.get("media_description"):
        lines.append(f"[dim]Media ({post.get('media_type', '?')}): {post['media_description']}[/dim]\n")

    metrics = []
    for k in ("views", "likes", "replies", "reposts", "shares"):
        v = post.get(k)
        metrics.append(f"{k.title()}: {v:,}" if v is not None else f"{k.title()}: —")
    lines.append(" | ".join(metrics))
    lines.append(f"\nTopic: {post.get('topic_category', '?')} | Sentiment: {post.get('sentiment', '?')} | Hook: {post.get('hook_type', '?')}")

    comments = post.get("comments", [])
    if comments:
        lines.append(f"\n[bold]Comments ({len(comments)}):[/bold]")
        for c in comments:
            likes = f" ({c.get('likes', 0):,} likes)" if c.get("likes") else ""
            lines.append(f"  @{c.get('author', '?')}{likes}: {c.get('text', '')[:150]}")

    if post.get("comments_summary"):
        lines.append(f"\n[bold]Discussion:[/bold] {post['comments_summary']}")
    if post.get("reply_insight"):
        lines.append(f"[bold]Audience:[/bold] {post['reply_insight']}")

    lines.append(f"\n[dim]{post.get('_url', '')}[/dim]")
    console.print(Panel("\n".join(lines), title=f"[cyan]Post {idx}[/cyan]"))


async def interactive_loop(posts: list[dict], session_name: str, provider: str = "kimi"):
    """Run the interactive REPL."""
    if not posts:
        console.print("[yellow]No posts to explore.[/yellow]")
        return

    context = _posts_to_context(posts)
    client, model = _get_llm(provider)
    conversation = []

    console.print(Panel(
        f"[bold]Interactive Mode[/bold] — {len(posts)} posts loaded\n\n"
        "Commands:\n"
        "  [cyan]#N[/cyan] or [cyan]post N[/cyan] — deep-dive into a post\n"
        "  [cyan]list[/cyan] — show all posts\n"
        "  [cyan]suggest[/cyan] — content ideas based on this data\n"
        "  [cyan]quit[/cyan] — exit\n\n"
        "Or just ask anything about the data.",
        title="[bold magenta]Threads Lens[/bold magenta]",
    ))

    _show_index(posts)

    while True:
        try:
            user_input = console.input("\n[bold cyan]threads>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        lower = user_input.lower().strip()

        if lower in ("quit", "exit", "q"):
            break

        if lower == "list":
            _show_index(posts)
            continue

        # Post number reference
        import re
        post_match = re.match(r"^(?:#|post\s+)(\d+)", lower)
        if post_match:
            idx = int(post_match.group(1)) - 1
            if 0 <= idx < len(posts):
                _show_post_detail(posts[idx], idx + 1)
            else:
                console.print(f"[red]Post {idx + 1} doesn't exist. Range: 1-{len(posts)}[/red]")
            continue

        # Content suggestion shortcut
        if lower == "suggest":
            user_input = (
                "Based on the collected data, suggest 5 post ideas I could write "
                "that would perform well. For each idea, include: the hook, "
                "suggested format (text/image/video), why it would work based on "
                "the data, and estimated engagement potential."
            )

        # Send to LLM
        conversation.append({"role": "user", "content": user_input})

        system = SYSTEM_PROMPT + f"\n\nCollected data ({len(posts)} posts):\n{context}"

        try:
            with console.status("Thinking..."):
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        *conversation[-6:],  # keep last 3 exchanges
                    ],
                    max_tokens=2000,
                    temperature=0.4,
                )
            answer = resp.choices[0].message.content
            conversation.append({"role": "assistant", "content": answer})
            console.print(Markdown(answer))
        except Exception as e:
            console.print(f"[red]LLM error: {e}[/red]")
            conversation.pop()  # remove failed message

    console.print("[dim]Session ended.[/dim]")
