"""Threads Lens — AI agent that reads Threads via Playwright screenshots + LLM vision."""

import click
from rich.console import Console
from rich.panel import Panel

from .analyzer import display_stats, generate_report, quick_stats
from .browser import get_browser
from .collector import analyze_profile, browse_trending, search_and_collect

console = Console()

BANNER = """[bold cyan]
  _____ _                    _       _
 |_   _| |_  _ __ ___  __ _| |___  | |    ___ _ __  ___
   | | | ' \\| '__/ _ \\/ _` | / __| | |   / _ \\ '_ \\/ __|
   | | | | | | | |  __/ (_| | \\__ \\ | |__|  __/ | | \\__ \\
   |_| |_| |_|_|  \\___|\\__,_|_|___/ |_____\\___|_| |_|___/
[/bold cyan]
[dim]See what the algorithm sees. No API needed.[/dim]
"""


@click.group(invoke_without_command=True)
@click.option("--provider", "-p", default="kimi", help="Vision LLM: kimi, glm, gemini")
@click.option("--headless", is_flag=True, help="Run browser headless (no visible window)")
@click.pass_context
def main(ctx, provider, headless):
    """Threads Lens — analyze Threads posts via screenshots + LLM vision."""
    ctx.ensure_object(dict)
    ctx.obj["provider"] = provider
    ctx.obj["headless"] = headless

    if ctx.invoked_subcommand is None:
        console.print(BANNER)
        console.print(ctx.get_help())


@main.command()
@click.argument("query")
@click.option("--count", "-n", default=10, help="Max posts to analyze")
@click.option("--report", "-r", is_flag=True, help="Generate full report after collection")
@click.pass_context
def research(ctx, query, count, report):
    """Research a topic on Threads — search, screenshot, analyze."""
    console.print(BANNER)
    provider = ctx.obj["provider"]
    headless = ctx.obj["headless"]

    console.print(
        Panel(
            f"[bold]Topic:[/bold] {query}\n"
            f"[bold]Posts:[/bold] up to {count}\n"
            f"[bold]Vision:[/bold] {provider}\n"
            f"[bold]Browser:[/bold] {'headless' if headless else 'visible'}",
            title="[bold green]Research Mode[/bold green]",
        )
    )

    browser = get_browser(headless=headless)
    try:
        session = search_and_collect(browser, query, max_posts=count, provider=provider)

        # Quick stats
        stats = quick_stats(session.posts)
        display_stats(stats)

        # Full report
        if report and session.posts:
            console.print("\n[bold]Generating virality report...[/bold]")
            report_md = generate_report(session.posts, session.session_name, provider)
            console.print(Panel(Markdown(report_md[:3000]), title="Report Preview"))

        console.print(f"\n[bold green]Done![/bold green] Data saved to: {session.session_dir}")

    finally:
        browser.stop()


@main.command()
@click.option("--count", "-n", default=15, help="Max posts to analyze")
@click.option("--report", "-r", is_flag=True, help="Generate full report after collection")
@click.pass_context
def trending(ctx, count, report):
    """Browse Threads feed and analyze trending posts."""
    console.print(BANNER)
    provider = ctx.obj["provider"]
    headless = ctx.obj["headless"]

    console.print(
        Panel(
            f"[bold]Posts:[/bold] up to {count}\n"
            f"[bold]Vision:[/bold] {provider}\n"
            f"[bold]Browser:[/bold] {'headless' if headless else 'visible'}",
            title="[bold green]Trending Mode[/bold green]",
        )
    )

    browser = get_browser(headless=headless)
    try:
        session = browse_trending(browser, max_posts=count, provider=provider)

        stats = quick_stats(session.posts)
        display_stats(stats)

        if report and session.posts:
            console.print("\n[bold]Generating virality report...[/bold]")
            report_md = generate_report(session.posts, session.session_name, provider)
            console.print(Panel(Markdown(report_md[:3000]), title="Report Preview"))

        console.print(f"\n[bold green]Done![/bold green] Data saved to: {session.session_dir}")

    finally:
        browser.stop()


@main.command()
@click.argument("username")
@click.option("--count", "-n", default=10, help="Max posts to analyze")
@click.option("--report", "-r", is_flag=True, help="Generate full report after collection")
@click.pass_context
def profile(ctx, username, count, report):
    """Analyze a Threads user's posts — engagement patterns, best content."""
    console.print(BANNER)
    provider = ctx.obj["provider"]
    headless = ctx.obj["headless"]

    console.print(
        Panel(
            f"[bold]Profile:[/bold] @{username.lstrip('@')}\n"
            f"[bold]Posts:[/bold] up to {count}\n"
            f"[bold]Vision:[/bold] {provider}\n"
            f"[bold]Browser:[/bold] {'headless' if headless else 'visible'}",
            title="[bold green]Profile Analysis[/bold green]",
        )
    )

    browser = get_browser(headless=headless)
    try:
        session = analyze_profile(browser, username, max_posts=count, provider=provider)

        stats = quick_stats(session.posts)
        display_stats(stats)

        if report and session.posts:
            console.print("\n[bold]Generating profile report...[/bold]")
            report_md = generate_report(session.posts, session.session_name, provider)
            console.print(Panel(Markdown(report_md[:3000]), title="Report Preview"))

        console.print(f"\n[bold green]Done![/bold green] Data saved to: {session.session_dir}")

    finally:
        browser.stop()


@main.command()
@click.argument("url")
@click.option("--save", "-s", is_flag=True, help="Save result to session file")
@click.pass_context
def post(ctx, url, save):
    """Analyze a single Threads post URL."""
    console.print(BANNER)
    provider = ctx.obj["provider"]
    headless = ctx.obj["headless"]

    browser = get_browser(headless=headless)
    try:
        if not browser.goto(url, wait=3):
            console.print("[red]Failed to load post[/red]")
            return

        browser.ensure_healthy()

        screenshot = browser.screenshot_post("single_post")
        console.print(f"  Screenshot: {screenshot}")

        from .vision import analyze_screenshot

        with console.status("Analyzing with vision model..."):
            data = analyze_screenshot(screenshot, provider)

        if "error" in data:
            console.print(f"[red]Error: {data['error']}[/red]")
        else:
            console.print(Panel(
                f"[bold]@{data.get('author_username', '?')}[/bold]\n\n"
                f"{data.get('post_text', 'N/A')[:500]}\n\n"
                f"Views: {data.get('views', '?')} | "
                f"Likes: {data.get('likes', '?')} | "
                f"Replies: {data.get('replies', '?')} | "
                f"Reposts: {data.get('reposts', '?')} | "
                f"Shares: {data.get('shares', '?')}\n\n"
                f"Topic: {data.get('topic_category', '?')} | "
                f"Sentiment: {data.get('sentiment', '?')} | "
                f"Hook: {data.get('hook_type', '?')}",
                title="[bold cyan]Post Analysis[/bold cyan]",
            ))

    finally:
        browser.stop()


if __name__ == "__main__":
    main()
