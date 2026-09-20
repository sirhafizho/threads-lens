# Threads Lens

AI agent that reads Threads posts via Playwright screenshots + LLM vision — no Threads API needed. It browses Threads as you (using your logged-in Chrome profile), screenshots posts, and uses vision AI to extract engagement metrics.

## Why?

Threads has an official API but it's limited — no engagement metrics on other people's posts without advanced access approval. This agent bypasses that entirely:

```
Playwright → Screenshot post → LLM Vision → Structured data → Markdown report
```

## Features

- **Human-in-the-loop**: Detects login walls, CAPTCHAs, popups — pauses and alerts you to fix, then continues
- **Vision extraction**: Kimi K2.5 reads likes, replies, reposts, shares, views from screenshots
- **No database**: All output to markdown files — easy to read, share, version control
- **Multiple modes**: Research topics, browse trending, analyze profiles, single post
- **Virality analysis**: Understand what makes posts go viral on Threads

## Quick Start

```bash
git clone https://github.com/sirhafizho/threads-lens.git
cd threads-lens
python -m venv .venv && source .venv/bin/activate
pip install -e .
playwright install chromium

# Set LLM key
export NEXUS_API_KEY="your-key"

# First run — browser opens, log into Threads once (session persists)
tlens research "malaysian politics" --count 5

# After that, it stays logged in
tlens trending --count 15 --report
tlens profile @username --count 10
tlens post https://www.threads.com/@user/post/ABC123
```

## Commands

| Command | What it does |
|---------|-------------|
| `tlens research <topic>` | Search Threads for a topic, screenshot + analyze posts |
| `tlens trending` | Browse your Threads feed, analyze trending posts |
| `tlens profile <user>` | Analyze a specific user's posts and engagement |
| `tlens post <url>` | Analyze a single Threads post |

## Options

| Flag | Description |
|------|-------------|
| `--count N` | Max posts to analyze (default: 10-15) |
| `--report` | Generate full virality report after collection |
| `--provider kimi` | Vision model: kimi (default), glm, gemini |
| `--headless` | Run browser headless (no visible window) |

## Human-in-the-Loop

When the agent hits an issue (login wall, CAPTCHA, popup):

```
⚠ ISSUE DETECTED: login_wall
  Screenshot saved: output/screenshots/issue_login_wall_....png
  Threads is asking for login.
  The browser window should be visible — please log in manually.
  Once you're logged in, press Enter here to continue.
  [Press Enter when ready] >
```

You fix it in the visible browser, press Enter, and the agent continues.

## What Gets Extracted Per Post

- Author (@username, display name)
- Post text + summary
- **Views, Likes, Replies, Reposts, Shares**
- Media type (text/image/video/carousel)
- Topic category (politics, entertainment, gossip, etc.)
- Controversy flag
- Hook type (question, bold claim, how-to, story)
- Sentiment (positive, negative, outrage, humorous)
- Language
- Hashtags

## Output

All data saved as markdown in `output/`:

```
output/
├── screenshots/           # Raw screenshots
├── search_<topic>_<ts>/   # Per-session data
│   └── data.md            # Structured post data
└── report_<name>_<ts>.md  # Analysis reports
```

## Vision Models

| Provider | Model | Notes |
|----------|-------|-------|
| `kimi` (default) | Kimi K2.5 | Good all-around, via genai-nexus |
| `glm` | GLM-5 | Alternative |
| `gemini` | Gemini 2.5 Flash | Fast |

All via genai-nexus (OpenAI-compatible API).

## License

MIT
