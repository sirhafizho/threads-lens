"""Vision analysis via genai-nexus — sends screenshots to LLM for structured extraction."""

import base64
import json
import os
import re
from pathlib import Path

from openai import OpenAI

PROVIDERS = {
    "kimi": {
        "base_url": "https://genai-nexus.api.corpinter.net/moonshot",
        "model": "moonshotai.kimi-k2.5",
        "name": "Kimi K2.5",
    },
    "glm": {
        "base_url": "https://genai-nexus.api.corpinter.net/zai",
        "model": "zai.glm-5",
        "name": "GLM-5",
    },
}

EXTRACTION_PROMPT = """You are analyzing screenshot(s) of a Threads (social media) post. The images may show the post and/or its replies section. Extract the following information and return ONLY valid JSON, no other text:

{
    "author_username": "the @username of the post author",
    "author_display_name": "display name if visible",
    "post_text": "the full text content of the post",
    "post_text_summary": "one-line summary of what the post is about",
    "language": "primary language of the post (e.g. en, ms, id)",
    "likes": <number or null>,
    "replies": <number or null>,
    "reposts": <number or null>,
    "shares": <number or null>,
    "views": <number or null>,
    "has_media": true/false,
    "media_type": "text|image|video|carousel|link",
    "media_description": "describe what is in the image/video frame — people, text overlay, scene, mood (null if text-only)",
    "topic_category": "categorize: politics|entertainment|tech|lifestyle|food|sports|religion|gossip|news|personal|humor|other",
    "is_controversial": true/false,
    "has_hook": true/false,
    "hook_type": "question|bold_claim|how_to|story|controversial|list|none",
    "has_hashtags": true/false,
    "sentiment": "positive|negative|neutral|outrage|humorous",
    "timestamp_visible": "the relative timestamp shown (e.g. '2h', '3d')",
    "comments": [
        {"author": "@username", "text": "comment text", "likes": <number or null>}
    ],
    "comments_summary": "2-3 sentence summary of the discussion — main arguments, sentiment, notable takes",
    "discussion_sentiment": "supportive|mixed|hostile|debating|humorous|none",
    "reply_insight": "what the replies reveal about audience reaction to this post"
}

Rules:
- If a metric isn't visible or readable, use null
- Numbers like "1.2K" = 1200, "5.6M" = 5600000
- post_text should be the actual text, not a summary
- Be accurate with numbers — read them carefully
- comments: extract up to 5 visible comments/replies (empty array if none visible)
- If no replies section is visible, set comments to [], and comments_summary/discussion_sentiment/reply_insight to null
- media_description: describe what is actually visible — do not guess what is off-screen
- For videos: describe the visible frame and any text overlay you can read"""


def get_api_key() -> str:
    key = os.environ.get("NEXUS_API_KEY", "")
    if not key:
        try:
            settings_path = os.path.expanduser("~/.claude/settings.json")
            with open(settings_path) as f:
                settings = json.load(f)
            key = settings.get("env", {}).get("AWS_BEARER_TOKEN_BEDROCK", "")
        except Exception:
            pass
    return key


def _get_client(provider: str = "kimi") -> tuple[OpenAI, str]:
    cfg = PROVIDERS.get(provider, PROVIDERS["kimi"])
    return (
        OpenAI(api_key=get_api_key(), base_url=cfg["base_url"], timeout=120),
        cfg["model"],
    )


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"raw_response": text, "parse_error": True}


def analyze_screenshot(
    image_paths: str | list[str],
    provider: str = "kimi",
    prompt: str = None,
) -> dict:
    """Send screenshot(s) to the vision LLM and get structured extraction.

    Accepts a single path or a list of paths (e.g. post + replies screenshots).
    Returns dict with post data or error info.
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    for p in image_paths:
        if not os.path.exists(p):
            return {"error": f"File not found: {p}"}

    client, model = _get_client(provider)
    actual_prompt = prompt or EXTRACTION_PROMPT

    # Build content array: text prompt + all images
    content = [{"type": "text", "text": actual_prompt}]
    for p in image_paths:
        b64 = _encode_image(p)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=2000,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content
        result = _parse_json_response(raw)
        result["_screenshot"] = image_paths[0]
        result["_provider"] = provider
        return result
    except Exception as e:
        return {"error": str(e), "_screenshot": image_paths[0], "_provider": provider}


def check_page_type(image_path: str, provider: str = "kimi") -> str:
    """Quick check: what type of page is this? Returns: post|feed|login|captcha|error|unknown"""
    prompt = """Look at this screenshot. What type of page is this? Reply with ONE word only:
- "post" if it's a single Threads post
- "feed" if it's a Threads feed/timeline with multiple posts
- "login" if it's a login/signup wall
- "captcha" if there's a CAPTCHA or verification challenge
- "error" if it's an error page
- "popup" if a popup/dialog is blocking the content
- "other" for anything else"""

    client, model = _get_client(provider)
    b64 = _encode_image(image_path)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=10,
            temperature=0,
        )
        return resp.choices[0].message.content.strip().lower()
    except Exception:
        return "unknown"
