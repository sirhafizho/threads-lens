"""Hybrid extractor — DOM for metrics/text/comments, vision only for media, text LLM for classification."""

import asyncio
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from .dom_parser import ParsedComment, ParsedPost, parse_page_text
from .vision import PROVIDERS, get_api_key

CLASSIFY_PROMPT = """You are classifying a Threads post. Given the post data below, return ONLY valid JSON:

{{
    "language": "primary language (en|ms|id|other)",
    "topic_category": "politics|entertainment|tech|lifestyle|food|sports|religion|gossip|news|personal|humor|other",
    "is_controversial": true/false,
    "has_hook": true/false,
    "hook_type": "question|bold_claim|how_to|story|controversial|list|none",
    "has_hashtags": true/false,
    "sentiment": "positive|negative|neutral|outrage|humorous",
    "post_text_summary": "one-line summary",
    "has_media": true/false,
    "media_type": "text|image|video|carousel|link"
}}

Post data:
- Author: {author}
- Text: {text}
- Metrics: likes={likes}, replies={replies}, reposts={reposts}, shares={shares}
- Comments: {comments_summary}"""

DISCUSSION_PROMPT = """You are analyzing the comment section of a Threads post. Given the comments below, return ONLY valid JSON:

{{
    "comments_summary": "2-3 sentence summary of the discussion",
    "discussion_sentiment": "supportive|mixed|hostile|debating|humorous|none",
    "reply_insight": "what the replies reveal about audience reaction"
}}

Post: {post_text}
Metrics: {likes} likes, {replies} replies

Comments:
{comments_text}"""

MEDIA_PROMPT = """Describe what's in this image or video frame. Be concise — people, text overlay, scene, mood. Return ONLY the description, no JSON."""


def _get_client(provider: str = "kimi") -> tuple[OpenAI, str]:
    cfg = PROVIDERS.get(provider, PROVIDERS["kimi"])
    return (
        OpenAI(api_key=get_api_key(), base_url=cfg["base_url"], timeout=120),
        cfg["model"],
    )


def _call_llm(client: OpenAI, model: str, prompt: str, max_tokens: int = 500) -> dict:
    """Call text-only LLM and parse JSON response."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content
        # Parse JSON
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw)
    except Exception as e:
        return {"_error": str(e)}


def _call_llm_text(client: OpenAI, model: str, prompt: str, max_tokens: int = 200) -> str:
    """Call text-only LLM, return raw text."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def _call_vision(client: OpenAI, model: str, prompt: str, image_path: str) -> str:
    """Call vision LLM with a single image. Returns text description."""
    import base64
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_tokens=300,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


async def extract_post(
    page_text: str,
    screenshot_path: str | None = None,
    provider: str = "kimi",
) -> dict:
    """Extract post data using DOM text + minimal vision.

    - DOM: author, text, metrics, comments (free, instant)
    - Vision: media_description only if image/video detected
    - Text LLM: topic, sentiment, hook, discussion summary
    """
    post, comments = parse_page_text(page_text)

    if not post:
        return {"error": "Could not parse post from page text", "parse_error": True}

    result = post.to_dict()
    result["comments"] = [c.to_dict() for c in comments]
    result["_screenshot"] = screenshot_path
    result["_provider"] = provider
    result["_extraction"] = "hybrid"

    client, model = _get_client(provider)

    # Detect media from screenshot (quick vision check)
    has_media = screenshot_path and await _has_media(screenshot_path, provider)
    media_type = "text"
    media_desc = None

    if has_media:
        media_type = await _detect_media_type(screenshot_path, provider)
        media_desc = await asyncio.to_thread(
            _call_vision, client, model, MEDIA_PROMPT, screenshot_path
        )

    # Classify post via text-only LLM
    comments_text = f"{len(comments)} comments" if comments else "no comments"
    classify_prompt = CLASSIFY_PROMPT.format(
        author=post.author,
        text=post.text[:500],
        likes=post.likes,
        replies=post.replies,
        reposts=post.reposts,
        shares=post.shares,
        comments_summary=comments_text,
    )
    classification = await asyncio.to_thread(_call_llm, client, model, classify_prompt)

    if "_error" not in classification:
        result.update(classification)

    # Override media fields with actual detection
    result["has_media"] = has_media
    result["media_type"] = media_type
    result["media_description"] = media_desc

    # Analyze discussion if comments exist
    if comments:
        comments_str = "\n".join(
            f"@{c.author}: {c.text[:200]} ({c.likes or 0} likes)" for c in comments[:8]
        )
        discussion_prompt = DISCUSSION_PROMPT.format(
            post_text=post.text[:300],
            likes=post.likes or 0,
            replies=post.replies or 0,
            comments_text=comments_str,
        )
        discussion = await asyncio.to_thread(_call_llm, client, model, discussion_prompt)

        if "_error" not in discussion:
            result.update(discussion)
    else:
        result["comments_summary"] = None
        result["discussion_sentiment"] = "none"
        result["reply_insight"] = None

    return result


async def _has_media(screenshot_path: str, provider: str) -> bool:
    """Quick check if the post has visual media."""
    from .vision import check_page_type
    # Simple heuristic — check if screenshot is tall (has media area)
    # For now, just check if there's a media indicator in the image
    # We'll use a lightweight vision call
    client, model = _get_client(provider)
    desc = await asyncio.to_thread(
        _call_vision, client, model,
        "Does this screenshot show an image or video post? Answer only 'yes' or 'no'.",
        screenshot_path,
    )
    return desc.lower().strip().startswith("y")


async def _detect_media_type(screenshot_path: str, provider: str) -> str:
    """Detect what type of media the post has."""
    client, model = _get_client(provider)
    desc = await asyncio.to_thread(
        _call_vision, client, model,
        "What type of media is in this post? Answer one word: image, video, carousel, or link.",
        screenshot_path,
    )
    desc = desc.lower().strip()
    if "video" in desc:
        return "video"
    if "carousel" in desc or "multiple" in desc:
        return "carousel"
    if "link" in desc:
        return "link"
    return "image"
