"""DOM text parser — extracts Threads post data from inner_text() output.

No CSS selectors needed — parses the plain text stream.
Pattern: username → timestamp → text → metrics (likes, replies, reposts, shares)
"""

import re
from dataclasses import dataclass, field


def _parse_number(s: str) -> int | None:
    """Parse Threads metric strings like '36.2K', '346', '7.2M', '2.3K'."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip().rstrip("\xa0")
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    # Handle K/M suffix
    match = re.match(r"^([\d,.]+)\s*([KkMm])$", s)
    if match:
        num = float(match.group(1).replace(",", ""))
        suffix = match.group(2).upper()
        return int(num * (1000 if suffix == "K" else 1000000))
    # Handle comma-separated numbers
    try:
        return int(s.replace(",", ""))
    except ValueError:
        return None


_TIMESTAMP_RE = re.compile(r"^\d+[smhdw]$|^\d+\s*(sec|min|hr|day|week)s?\s*ago$", re.IGNORECASE)

# Sidebar/nav items to skip when looking for the first post
_NAV_ITEMS = {
    "For you", "New thread", "Search", "Messages", "Activity", "Profile",
    "Insights", "Communities", "Other feeds", "Edit", "Following", "Saved",
    "Liked", "Show more", "Post", "First thread", "What's new?",
    "Germany", "© 2026", "Threads", "Terms", "Privacy Policy",
    "Cookies Policy", "Log in", "Sign up", "Translate", "More", "See translation",
    "Reply", "Share", "Repost", "Like", "View replies", "Hide replies",
}


def _is_timestamp(line: str) -> bool:
    """Check if a line looks like a Threads timestamp (2d, 3h, 5m, etc)."""
    return bool(_TIMESTAMP_RE.match(line.strip()))


def _is_metric_line(line: str) -> bool:
    """Check if a line is a bare metric number (346, 36.2K, 2.3M)."""
    s = line.strip()
    if not s:
        return False
    # Plain number or K/M suffixed
    return bool(re.match(r"^[\d,.]+[KMkm]?$", s))


def _looks_like_username(line: str) -> bool:
    """Check if a line looks like a Threads username."""
    s = line.strip()
    if not s or len(s) > 60:
        return False
    if s in _NAV_ITEMS:
        return False
    # Usernames: alphanumeric, dots, underscores — no spaces
    return bool(re.match(r"^[a-zA-Z0-9._]+$", s))


@dataclass
class ParsedPost:
    """Structured data extracted from DOM text."""
    author: str = ""
    timestamp: str = ""
    text_lines: list[str] = field(default_factory=list)
    likes: int | None = None
    replies: int | None = None
    reposts: int | None = None
    shares: int | None = None
    views: int | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.text_lines)

    def to_dict(self) -> dict:
        return {
            "author_username": self.author,
            "timestamp_visible": self.timestamp,
            "post_text": self.text,
            "likes": self.likes,
            "replies": self.replies,
            "reposts": self.reposts,
            "shares": self.shares,
            "views": self.views,
        }


@dataclass
class ParsedComment:
    """A comment/reply extracted from DOM text."""
    author: str = ""
    timestamp: str = ""
    text_lines: list[str] = field(default_factory=list)
    likes: int | None = None
    replies: int | None = None
    reposts: int | None = None

    @property
    def text(self) -> str:
        return " ".join(self.text_lines)

    def to_dict(self) -> dict:
        return {
            "author": self.author,
            "text": self.text[:300],
            "likes": self.likes,
        }


def parse_page_text(text: str) -> tuple[ParsedPost | None, list[ParsedComment]]:
    """Parse inner_text() output into a post + comments.

    Returns (post, comments) tuple.
    """
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return None, []

    # Find where the actual content starts (skip nav)
    start = 0
    for i, line in enumerate(lines):
        if _looks_like_username(line) and i + 1 < len(lines) and _is_timestamp(lines[i + 1]):
            start = i
            break

    if start >= len(lines):
        return None, []

    # Parse post: username → timestamp → text → metrics
    post = ParsedPost()
    i = start

    post.author = lines[i]
    i += 1
    if i < len(lines) and _is_timestamp(lines[i]):
        post.timestamp = lines[i]
        i += 1

    # Collect text lines until we hit a metric line
    while i < len(lines):
        if _is_metric_line(lines[i]):
            # Check if next few lines are also metrics (the 4-metric block)
            metric_count = 0
            j = i
            while j < len(lines) and _is_metric_line(lines[j]) and metric_count < 5:
                metric_count += 1
                j += 1
            if metric_count >= 2:  # At least 2 metric lines = metrics block
                break
        post.text_lines.append(lines[i])
        i += 1

    # Parse metrics block (likes, replies, reposts, shares, views)
    metrics = []
    while i < len(lines) and _is_metric_line(lines[i]) and len(metrics) < 5:
        metrics.append(_parse_number(lines[i]))
        i += 1

    if len(metrics) >= 1:
        post.likes = metrics[0]
    if len(metrics) >= 2:
        post.replies = metrics[1]
    if len(metrics) >= 3:
        post.reposts = metrics[2]
    if len(metrics) >= 4:
        post.shares = metrics[3]

    # Parse comments
    comments = []
    while i < len(lines):
        # Look for username + timestamp pattern
        if (
            _looks_like_username(lines[i])
            and i + 1 < len(lines)
            and _is_timestamp(lines[i + 1])
        ):
            comment = ParsedComment(author=lines[i], timestamp=lines[i + 1])
            i += 2

            # Collect comment text until metrics or next comment
            while i < len(lines):
                if _is_metric_line(lines[i]):
                    # Check for metric block
                    metric_count = 0
                    j = i
                    while j < len(lines) and _is_metric_line(lines[j]) and metric_count < 4:
                        metric_count += 1
                        j += 1
                    if metric_count >= 1:
                        break
                if (
                    _looks_like_username(lines[i])
                    and i + 1 < len(lines)
                    and _is_timestamp(lines[i + 1])
                ):
                    break
                comment.text_lines.append(lines[i])
                i += 1

            # Parse comment metrics
            comment_metrics = []
            while i < len(lines) and _is_metric_line(lines[i]) and len(comment_metrics) < 4:
                comment_metrics.append(_parse_number(lines[i]))
                i += 1

            if len(comment_metrics) >= 1:
                comment.likes = comment_metrics[0]
            if len(comment_metrics) >= 2:
                comment.replies = comment_metrics[1]
            if len(comment_metrics) >= 3:
                comment.reposts = comment_metrics[2]

            # Filter out suggested posts (not real replies)
            # Suggested posts have no metrics and often contain hashtags/promotional text
            has_hashtag = any("#" in l for l in comment.text_lines)
            if comment_metrics or (comment.text_lines and not has_hashtag):
                comments.append(comment)
        else:
            i += 1

    return post, comments
