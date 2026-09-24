"""ContentReviewMixin: auto-reviews video transcripts via Claude or Ollama (local or cloud)."""

import asyncio
import logging
import os
import re

from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# Reviews are fire-and-forget tasks (see approval.py), so a burst of video
# requests would otherwise hit the model all at once. Locally that exhausts
# VRAM and can hang the host; on cloud it trips the plan's concurrency cap.
# Free tier allows 1 concurrent request, Pro 3, Max 10.
_review_sem: asyncio.Semaphore | None = None


def _get_review_semaphore() -> asyncio.Semaphore:
    global _review_sem
    if _review_sem is None:
        limit = max(1, int(os.environ.get("CONTENT_REVIEW_CONCURRENCY", "1")))
        _review_sem = asyncio.Semaphore(limit)
    return _review_sem


def _resolve_provider() -> dict | None:
    """Pick the review backend: Anthropic if keyed, else Ollama Cloud, else local Ollama."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        # Claude model IDs have no colons, so overriding via Unraid env is safe here.
        model = os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-5"
        return {"kind": "anthropic", "api_key": anthropic_key, "model": model}

    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        # NOTE: model default is baked in, not read from an Unraid env var --
        # colons in values break that template parser.
        model = os.environ.get("OLLAMA_CLOUD_MODEL") or "gpt-oss:120b"
        return {"kind": "openai", "base_url": "https://ollama.com/v1",
                "api_key": api_key, "model": model}

    base = os.environ.get("OLLAMA_BASE_URL")
    if not base:
        return None
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
    return {"kind": "openai", "base_url": f"{base.rstrip('/')}/v1",
            "api_key": "ollama", "model": model}


def _call_model(provider: dict, user_content: str) -> str:
    """Run the review synchronously against the chosen provider; returns review text."""
    if provider["kind"] == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=provider["api_key"])
        response = client.messages.create(
            model=provider["model"],
            # Room for long videos with many flags.
            max_tokens=4096,
            # Straightforward classification -- skip thinking to keep cost/latency down.
            thinking={"type": "disabled"},
            system=REVIEW_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined to review this transcript")
        return "".join(b.text for b in response.content if b.type == "text")

    from openai import OpenAI

    client = OpenAI(base_url=provider["base_url"], api_key=provider["api_key"])
    response = client.chat.completions.create(
        model=provider["model"],
        max_tokens=1024,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content

REVIEW_SYSTEM_PROMPT = (
    "You are a Christian content reviewer. Screen video transcripts and flag anything "
    "inappropriate for a Christian family audience.\n\n"
    "Review for:\n"
    "- Language: profanity (any level), euphemisms (freaking, shoot, crap), "
    "blasphemy/Lord's name in vain, crude humor\n"
    "- Sexual content: explicit suggestive language, innuendo, or direct references to sexual activity. "
    "Requires clear intent — do not infer adult themes from ambiguous words alone.\n"
    "- Violence: graphic violence, dark/occult themes, aggressive language directed at real people\n"
    "- Substances: alcohol or drug use portrayed positively\n"
    "- Other: mockery of faith/Christianity, anti-Christian worldviews, gambling, "
    "disrespect toward authority\n\n"
    "YouTube transcript note: YouTube auto-censors profanity by replacing it with [ __ ] in "
    "transcripts. Treat any occurrence of [ __ ] as censored profanity — flag it the same as "
    "if the actual word were present. Do not dismiss [ __ ] as unknown or benign. "
    "Only flag [ __ ] if it literally appears in the transcript text provided — do not infer "
    "censored profanity based on your knowledge of the song or artist.\n\n"
    "Common false positives — dismiss these without flagging:\n"
    "- Sports team names that contain flaggable words: Magic (Orlando Magic), Heat, Wizards, "
    "Devils, Rockets, Warriors, Thunder, Bulls, etc.\n"
    "- Sports violence language: kill, crush, destroy, murder, attack, beat — when used to "
    "describe game outcomes or play\n"
    "- Arena, court, field, stadium references in a sports context\n"
    "- 'Magic' in gaming or sports context (Orlando Magic, magic spells in a fantasy game)\n"
    "- Travel or location references that happen to sound suggestive\n\n"
    "When context clearly indicates sports, gaming, or other benign activity, assume that "
    "interpretation before inferring adult or violent intent.\n\n"
    "The transcript includes [M:SS] timestamp markers. When listing a flag, cite the nearest "
    "timestamp using the same [M:SS] format so the parent can jump directly to that moment.\n\n"
    "Report format (concise):\n"
    "1. Summary — one sentence on what the video is about\n"
    "2. Flags — each concern with severity (mild/moderate/strong), a [M:SS] timestamp, "
    "and brief context. Dismiss false positives briefly.\n"
    "3. Clean — categories with nothing flagged\n"
    "4. Verdict — Suitable / Not suitable / Borderline (with one-line reason)\n\n"
    "Be thorough but not alarmist. Flag real concerns clearly."
)

_TS_PAT = re.compile(r'\[(\d{1,3}):(\d{2})\]')


def _build_timestamped_transcript(entries, interval=20):
    """Build transcript text with [M:SS] markers every `interval` seconds."""
    parts = []
    last_marker = -interval
    for e in entries:
        if e.start - last_marker >= interval:
            m, s = divmod(int(e.start), 60)
            parts.append(f"[{m}:{s:02d}]")
            last_marker = e.start
        parts.append(e.text)
    return " ".join(parts)


def _extract_flag_links(review_text, video_id, limit=25):
    """Parse [M:SS] timestamps from the review and return HTML lines with context, up to limit."""
    lines = review_text.splitlines()
    seen = set()
    results = []
    for line in lines:
        for m in _TS_PAT.finditer(line):
            total_s = int(m.group(1)) * 60 + int(m.group(2))
            if total_s in seen:
                continue
            seen.add(total_s)
            ts_str = f"{m.group(1)}:{m.group(2)}"
            url = f"https://www.youtube.com/watch?v={video_id}&t={total_s}s"
            # Strip the timestamp marker and clean up surrounding punctuation/whitespace
            context = _TS_PAT.sub("", line).strip().lstrip("-•: ").strip()
            context = context[:120] + ("…" if len(context) > 120 else "")
            link = f'<a href="{url}">[{ts_str}]</a>'
            results.append(f"{link} {context}" if context else link)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break
    return results


class ContentReviewMixin:
    """Adds automatic content review after video request notifications."""

    async def _cmd_review(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/review <youtube_url_or_video_id> — manually run a content review."""
        if not await self._require_admin(update):
            return

        args = context.args
        if not args:
            await update.effective_message.reply_text(
                "Usage: /review <youtube_url_or_video_id>"
            )
            return

        from youtube.extractor import extract_video_id
        raw = args[0].strip()
        video_id = extract_video_id(raw) or raw

        # Basic sanity check on the ID
        import re
        if not re.fullmatch(r"[a-zA-Z0-9_-]{11}", video_id):
            await update.effective_message.reply_text(
                f"⚠️ Couldn't extract a valid video ID from: {raw}"
            )
            return

        # Try to get the title from metadata; fall back to the ID
        title = video_id
        try:
            from youtube.extractor import extract_metadata
            metadata = await extract_metadata(video_id)
            if metadata and metadata.get("title"):
                title = metadata["title"]
        except Exception:
            pass

        ack = await update.effective_message.reply_text(
            f"🔍 Reviewing: {title}\nThis may take up to a minute..."
        )

        await self._send_content_review({"video_id": video_id, "title": title})

        # Clean up the "reviewing..." message
        try:
            await ack.delete()
        except Exception:
            pass

    async def _send_content_review(self, video: dict) -> None:
        """Fetch transcript and post a content review as a follow-up Telegram message."""
        video_id = video["video_id"]
        title = video["title"]

        provider = _resolve_provider()
        if provider is None:
            logger.warning(
                "No ANTHROPIC_API_KEY, OLLAMA_API_KEY, or OLLAMA_BASE_URL set — "
                "skipping content review"
            )
            return
        loop = asyncio.get_event_loop()

        # Fetch transcript
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

            def _fetch():
                try:
                    api = YouTubeTranscriptApi()
                    transcript_list = api.list(video_id)
                    transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
                    return list(transcript.fetch())
                except (TranscriptsDisabled, NoTranscriptFound):
                    return None

            transcript_list = await loop.run_in_executor(None, _fetch)
            if transcript_list is None:
                await self._app.bot.send_message(
                    chat_id=self.admin_chat_target,
                    text=f"⚠️ No English transcript available for review: {title}",
                )
                return

            timestamped_transcript = _build_timestamped_transcript(transcript_list)

        except Exception as e:
            logger.warning(f"Transcript fetch failed for {video_id}: {e}")
            await self._app.bot.send_message(
                chat_id=self.admin_chat_target,
                text=f"⚠️ Could not fetch transcript for review: {title}",
            )
            return

        try:
            # Claude's 1M-token context fits any video's full transcript; local
            # Ollama models have small contexts, so they still get a capped slice.
            if provider["kind"] != "anthropic":
                timestamped_transcript = timestamped_transcript[:30000]
            user_content = (
                f"Review this transcript for the video \"{title}\":\n\n"
                f"{timestamped_transcript}"
            )

            async with _get_review_semaphore():
                review_text = await loop.run_in_executor(
                    None, _call_model, provider, user_content
                )

            # Extract flagged timestamps and send as clickable links
            flag_links = _extract_flag_links(review_text, video_id, limit=25)
            if flag_links:
                links_html = "⚠️ Flagged moments:\n" + "\n".join(flag_links)
                await self._app.bot.send_message(
                    chat_id=self.admin_chat_target,
                    text=links_html,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )

            header = f"\U0001f50d Content Review: {title}\n\n"
            full_text = header + review_text

            # Send, splitting at Telegram's 4096-char limit if needed
            for i in range(0, len(full_text), 4096):
                await self._app.bot.send_message(
                    chat_id=self.admin_chat_target,
                    text=full_text[i:i + 4096],
                )

        except Exception as e:
            logger.error(f"Content review failed for {video_id}: {e}")
