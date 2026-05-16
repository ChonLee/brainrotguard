"""ContentReviewMixin: auto-reviews video transcripts via a local Ollama instance."""

import asyncio
import logging
import os

from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

REVIEW_SYSTEM_PROMPT = (
    "You are a Christian content reviewer. Screen video transcripts and flag anything "
    "inappropriate for a Christian family audience.\n\n"
    "Review for:\n"
    "- Language: profanity (any level), euphemisms (freaking, shoot, crap), "
    "blasphemy/Lord's name in vain, crude humor\n"
    "- Sexual content: suggestive language, innuendo, references to sexual activity\n"
    "- Violence: graphic violence, dark/occult themes, aggressive language\n"
    "- Substances: alcohol or drug use portrayed positively\n"
    "- Other: mockery of faith/Christianity, anti-Christian worldviews, gambling, "
    "disrespect toward authority\n\n"
    "Report format (concise):\n"
    "1. Summary — one sentence on what the video is about\n"
    "2. Flags — each concern with severity (mild/moderate/strong) and brief context. "
    "Dismiss false positives briefly.\n"
    "3. Clean — categories with nothing flagged\n"
    "4. Verdict — Suitable / Not suitable / Borderline (with one-line reason)\n\n"
    "Be thorough but not alarmist. Flag real concerns clearly."
)


class ContentReviewMixin:
    """Adds automatic Ollama content review after video request notifications."""

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
        """Fetch transcript and post an Ollama content review as a follow-up Telegram message."""
        video_id = video["video_id"]
        title = video["title"]

        ollama_url = os.environ.get("OLLAMA_BASE_URL")
        if not ollama_url:
            logger.warning("OLLAMA_BASE_URL not set — skipping content review")
            return

        ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
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

            transcript_text = " ".join(entry["text"] for entry in transcript_list)

        except Exception as e:
            logger.warning(f"Transcript fetch failed for {video_id}: {e}")
            await self._app.bot.send_message(
                chat_id=self.admin_chat_target,
                text=f"⚠️ Could not fetch transcript for review: {title}",
            )
            return

        # Call Ollama via OpenAI-compatible API
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=f"{ollama_url.rstrip('/')}/v1",
                api_key="ollama",  # Ollama doesn't require a real key
            )

            def _review():
                return client.chat.completions.create(
                    model=ollama_model,
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Review this transcript for the video \"{title}\":\n\n"
                                f"{transcript_text[:30000]}"
                            ),
                        },
                    ],
                )

            response = await loop.run_in_executor(None, _review)
            review_text = response.choices[0].message.content

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
