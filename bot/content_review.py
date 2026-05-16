"""ContentReviewMixin: auto-reviews video transcripts via Claude API after approval notifications."""

import asyncio
import logging
import os

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
    """Adds automatic Claude content review after video request notifications."""

    async def _send_content_review(self, video: dict) -> None:
        """Fetch transcript and post a Claude content review as a follow-up Telegram message."""
        video_id = video["video_id"]
        title = video["title"]

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — skipping content review")
            return

        loop = asyncio.get_event_loop()

        # Fetch transcript
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

            def _fetch():
                try:
                    return YouTubeTranscriptApi.get_transcript(
                        video_id, languages=["en", "en-US", "en-GB"]
                    )
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

        # Call Claude API
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            def _review():
                return client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=REVIEW_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Review this transcript for the video \"{title}\":\n\n"
                            f"{transcript_text[:30000]}"
                        ),
                    }],
                )

            response = await loop.run_in_executor(None, _review)
            review_text = response.content[0].text

            header = f"\U0001f50d Content Review: {title}\n\n"
            full_text = header + review_text

            # Send, splitting at Telegram's 4096-char limit if needed
            for i in range(0, len(full_text), 4096):
                await self._app.bot.send_message(
                    chat_id=self.admin_chat_target,
                    text=full_text[i:i + 4096],
                )

        except Exception as e:
            logger.error(f"Content review API call failed for {video_id}: {e}")
