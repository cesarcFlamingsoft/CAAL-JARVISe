"""One delayed cue within the answer's audio stream, never a queued speech."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import TypeVar

Frame = TypeVar("Frame")
WAITING_CUE = "One moment."
WAITING_CUE_DELAY = 3.0
# A whitespace-only first chunk starts LiveKit's TTS segment before LLM text.
# It is removed before synthesis; cue words never enter the model's transcript.
WAITING_AUDIO_START = " "


async def with_waiting_cue(
    answer: AsyncIterator[Frame], cue: Callable[[], AsyncIterator[Frame]], *, delay: float
) -> AsyncIterator[Frame]:
    first = asyncio.create_task(anext(answer, None))
    try:
        done, _ = await asyncio.wait({first}, timeout=delay)
        if not done:
            cue_audio = cue()
            next_cue = None
            try:
                while not first.done():
                    next_cue = asyncio.create_task(anext(cue_audio, None))
                    ready, _ = await asyncio.wait(
                        {first, next_cue}, return_when=asyncio.FIRST_COMPLETED
                    )
                    # Answer wins a tie, including the last cue synthesis frame.
                    if first in ready:
                        break
                    frame = next_cue.result()
                    if frame is None:
                        break
                    yield frame
            finally:
                if next_cue is not None:
                    next_cue.cancel()
                    await asyncio.gather(next_cue, return_exceptions=True)
                await cue_audio.aclose()
        frame = await first
        if frame is not None:
            yield frame
            async for frame in answer:
                yield frame
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        await answer.aclose()


async def waiting_text_transform(text):
    """Keep the start marker ahead of the SDK's buffering speech filters.

    Passed as the session's text transform. Retain the same Markdown/emoji
    filtering on all actual speech, including fixed say() confirmations.
    """
    from livekit.agents.voice.transcription.filters import filter_emoji, filter_markdown

    source = text.__aiter__()
    first = await anext(source, None)
    if first == WAITING_AUDIO_START:
        yield first

    async def words():
        if first is not None and first != WAITING_AUDIO_START:
            yield first
        async for part in source:
            yield part

    async for part in filter_emoji(filter_markdown(words())):
        yield part
