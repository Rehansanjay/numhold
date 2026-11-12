"""numhold for LiveKit Agents: don't end the caller's turn in the middle of a number.

Wrap the turn detector you use now:

    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    from livekit_numhold import NumberHoldTurnDetector

    session = AgentSession(
        ...,
        turn_handling={
            "turn_detection": NumberHoldTurnDetector(MultilingualModel()),
            "endpointing": {"min_delay": 0.3, "max_delay": 2.0},
        },
    )

or NumberHoldTurnDetector() with no model, if you end turns on silence alone
(it then ends the turn after `min_delay`, like VAD endpointing).

When LiveKit asks whether the caller has finished, the hold looks at the
transcript so far and at what the agent asked just before. If a number is
clearly still coming ("my number is nine eight four zero one"), it answers
"not finished", so LiveKit waits up to `max_delay` instead of `min_delay`, and
the caller resuming cancels the wait as usual. Otherwise the wrapped model
answers, unchanged. Turns with no number in them are not touched.

How long a hold lasts is `max_delay`. The measurements in this repo use 2 s;
LiveKit's default is 3 s.

Works with turn detectors that score the transcript (MultilingualModel,
EnglishModel). Not with the streaming audio turn detectors, which LiveKit
calls a different way. Tested with livekit-agents 1.8.11.
"""
from __future__ import annotations

from collections.abc import Callable

from livekit.agents import llm

from numhold import number_in_progress

FALLBACK_THRESHOLD = 0.5


def _texts(chat_ctx: llm.ChatContext) -> tuple[str, str]:
    """(what the caller has said this turn, what the agent said just before it)."""
    msgs = [m for m in chat_ctx.items if m.type == "message" and m.role in ("user", "assistant")]
    user, agent = [], []
    i = len(msgs) - 1
    while i >= 0 and msgs[i].role == "user":
        user.insert(0, msgs[i].text_content or "")
        i -= 1
    while i >= 0 and msgs[i].role == "assistant":
        agent.insert(0, msgs[i].text_content or "")
        i -= 1
    return " ".join(t for t in user if t), " ".join(t for t in agent if t)


class NumberHoldTurnDetector:
    """A LiveKit turn detector that holds while a number is clearly unfinished."""

    def __init__(self, model=None, *, on_event: Callable[[str], None] | None = None):
        self._model = model
        self._model_speaks_language = model is not None
        self._on_event = on_event or (lambda msg: None)

    @property
    def model(self) -> str:
        return f"numhold+{self._model.model}" if self._model is not None else "numhold"

    @property
    def provider(self) -> str:
        return self._model.provider if self._model is not None else "numhold"

    async def supports_language(self, language) -> bool:
        # Always say yes, so LiveKit asks us at every pause and the hold works in
        # any language. Remember whether the wrapped model can answer.
        if self._model is not None:
            self._model_speaks_language = await self._model.supports_language(language)
        return True

    async def unlikely_threshold(self, language) -> float | None:
        if self._model is not None and self._model_speaks_language:
            t = await self._model.unlikely_threshold(language)
            if t is not None:
                return t
        return FALLBACK_THRESHOLD

    async def predict_end_of_turn(self, chat_ctx: llm.ChatContext, *, timeout: float | None = None) -> float:
        said, asked = _texts(chat_ctx)
        hold, why = number_in_progress(said, asked)
        if hold:
            self._on_event(f"HOLD ({why}) at {said!r}")
            return 0.0
        if self._model is not None and self._model_speaks_language:
            return await self._model.predict_end_of_turn(chat_ctx, timeout=timeout)
        return 1.0
