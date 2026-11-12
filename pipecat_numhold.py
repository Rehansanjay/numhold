"""numhold for Pipecat: a user-turn stop strategy that doesn't end the caller's
turn while they are in the middle of giving a number.

Wrap whichever turn detection you use now:

    from pipecat_numhold import AgentTextRecorder, NumberHoldSmartTurn

    hold = NumberHoldSmartTurn(turn_analyzer=LocalSmartTurnAnalyzerV3())
    aggregators = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(stop=[hold])))
    pipeline = Pipeline([transport.input(), stt, aggregators.user(), llm, tts,
                         AgentTextRecorder(hold), transport.output(), aggregators.assistant()])

or NumberHoldSpeechTimeout(user_speech_timeout=...) for a plain silence timer.

When your detector says the caller has finished, the hold looks at the
transcript so far (and at what the agent just asked). If a number is clearly
still coming ("my number is nine eight four zero one"), it waits until the
caller speaks again or `max_hold_s` of silence passes. Turns with no number in
them are not touched and get no extra delay.

AgentTextRecorder is optional but matters on real calls: callers usually answer
"what's your mobile number?" with digits alone, and the expected length of the
number is in the question.

Tested with Pipecat 1.15.
"""
import asyncio
from collections.abc import Callable

from loguru import logger

from numhold import MAX_HOLD_S, number_in_progress
from pipecat.frames.frames import TranscriptionFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy


class NumberHold:
    """Mix in before a Pipecat user-turn stop strategy (see the classes below)."""

    def __init__(self, *, max_hold_s: float = MAX_HOLD_S, on_event: Callable[[str], None] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.max_hold_s = max_hold_s
        self.agent_said: list[str] = []  # what the agent said since the caller's last turn
        self._on_event = on_event or (lambda msg: logger.debug(f"numhold: {msg}"))
        self._turn_text: list[str] = []
        self._hold_task = None
        self._hold_expired = False

    async def process_frame(self, frame):
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._turn_text.append(frame.text.strip())
        await super().process_frame(frame)

    async def report_user_turn_stopped(self, **kwargs):
        text = " ".join(self._turn_text)
        hold, why = number_in_progress(text, " ".join(self.agent_said))
        if hold and not self._hold_expired:
            if self._hold_task is None:
                self._on_event(f"HOLD ({why}): turn detector said finished at {text!r}")
                self._hold_task = self.task_manager.create_task(self._expire(kwargs), f"{self}::number_hold")
            return
        self._on_event(f"TURN ENDED at {text!r}")
        self.agent_said.clear()
        await super().report_user_turn_stopped(**kwargs)

    async def _expire(self, kwargs):
        try:
            await asyncio.sleep(self.max_hold_s)
        except asyncio.CancelledError:
            return
        self._hold_task = None
        self._hold_expired = True
        self._on_event(f"hold expired after {self.max_hold_s:.1f} s")
        await self.report_user_turn_stopped(**kwargs)

    async def _discard_pending_end_of_turn(self):
        # The caller spoke again: whatever the hold was waiting for has happened.
        if self._hold_task:
            await self.task_manager.cancel_task(self._hold_task)
            self._hold_task = None
            self._on_event("caller resumed during the hold")
        self._hold_expired = False
        await super()._discard_pending_end_of_turn()

    async def _reset(self, *args, **kwargs):
        self._turn_text = []
        await super()._reset(*args, **kwargs)


class NumberHoldSmartTurn(NumberHold, TurnAnalyzerUserTurnStopStrategy):
    """Smart Turn (or any turn analyzer), plus the number hold."""


class NumberHoldSpeechTimeout(NumberHold, SpeechTimeoutUserTurnStopStrategy):
    """A plain silence timer, plus the number hold."""


class AgentTextRecorder(FrameProcessor):
    """Place after the TTS service. Tells the hold what the agent just said."""

    def __init__(self, hold: NumberHold, **kwargs):
        super().__init__(**kwargs)
        self._hold = hold

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSTextFrame) and frame.text.strip():
            self._hold.agent_said.append(frame.text.strip())
        await self.push_frame(frame, direction)
