"""NumberHoldTurnDetector wrapped around LiveKit's real MultilingualModel.

    python eval/livekit_model_check.py      (needs livekit-plugins-turn-detector and its model files)

Checks that the wrapper holds on an unfinished number (including digits said on
their own after the agent asked for a mobile number), and otherwise returns
exactly what the model returns.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit.agents import LanguageCode, llm  # noqa: E402
from livekit.plugins.turn_detector.multilingual import MultilingualModel  # noqa: E402

from livekit_numhold import NumberHoldTurnDetector  # noqa: E402

EN = LanguageCode("en")
CASES = [  # (agent asked, caller said so far, should hold)
    ("How can I help you?", "my mobile number is nine eight four zero one", True),
    ("What's your mobile number?", "nine eight four zero one", True),
    ("What's your mobile number?", "nine eight four zero one two three four five six", False),
    ("What's your pin code?", "six zero zero", True),
    ("How can I help you?", "I want to change my booking please", False),
    ("How can I help you?", "I want to change my", False),
]


def ctx(asked: str, said: str) -> llm.ChatContext:
    c = llm.ChatContext()
    c.add_message(role="assistant", content=asked)
    c.add_message(role="user", content=said)
    return c


class _InProcess:
    """Runs the model in this process. In an agent, LiveKit's job context does this."""

    def __init__(self):
        from livekit.plugins.turn_detector.multilingual import _EUORunnerMultilingual
        _EUORunnerMultilingual._download_files()
        self.runner = _EUORunnerMultilingual()
        self.runner.initialize()
        self.inference_executor = self

    async def do_inference(self, method: str, data: bytes) -> bytes:
        return self.runner.run(data)


async def main() -> None:
    import livekit.plugins.turn_detector.base as base
    job = _InProcess()
    base.get_job_context = lambda: job
    model = MultilingualModel()
    hold = NumberHoldTurnDetector(MultilingualModel())
    ok = True
    print(f"{'agent asked':30s} {'caller said so far':50s} {'model':>6s} {'+hold':>6s}")
    for asked, said, want_hold in CASES:
        await hold.supports_language(EN)
        p_model = await model.predict_end_of_turn(ctx(asked, said))
        p_hold = await hold.predict_end_of_turn(ctx(asked, said))
        good = (p_hold == 0.0) if want_hold else abs(p_hold - p_model) < 1e-6
        ok &= good
        print(f"{asked:30s} {said:50s} {p_model:6.3f} {p_hold:6.3f} {'' if good else '  <-- WRONG'}")
    thr = await hold.unlikely_threshold(EN)
    print(f"threshold (en): {thr}  (below it, LiveKit waits up to max_delay)")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
