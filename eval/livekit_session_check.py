"""End-to-end check of livekit_numhold inside a real LiveKit AgentSession.

Uses the fake VAD / STT / LLM / TTS from LiveKit's own test suite, so it needs a
checkout of livekit/agents (the version you have installed) next to this repo,
or LIVEKIT_AGENTS_REPO pointing at one:

    python eval/livekit_session_check.py

A caller says "my mobile number is nine eight four zero one", pauses 1 s, then
"two three four five six". The session is tuned to reply fast (end of speech
after 0.5 s of silence, then a 0.2 s endpointing delay), so without the hold it
answers half the number. With NumberHoldTurnDetector it waits for the rest.
Then a turn with no number, to show the hold leaves it alone.
"""
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, os.environ.get("LIVEKIT_AGENTS_REPO", str(HERE.parents[1] / "livekit-agents")))

from livekit.agents import Agent  # noqa: E402

from livekit_numhold import NumberHoldTurnDetector  # noqa: E402
from tests.fake_session import FakeActions, create_session, run_session  # noqa: E402

FIRST = "Hi, my mobile number is nine eight four zero one"
REST = "two three four five six."
PLAIN = "I want to change my booking please."
FAST = {"min_delay": 0.2, "max_delay": 2.0}


def actions_number() -> FakeActions:
    a = FakeActions()
    a.add_user_speech(0.5, 3.0, FIRST, stt_delay=0.2)
    a.add_user_speech(4.0, 5.2, REST, stt_delay=0.2)  # 1.0 s pause in the middle
    a.add_llm("I heard 98401. What's the rest?", input=FIRST)
    a.add_tts(1.5, input="I heard 98401. What's the rest?")
    a.add_llm("Got it, 98401 23456.", input=f"{FIRST} {REST}")
    a.add_tts(1.5, input="Got it, 98401 23456.")
    a.add_llm("Sure.", input=REST)
    a.add_tts(0.5, input="Sure.")
    return a


def actions_plain() -> FakeActions:
    a = FakeActions()
    a.add_user_speech(0.5, 2.5, PLAIN, stt_delay=0.2)
    a.add_llm("Sure, which booking?", input=PLAIN)
    a.add_tts(1.0, input="Sure, which booking?")
    return a


async def run(actions: FakeActions, detector) -> list[str]:
    """The caller's turns, as the agent received them."""
    turn_handling = {"endpointing": dict(FAST)}
    if detector is not None:
        turn_handling["turn_detection"] = detector
    session = create_session(actions, turn_handling=turn_handling)
    user_turns = []

    def added(ev):
        if ev.item.type == "message" and ev.item.role == "user":
            user_turns.append(ev.item.text_content)

    session.on("conversation_item_added", added)
    await asyncio.wait_for(run_session(session, Agent(instructions="test"), drain_delay=4), 30)
    return user_turns


async def main() -> None:
    held = []
    ok = True
    for name, det in (("without the hold", None),
                      ("with NumberHoldTurnDetector", NumberHoldTurnDetector(on_event=held.append))):
        turns = await run(actions_number(), det)
        print(f"{name}: caller's turns as the agent got them:")
        for t in turns:
            print(f"    {t!r}")
        whole = len(turns) == 1 and "zero one" in turns[0] and "two three" in turns[0]
        if det is None:
            ok &= len(turns) == 2  # the half-number turn
        else:
            ok &= whole
    print("hold events:", held)

    held.clear()
    turns = await run(actions_plain(), NumberHoldTurnDetector(on_event=held.append))
    print(f"no number, with the hold: {turns}, hold events: {held}")
    ok &= turns == [PLAIN] and not held
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
