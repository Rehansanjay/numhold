# numhold

Don't end the caller's turn in the middle of a number. For Pipecat and LiveKit Agents.

People read phone numbers, PIN codes and account numbers in chunks: "nine eight four zero one ... two three four five six". A voice agent tuned to reply quickly hears the pause, decides the caller has finished, and answers with half a number.

numhold is a small check that sits in front of whatever turn detection you use. When the detector says "finished", numhold looks at the transcript so far, and at what the agent just asked. If a number is clearly still coming, it waits until the caller speaks again or 2 s of silence pass. Turns with no number in them are not touched.

## Listen

Same Pipecat bot (Smart Turn v3, Pipecat 1.15), same caller audio, a 1 s pause in the middle of the number:

- [`audio/smart_turn_without_hold.wav`](audio/smart_turn_without_hold.wav): the turn ends after "nine eight four zero one". The agent says "I heard 98401. Now please repeat your PIN code."
- [`audio/smart_turn_with_hold.wav`](audio/smart_turn_with_hold.wav): the agent waits, then reads back all ten digits.

## How often it happens

341 numbers read out by real speakers (188 Indian English, 153 Hindi). At every pause inside or just before the number, each setup was asked whether the caller had finished. A number counts as cut off if any of its pauses ends the turn.

| Turn detection | English | Hindi | All |
|---|---|---|---|
| Silence timer, 0.5 s | 11% -> 3% | 27% -> 9% | 18% -> 6% |
| Pipecat Smart Turn v3 | 28% -> 5% | 33% -> 12% | 30% -> 8% |
| LiveKit turn detector, default settings | 6% -> 2% | 9% -> 4% | 8% -> 3% |
| LiveKit turn detector, tuned fast | 40% -> 9% | 38% -> 16% | 39% -> 12% |

Left of the arrow: cut off without numhold. Right: with it.

**What it costs:** at the real end of what each speaker said, numhold held 18 of 312 turns (6%), adding up to 2 s before the agent replied.

**Control:** at the real end of speech, Smart Turn correctly said "finished" 87% of the time, so the model is working normally on this audio. The mid-number cut-offs are pauses that sound finished.

## Use it with Pipecat

```python
from pipecat_numhold import AgentTextRecorder, NumberHoldSmartTurn

hold = NumberHoldSmartTurn(turn_analyzer=LocalSmartTurnAnalyzerV3())
aggregators = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
    user_turn_strategies=UserTurnStrategies(stop=[hold])))

pipeline = Pipeline([transport.input(), stt, aggregators.user(), llm, tts,
                     AgentTextRecorder(hold), transport.output(), aggregators.assistant()])
```

`NumberHoldSpeechTimeout(user_speech_timeout=...)` does the same over a plain silence timer. `AgentTextRecorder` tells the hold what the agent just asked. On a real call people answer "what's your mobile number?" with the digits alone, so the expected length of the number is in the question.

## Use it with LiveKit Agents

```python
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit_numhold import NumberHoldTurnDetector

session = AgentSession(
    ...,
    turn_handling={
        "turn_detection": NumberHoldTurnDetector(MultilingualModel()),
        "endpointing": {"min_delay": 0.3, "max_delay": 2.0},
    },
)
```

When LiveKit asks whether the caller has finished and a number is clearly unfinished, it answers "not finished", so LiveKit waits up to `max_delay` instead of `min_delay`; the caller carrying on cancels the wait as usual. Otherwise the wrapped model answers, unchanged. LiveKit passes the turn detector the conversation so far, so the agent's question is picked up without any extra wiring. `NumberHoldTurnDetector()` with no model works for agents that end turns on silence alone. It works with the turn detectors that read the transcript (MultilingualModel, EnglishModel), not the streaming audio ones. Tested with livekit-agents 1.8.11.

Two checks, both run with `python` from a LiveKit environment:

- `eval/livekit_session_check.py` runs a real `AgentSession` with LiveKit's own fake VAD, STT, LLM and TTS (needs a livekit/agents checkout beside this repo). A caller says "my mobile number is nine eight four zero one", pauses 1 s, then "two three four five six". Tuned to reply fast, the agent gets two turns without the hold and answers half the number. With `NumberHoldTurnDetector` it gets one turn with all ten digits. A turn with no number is left alone.
- `eval/livekit_model_check.py` wraps the real MultilingualModel. The model alone already waits after "nine eight four zero one" (0.001, below its English threshold of 0.011), but says "finished" after the PIN code's "six zero zero" (0.075). The wrapper holds there, and returns exactly the model's answer everywhere else.

## The rule on its own

The rule itself is [`numhold.py`](numhold.py), plain Python with no dependencies, so it can sit in front of any framework's turn detection:

```python
from numhold import number_in_progress

number_in_progress("my phone number is nine eight four zero one")
# (True, '5 of 10+ digits')
number_in_progress("9840012345", asked="What's your mobile number?")
# (False, '')
```

## Limits

- The clips are people **reading prompted sentences** (AI4Bharat Svarah and Lahaja), resampled to 8 kHz. They are not real phone calls. The phone band isn't the cause: on the original 16 kHz audio Smart Turn still cuts off 27% of numbers (English 24%, Hindi 30%). Callers reading a number off a card probably pause longer than this, so real cut-off rates may be higher.
- English and Hindi only. It knows mobile numbers, PIN codes, OTPs, Aadhaar, UAN, IFSC, and generic "number / ID / account". It works on digits said one at a time or written as digits. It does not hold for amounts ("twenty thousand").
- It reads the transcript, so it can only hold if the speech-to-text has caught up with the words before the pause. In the live tests (Deepgram streaming) it had.
- Agents that wait a long time before replying (1.5 s and more) rarely cut people off. They pay for it on every reply instead. numhold is for agents that want to reply fast.
- Smart Turn's numbers move a little with the onnxruntime version: 30% with onnxruntime 1.31, 32% with 1.24, on the same model file. Scores near the 0.5 threshold flip.

## Reproduce

Re-score from the included per-pause data. This takes a few seconds and needs no keys:

```
python eval/fix_eval.py
```

To rebuild the data from the audio:

1. Download the Svarah and Lahaja parquet files from Hugging Face (`ai4bharat/Svarah`, `ai4bharat/Lahaja`; you may have to accept the dataset terms) into `eval/data/svarah/data/` and `eval/data/lahaja/data/`.
2. Run, from `eval/`:

```
python scan.py               # find clips with numbers
python bench.py prep         # clips -> 8 kHz wav
python turns.py words        # word timings (Deepgram batch API, DEEPGRAM_API_KEY); included as results/deepgram_words.jsonl
python turns.py analyse      # pause lengths inside numbers -> results/turn_gaps.csv
python turns.py smartturn    # Smart Turn v3 at every pause (needs pipecat-ai)
python lk_turns.py export    # pauses for the LiveKit model
python lk_turns.py score     # LiveKit multilingual turn detector (needs livekit-plugins-turn-detector)
python fix_eval.py
```

## Credits

Audio and transcripts: AI4Bharat [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) and [Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja), CC BY 4.0.

MIT licensed. Rehan Sanjay.
