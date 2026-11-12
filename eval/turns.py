"""Turn-taking test, part 1: do callers pause long enough inside a number to get cut off?

    python turns.py words      # word timings for every digit-string clip (Deepgram, batch)
    python turns.py analyse    # pauses inside numbers vs common endpointing settings

A voice agent decides the caller has finished when it hears a long enough
silence (the "endpointing" or "stop" setting, often 0.5-1.2 s). If a caller
pauses that long while reading a number, the agent replies with half a number.

We take word-level timestamps, find the span from the word before the first
digit (the "my number is" moment) to the last digit, and measure every silence
inside it.
"""
import asyncio
import csv
import json
import sys
import urllib.request
from pathlib import Path

from bench import CLIPS, RESULTS, ROOT, DEEPGRAM_LANG, DIGITS, REPEAT, DEV_DIGITS, secret, truth_strings

WORDS = RESULTS / "deepgram_words.jsonl"
SETTINGS = (0.5, 0.8, 1.2)


def digit_clips():
    rows = list(csv.DictReader(open(ROOT / "data" / "clips.csv", encoding="utf-8")))
    return [r for r in rows if truth_strings(r["text"])]


def fetch_words(clip_id, dataset, key):
    # Batch transcription with word timings. No smart formatting, so each spoken
    # digit stays its own word with its own start and end time.
    url = (f"https://api.deepgram.com/v1/listen?model=nova-3&language={DEEPGRAM_LANG[dataset]}"
           "&smart_format=false&punctuate=false&numerals=false")
    req = urllib.request.Request(url, data=(CLIPS / f"{clip_id}.wav").read_bytes(), method="POST",
                                 headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    words = data["results"]["channels"][0]["alternatives"][0].get("words", [])
    return [{"w": w["word"], "s": w["start"], "e": w["end"]} for w in words]


async def words_cmd():
    key = secret("DEEPGRAM_API_KEY")
    done = set()
    if WORDS.exists():
        done = {json.loads(l)["id"] for l in WORDS.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in digit_clips() if r["id"] not in done]
    print(f"{len(todo)} clips to fetch ({len(done)} done)", flush=True)
    sem = asyncio.Semaphore(8)

    async def one(r):
        async with sem:
            for attempt in range(3):
                try:
                    words = await asyncio.to_thread(fetch_words, r["id"], r["dataset"], key)
                    break
                except Exception as e:
                    words = None
                    err = f"{type(e).__name__}: {e}"[:120]
                    await asyncio.sleep(3 * (attempt + 1))
            rec = {"id": r["id"], "words": words, "error": "" if words is not None else err}
            with open(WORDS, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(r) for r in todo))
    print("done")


def digit_of(word):
    t = word.lower().strip(".,?!।").translate(DEV_DIGITS)
    if t.isdigit():
        return t
    if t in DIGITS:
        return DIGITS[t]
    if t in REPEAT:
        return "rep"
    return None


def number_spans(words):
    """Runs of 4+ consecutive digit words, each extended back by one word (the lead-in)."""
    spans, i = [], 0
    while i < len(words):
        if digit_of(words[i]["w"]):
            j = i
            while j + 1 < len(words) and digit_of(words[j + 1]["w"]):
                j += 1
            n_digits = sum(len(d) for d in (digit_of(w["w"]) for w in words[i:j + 1]) if d != "rep")
            if n_digits >= 5:
                spans.append((max(i - 1, 0), i, j))
            i = j + 1
        else:
            i += 1
    return spans


def speech_windows(clip_id):
    """Silero VAD over the clip in 32 ms windows -> list of (start_s, is_speech)."""
    import numpy as np
    import soundfile as sf
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    global _VAD
    if "_VAD" not in globals():
        _VAD = SileroVADAnalyzer(sample_rate=8000)
        _VAD.set_sample_rate(8000)
    pcm, _ = sf.read(CLIPS / f"{clip_id}.wav", dtype="int16")
    out = []
    for i in range(0, len(pcm) - 256 + 1, 256):
        conf = float(np.asarray(_VAD.voice_confidence(pcm[i:i + 256].tobytes())).reshape(-1)[0])
        out.append((i / 8000, conf >= 0.5))
    return out


def silences_between(windows, t0, t1):
    """Silences inside [t0, t1] that have speech on both sides: (start, length)."""
    inside = [(t, sp) for t, sp in windows if t0 <= t < t1]
    gaps, run_start, seen_speech = [], None, False
    for t, sp in inside:
        if sp:
            if run_start is not None and seen_speech:
                gaps.append((run_start, round(t - run_start, 3)))
            run_start, seen_speech = None, True
        elif run_start is None:
            run_start = t
    return gaps


def analyse():
    clips = {r["id"]: r for r in digit_clips()}
    recs = [json.loads(l) for l in WORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for rec in recs:
        if rec["error"] or not rec["words"]:
            continue
        w = rec["words"]
        spans = number_spans(w)
        if not spans:
            continue
        win = speech_windows(rec["id"])
        for lead, first, last in spans:
            boundary = w[first]["s"]  # where the lead-in ends and the number begins
            gaps = silences_between(win, w[lead]["s"], w[last]["e"])
            before = max([g for t, g in gaps if t <= boundary <= t + g] or [0.0])
            inside = max([g for t, g in gaps if t > boundary] or [0.0])
            rows.append({
                "id": rec["id"], "dataset": clips[rec["id"]]["dataset"],
                "digits": "".join(d for d in (digit_of(x["w"]) for x in w[first:last + 1]) if d != "rep"),
                "lead_in_word": w[lead]["w"] if lead < first else "",
                "gap_before": before, "max_gap_inside": inside,
                "worst_gap": max(before, inside),
                "duration": round(w[last]["e"] - w[first]["s"], 2),
            })
    with open(RESULTS / "turn_gaps.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"{len(rows)} numbers read out, from {len({r['id'] for r in rows})} clips")
    print()
    print("Share of numbers with a silence at least this long (the agent would cut in):")
    print(f"{'':10s}" + "".join(f"{f'>= {s} s':>18s}" for s in SETTINGS))
    for ds, label in (("svarah", "English"), ("lahaja", "Hindi"), (None, "All")):
        sub = [r for r in rows if ds is None or r["dataset"] == ds]
        cells = []
        for s in SETTINGS:
            n = sum(r["worst_gap"] >= s for r in sub)
            cells.append(f"{n}/{len(sub)} = {100 * n / len(sub):.0f}%")
        print(f"{label:10s}" + "".join(f"{c:>18s}" for c in cells))
    print()
    print("Where the long silences are (all numbers):")
    for s in SETTINGS:
        b = sum(r["gap_before"] >= s for r in rows)
        i = sum(r["max_gap_inside"] >= s for r in rows)
        print(f"  >= {s} s: before the number {b}, inside the number {i}")
    gaps = sorted(r["worst_gap"] for r in rows)
    q = lambda p: gaps[min(len(gaps) - 1, int(p * len(gaps)))]
    print()
    print(f"Worst silence per number: median {q(0.5):.2f} s, 90th percentile {q(0.9):.2f} s, max {gaps[-1]:.2f} s")


if __name__ == "__main__":
    if sys.argv[1:] == ["words"]:
        asyncio.run(words_cmd())
    elif sys.argv[1:] == ["analyse"]:
        analyse()
    else:
        print(__doc__)


# ---------------------------------------------------------------- part 2: Smart Turn

def smart_turn_cmd():
    """At every pause of 0.2 s or more around a number, ask Smart Turn v3 whether the
    caller has finished, the way Pipecat does: the audio from the start of the turn up
    to 0.2 s into the silence (Pipecat's stop_secs when Smart Turn is used)."""
    import numpy as np
    import soundfile as sf
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    st = LocalSmartTurnAnalyzerV3()
    st.set_sample_rate(8000)
    STOP = 0.2

    def p_complete(pcm, upto):
        audio = pcm[: int(upto * 8000)].astype(np.float32) / 32768.0
        return float(st._predict_endpoint(audio)["probability"])

    clips = {r["id"]: r for r in digit_clips()}
    recs = [json.loads(l) for l in WORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, controls = [], []
    for rec in recs:
        if rec["error"] or not rec["words"]:
            continue
        w = rec["words"]
        spans = number_spans(w)
        if not spans:
            continue
        pcm, _ = sf.read(CLIPS / f"{rec['id']}.wav", dtype="int16")
        win = speech_windows(rec["id"])
        # Control: the real end of what the speaker said, plus 0.2 s of silence.
        speech_t = [t for t, sp in win if sp]
        if speech_t:
            end = speech_t[-1] + 0.032
            padded = np.concatenate([pcm[: int(end * 8000)], np.zeros(int(STOP * 8000), dtype=np.int16)])
            controls.append(p_complete(padded, len(padded) / 8000))
        for lead, first, last in spans:
            gaps = [(t, g) for t, g in silences_between(win, w[lead]["s"], w[last]["e"]) if g >= STOP]
            probs = [(round(g, 2), round(p_complete(pcm, t + STOP), 3)) for t, g in gaps]
            rows.append({
                "id": rec["id"], "dataset": clips[rec["id"]]["dataset"],
                "pauses": len(probs),
                "max_p_complete": max([p for _, p in probs] or [0.0]),
                "cut_off": any(p > 0.5 for _, p in probs),
                "detail": json.dumps(probs),
            })
    with open(RESULTS / "smart_turn.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"Smart Turn v3 at every pause of {STOP}+ s around {len(rows)} numbers")
    for ds, label in (("svarah", "English"), ("lahaja", "Hindi"), (None, "All")):
        sub = [r for r in rows if ds is None or r["dataset"] == ds]
        n = sum(r["cut_off"] for r in sub)
        np_ = sum(r["pauses"] for r in sub)
        print(f"  {label:8s} wrongly said 'finished' mid-number: {n}/{len(sub)} = {100 * n / len(sub):.0f}%   ({np_} pauses checked)")
    ok = sum(p > 0.5 for p in controls)
    print(f"Control, real end of speech: said 'finished' {ok}/{len(controls)} = {100 * ok / len(controls):.0f}%")


if __name__ == "__main__" and sys.argv[1:] == ["smartturn"]:
    smart_turn_cmd()
