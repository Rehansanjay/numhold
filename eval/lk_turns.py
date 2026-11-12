"""Turn-taking test, part 3: LiveKit's multilingual end-of-turn model at the pauses inside numbers.

LiveKit's turn detector reads the TRANSCRIPT, not the audio. In LiveKit Agents the
VAD first has to hear `min_silence_duration` of silence (Silero default 0.55 s);
then the model scores the transcript so far. If the probability is above the
language's threshold, the agent replies after `min_delay` (default 0.5 s);
otherwise it waits up to `max_delay` (3 s).

Step 1 (system Python, has Silero):      python lk_turns.py export
Step 2 (env with livekit turn detector): python lk_turns.py score
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PAUSES = ROOT / "results" / "lk_pauses.jsonl"
PROMPT = "Sure, I can help with that. What details can you share?"


def export():
    from turns import WORDS, digit_clips, number_spans, silences_between, speech_windows

    clips = {r["id"]: r for r in digit_clips()}
    out = open(PAUSES, "w", encoding="utf-8")
    n = 0
    for line in WORDS.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        w = rec["words"]
        if rec["error"] or not w:
            continue
        spans = number_spans(w)
        if not spans:
            continue
        win = speech_windows(rec["id"])
        for k, (lead, first, last) in enumerate(spans):
            pauses = []
            for t, g in silences_between(win, w[lead]["s"], w[last]["e"]):
                if g < 0.2:
                    continue
                # what the STT has delivered by the time the caller goes quiet
                said = " ".join(x["w"] for x in w if x["s"] < t)
                pauses.append({"t": t, "gap": g, "text": said})
            out.write(json.dumps({"id": rec["id"], "span": k, "dataset": clips[rec["id"]]["dataset"],
                                  "pauses": pauses,
                                  "full": " ".join(x["w"] for x in w)}, ensure_ascii=False) + "\n")
            n += 1
    print(f"exported {n} numbers")


def score():
    from livekit.plugins.turn_detector.multilingual import _EUORunnerMultilingual
    from huggingface_hub import hf_hub_download
    from livekit.plugins.turn_detector.models import HG_MODEL, MODEL_REVISIONS

    _EUORunnerMultilingual._download_files()
    runner = _EUORunnerMultilingual()
    runner.initialize()
    langs = json.load(open(hf_hub_download(HG_MODEL, "languages.json",
                                           revision=MODEL_REVISIONS["multilingual"]), encoding="utf-8"))
    thr = {"svarah": langs["en"]["threshold"], "lahaja": langs["hi"]["threshold"]}
    print("thresholds:", thr)

    def p_end(text):
        ctx = [{"role": "assistant", "content": PROMPT}, {"role": "user", "content": text}]
        return json.loads(runner.run(json.dumps({"chat_ctx": ctx}).encode()))["eou_probability"]

    rows, controls = [], []
    for line in PAUSES.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        t = thr[rec["dataset"]]
        scored = [(round(p["gap"], 2), round(p_end(p["text"]), 3)) for p in rec["pauses"] if p["text"]]
        # LiveKit only asks the model once the VAD has heard min_silence_duration (0.55 s)
        default_cut = any(g >= 0.55 and pr >= t for g, pr in scored)
        model_wrong = any(pr >= t for g, pr in scored)
        if rec["span"] == 0:
            controls.append(p_end(rec["full"]) >= t)
        rows.append({"id": rec["id"], "dataset": rec["dataset"], "default_cut": default_cut,
                     "model_wrong_any_pause": model_wrong, "detail": json.dumps(scored)})
    with open(ROOT / "results" / "lk_turns.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"LiveKit multilingual turn detector, {len(rows)} numbers")
    for ds, label in (("svarah", "English"), ("lahaja", "Hindi"), (None, "All")):
        sub = [r for r in rows if ds is None or r["dataset"] == ds]
        a = sum(r["default_cut"] for r in sub)
        b = sum(r["model_wrong_any_pause"] for r in sub)
        print(f"  {label:8s} cut off with LiveKit defaults: {a}/{len(sub)} = {100*a/len(sub):.0f}%"
              f"   | model says 'finished' at some 0.2 s+ pause: {b}/{len(sub)} = {100*b/len(sub):.0f}%")
    ok = sum(controls)
    print(f"Control, full transcript: says 'finished' {ok}/{len(controls)} = {100*ok/len(controls):.0f}%")


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1]]()
