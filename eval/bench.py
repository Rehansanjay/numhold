"""India Voice Benchmark, v0 runner.

    python bench.py prep                      # clips -> data/clips/*.wav (8 kHz, phone quality)
    python bench.py run deepgram --limit 20   # stream clips to a service in real time
    python bench.py score deepgram            # exact-value scoring, per dataset and category

Audio is resampled to 8 kHz and sent as mu-law, the format phone calls use, in
20 ms chunks at real-time pace, because that is how a voice agent feeds a
streaming speech-to-text service.
"""
import argparse
import asyncio
import csv
import io
import json
import os
import re
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).parent
CLIPS = ROOT / "data" / "clips"
RESULTS = ROOT / "results"
CATEGORIES = ("digit_string", "amount", "datetime")


def secret(name):
    """Read a key from the environment or a .env file next to this script. Never printed."""
    if os.environ.get(name):
        return os.environ[name]
    for path in (ROOT / ".env",):
        if path.exists():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                m = re.match(rf"\s*{name}\s*=\s*\"?([^\"\s]+)", line)
                if m:
                    return m.group(1)
    sys.exit(f"{name} not found in environment or .env files")


# ---------------------------------------------------------------- prep

def clip_id(row):
    return f"{row['dataset']}_{Path(row['file']).stem.split('-')[1]}_{int(row['row']):05d}"


def load_manifest():
    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8")))
    keep = [r for r in rows if set(r["tags"].split("|")) & set(CATEGORIES)]
    for r in keep:
        r["id"] = clip_id(r)
    return keep


def prep():
    CLIPS.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    by_file = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)
    done = 0
    for f, items in by_file.items():
        table = pq.read_table(ROOT / f, columns=["audio_filepath"])
        for r in items:
            out = CLIPS / f"{r['id']}.wav"
            if out.exists():
                continue
            audio = table.column("audio_filepath")[int(r["row"])].as_py()
            data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            g = gcd(sr, 8000)
            data = resample_poly(data, 8000 // g, sr // g)
            sf.write(out, np.clip(data, -1, 1), 8000, subtype="PCM_16")
            done += 1
    with open(ROOT / "data" / "clips.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "dataset", "tags", "duration", "text"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("id", "dataset", "tags", "duration", "text")})
    print(f"{len(rows)} clips listed, {done} new wav files in {CLIPS}")


def mulaw(pcm16):
    """16-bit PCM -> G.711 mu-law bytes (audioop is gone in Python 3.13+)."""
    x = pcm16.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0)
    x = np.minimum(np.abs(x), 32635) + 132
    exp = np.floor(np.log2(x)).astype(np.int32) - 7
    exp = np.clip(exp, 0, 7)
    mant = (x >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


# ---------------------------------------------------------------- services

DEEPGRAM_LANG = {"svarah": "en-IN", "lahaja": "hi"}


async def deepgram(clip, dataset, key):
    import websockets

    # Close to Pipecat's DeepgramSTTService defaults: nova-3, smart_format, interim results.
    params = (
        f"model=nova-3&language={DEEPGRAM_LANG[dataset]}&encoding=mulaw&sample_rate=8000"
        "&channels=1&smart_format=true&punctuate=true&interim_results=true&endpointing=300"
    )
    url = f"wss://api.deepgram.com/v1/listen?{params}"
    pcm, _ = sf.read(clip, dtype="int16")
    audio = mulaw(pcm)
    finals = []
    async with websockets.connect(url, additional_headers={"Authorization": f"Token {key}"}) as ws:

        async def send():
            for i in range(0, len(audio), 160):  # 20 ms at 8 kHz
                await ws.send(audio[i:i + 160])
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({"type": "CloseStream"}))

        sender = asyncio.create_task(send())
        async for msg in ws:
            m = json.loads(msg)
            if m.get("type") == "Results" and m.get("is_final"):
                alt = m["channel"]["alternatives"][0]["transcript"]
                if alt:
                    finals.append(alt)
        await sender
    return " ".join(finals)


SARVAM_LANG = {"svarah": "en-IN", "lahaja": "hi-IN"}


async def sarvam(clip, dataset, key):
    """Sarvam streaming, set up the way Pipecat's SarvamSTTService does it:
    saaras:v4, mode transcribe, raw PCM sent as audio/wav, flush at end of speech."""
    import base64

    from sarvamai import AsyncSarvamAI

    pcm, _ = sf.read(clip, dtype="int16")
    # A caller's audio arrives, then the agent's VAD notices the end of speech and flushes.
    pcm = np.concatenate([pcm, np.zeros(8000 // 2, dtype=np.int16)])
    raw = pcm.tobytes()
    client = AsyncSarvamAI(api_subscription_key=key)
    texts = []
    async with client.speech_to_text_streaming.connect(
        model="saaras:v4", mode="transcribe", language_code=SARVAM_LANG[dataset],
        sample_rate="8000", flush_signal="true",
    ) as ws:
        last = [time.time()]

        async def receive():
            while True:
                msg = await ws.recv()
                if getattr(msg, "type", None) == "data":
                    t = msg.data.transcript
                    if t and t.strip():
                        texts.append(t.strip())
                    last[0] = time.time()

        rx = asyncio.create_task(receive())
        for i in range(0, len(raw), 320):  # 20 ms of 16-bit audio at 8 kHz
            await ws.transcribe(audio=base64.b64encode(raw[i:i + 320]).decode(),
                                encoding="audio/wav", sample_rate=8000)
            await asyncio.sleep(0.02)
        await ws.flush()
        flushed = time.time()
        last[0] = flushed
        # Wait for the final transcript: stop after 2 s of quiet, or 10 s at most.
        while time.time() - last[0] < 2.0 and time.time() - flushed < 10:
            await asyncio.sleep(0.1)
        rx.cancel()
    return " ".join(texts)


SERVICES = {"deepgram": (deepgram, "DEEPGRAM_API_KEY"), "sarvam": (sarvam, "SARVAM_API_KEY")}


async def run(service, limit, concurrency):
    fn, key_name = SERVICES[service]
    key = secret(key_name)
    rows = list(csv.DictReader(open(ROOT / "data" / "clips.csv", encoding="utf-8")))
    if limit:
        # Spread a small run across both datasets.
        rows = [r for r in rows if r["dataset"] == "svarah"][: limit // 2] + \
               [r for r in rows if r["dataset"] == "lahaja"][: limit - limit // 2]
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{service}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{service}: {len(todo)} clips to run ({len(done)} already done)", flush=True)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    count = 0

    async def one(r):
        nonlocal count
        async with sem:
            t0 = time.time()
            # Connection drops and rate limits are not the service mishearing,
            # so retry them a few times with a growing pause before giving up.
            for attempt in range(4):
                try:
                    hyp, err = await fn(CLIPS / f"{r['id']}.wav", r["dataset"], key), ""
                    break
                except Exception as e:  # keep going; record the failure
                    hyp, err = "", f"{type(e).__name__}: {str(e)[:120]}"
                    await asyncio.sleep(5 * (attempt + 1))
            rec = {"id": r["id"], "service": service, "hyp": hyp, "error": err,
                   "seconds": round(time.time() - t0, 2)}
            async with lock:
                with open(out_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
                if count % 25 == 0 or count == len(todo):
                    print(f"  {count}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(r) for r in todo))


# ---------------------------------------------------------------- scoring

DIGITS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "शून्य": "0", "ज़ीरो": "0", "जीरो": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पांच": "5", "पाँच": "5",
    "छह": "6", "छः": "6", "छे": "6", "चे": "6", "छै": "6", "सात": "7", "आठ": "8", "नौ": "9",
}
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
REPEAT = {"double": 2, "triple": 3, "डबल": 2, "ट्रिपल": 3}


def digit_stream(text):
    """Turn spoken digits into digit characters and join runs of digits.

    "nine one five three zero nine" -> "915309"; "double five" -> "55";
    "98400 12345" -> "9840012345". Only single-digit words are converted, so
    this scores numbers read digit by digit, which is how callers give phone
    numbers, PIN codes and account numbers.
    """
    text = text.lower().translate(DEV_DIGITS)
    # Dates (11/09/2021) and comma amounts (2,000 / 1,50,000) belong to other categories.
    text = re.sub(r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\b", " ", text)
    text = re.sub(r"\b\d{1,3}(?:,\d{2,3})+\b", " ", text)
    # Split letters from digits: "otp453501" -> "otp 453501".
    text = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", text)
    text = text.replace("।", " ").replace("॥", " ")  # Hindi full stops
    tokens = re.findall(r"[\wऀ-ॿ]+", text)
    out, rep = [], 1
    for t in tokens:
        if t in REPEAT:
            rep = REPEAT[t]
            continue
        if t.isdigit():
            out.append(t * rep if len(t) == 1 else t)
        elif t in DIGITS:
            out.append(DIGITS[t] * rep)
        else:
            out.append(" ")
        rep = 1
    joined = "".join(x if x != " " else "|" for x in out)
    return [run for run in joined.split("|") if run]


DATE_WORDS = re.compile(r"\b(year|years|date|birth|dob|january|february|march|april|may|june|july|august|september|october|november|december)\b|वर्ष|साल|तारीख|जन्म")


def truth_strings(text):
    # Clips that talk about dates or years are scored in the datetime category instead,
    # because "twenty twenty" or "11/09/2021" are not digit strings.
    if DATE_WORDS.search(text.lower()) or re.search(r"(?<!\d)(19|20)\d\d(?!\d)|nineteen|twenty twenty|two thousand|academic|दो हज़ार|उन्नीस सौ", text.lower()):
        return []
    # Round figures like 70000 are amounts, scored in the amount category.
    return [s for s in digit_stream(text) if len(s) >= 5 and not re.fullmatch(r"[1-9]\d?0{3,5}", s)]


def score(service):
    clips = {r["id"]: r for r in csv.DictReader(open(ROOT / "data" / "clips.csv", encoding="utf-8"))}
    recs = [json.loads(l) for l in (RESULTS / f"{service}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, stats = [], {}
    for rec in recs:
        c = clips[rec["id"]]
        truths = truth_strings(c["text"])
        if not truths:
            continue
        hyp_runs = digit_stream(rec["hyp"])
        hyp_all = "|".join(hyp_runs)
        ok = all(t in hyp_runs or t in hyp_all.replace("|", "") for t in truths) and not rec["error"]
        key = c["dataset"]
        s = stats.setdefault(key, [0, 0])
        s[0] += ok
        s[1] += 1
        rows.append({"id": rec["id"], "dataset": c["dataset"], "ok": ok, "truth": " ".join(truths),
                     "heard": " ".join(hyp_runs), "hyp": rec["hyp"], "text": c["text"], "error": rec["error"]})
    with open(RESULTS / f"{service}_digits_scored.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{service}: long digit strings (5+ digits) captured exactly")
    for k, (ok, n) in stats.items():
        print(f"  {k:8s} {ok}/{n} = {100 * ok / n:.0f}%")
    errs = sum(1 for r in recs if r["error"])
    if errs:
        print(f"  ({errs} clips failed to run; counted as wrong)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    r = sub.add_parser("run")
    r.add_argument("service", choices=SERVICES)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=10)
    s = sub.add_parser("score")
    s.add_argument("service", choices=SERVICES)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "run":
        asyncio.run(run(a.service, a.limit, a.concurrency))
    else:
        score(a.service)


if __name__ == "__main__":
    main()
