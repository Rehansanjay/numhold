"""Score the number-aware hold against the three turn-taking setups, on the same pauses.

A pause cuts the caller off when the base setup would end the turn there AND the
hold does not apply (or the silence outlasts MAX_HOLD_S).

Cost: at the real end of what each speaker said, how often does the hold make the
agent wait when the caller had actually finished?

    python fix_eval.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from numhold import MAX_HOLD_S, number_in_progress  # noqa: E402

R = Path(__file__).parent / "results"
LK_THR = {"svarah": 0.011, "lahaja": 0.0398}


def load():
    pauses = [json.loads(l) for l in (R / "lk_pauses.jsonl").read_text(encoding="utf-8").splitlines()]
    st = list(csv.DictReader(open(R / "smart_turn.csv", encoding="utf-8")))
    lk = list(csv.DictReader(open(R / "lk_turns.csv", encoding="utf-8")))
    assert len(pauses) == len(st) == len(lk)
    out = []
    for rec, s, l in zip(pauses, st, lk):
        assert rec["id"] == s["id"] == l["id"]
        st_p = json.loads(s["detail"])  # one per pause
        lk_p = iter(json.loads(l["detail"]))  # one per pause that had text
        items = []
        for k, p in enumerate(rec["pauses"]):
            lkp = next(lk_p)[1] if p["text"] else 0.0
            hold, why = number_in_progress(p["text"])
            items.append({"gap": p["gap"], "st": st_p[k][1], "lk": lkp, "hold": hold, "why": why,
                          "text": p["text"]})
        out.append({"id": rec["id"], "dataset": rec["dataset"], "span": rec["span"],
                    "full": rec["full"], "pauses": items})
    return out


SETUPS = {
    "Silence 0.5 s": lambda p, ds: p["gap"] >= 0.5,
    "Pipecat Smart Turn": lambda p, ds: p["gap"] >= 0.2 and p["st"] > 0.5,
    "LiveKit defaults": lambda p, ds: p["gap"] >= 0.55 and p["lk"] >= LK_THR[ds],
    "LiveKit tuned fast": lambda p, ds: p["gap"] >= 0.2 and p["lk"] >= LK_THR[ds],
}


def main():
    nums = load()
    print(f"{len(nums)} numbers. Cut off mid-number, before -> with the number-aware hold:\n")
    print(f"{'':22s}{'English':>20s}{'Hindi':>20s}{'All':>20s}")
    for name, ends_turn in SETUPS.items():
        cells = []
        for ds in ("svarah", "lahaja", None):
            sub = [n for n in nums if ds is None or n["dataset"] == ds]
            before = sum(any(ends_turn(p, n["dataset"]) for p in n["pauses"]) for n in sub)
            after = sum(any(ends_turn(p, n["dataset"]) and (not p["hold"] or p["gap"] >= MAX_HOLD_S)
                            for p in n["pauses"]) for n in sub)
            cells.append(f"{100*before/len(sub):.0f}% -> {100*after/len(sub):.0f}%")
        print(f"{name:22s}" + "".join(f"{c:>20s}" for c in cells))

    # What still gets through, so the rules can be judged honestly.
    missed = [(n, p) for n in nums for p in n["pauses"]
              if SETUPS["Pipecat Smart Turn"](p, n["dataset"]) and not p["hold"]]
    print(f"\nPauses Smart Turn still ends with the hold on: {len(missed)}. Examples:")
    for n, p in missed[:8]:
        print(f"  [{n['dataset']}] ...{p['text'][-70:]}")

    # Cost: does the hold fire at the real end of speech?
    ends = [n for n in nums if n["span"] == 0]
    held = [n for n in ends if number_in_progress(n["full"])[0]]
    print(f"\nCost at the real end of speech: held {len(held)}/{len(ends)} = {100*len(held)/len(ends):.0f}% "
          f"(each adds up to {MAX_HOLD_S:.1f} s before the agent replies)")
    for n in held[:6]:
        print(f"  held: ...{n['full'][-70:]}  ({number_in_progress(n['full'])[1]})")


if __name__ == "__main__":
    main()
