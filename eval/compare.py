"""Score one or more versions of the hold rule on the same 341 numbers, side by side.

    python compare.py                          # this repo's rule
    python compare.py ../../other-numhold      # also a second checkout (e.g. a branch)
    python compare.py --hint ../../other       # give the second rule the true digit count

Each positional argument is a directory containing a `numhold.py`. The rule in this
repo is always scored first, as the baseline.

--hint passes each number's real length to rules that accept it (a keyword named
num_digits, expected_digits or expected_length), the way LiveKit's GetDtmfTask knows
`num_digits` before the caller starts. Rules without such a keyword are scored
without it.

Prints, per setup, the share of numbers cut off mid-number, and the cost: how often
the rule holds at the real end of what the speaker said.
"""
import argparse
import csv
import importlib.util
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = Path(__file__).parent / "results"
LK_THR = {"svarah": 0.011, "lahaja": 0.0398}
HINT_KWARGS = ("num_digits", "expected_digits", "expected_length")

SETUPS = {
    "Silence 0.5 s": lambda p, ds: p["gap"] >= 0.5,
    "Pipecat Smart Turn": lambda p, ds: p["gap"] >= 0.2 and p["st"] > 0.5,
    "LiveKit defaults": lambda p, ds: p["gap"] >= 0.55 and p["lk"] >= LK_THR[ds],
    "LiveKit tuned fast": lambda p, ds: p["gap"] >= 0.2 and p["lk"] >= LK_THR[ds],
}


def load_rule(directory, name):
    path = Path(directory).resolve() / "numhold.py"
    if not path.exists():
        sys.exit(f"no numhold.py in {directory}")
    spec = importlib.util.spec_from_file_location(f"numhold_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def true_length(base, text):
    """Length of the longest run of digits in the full transcript."""
    tokens = base._tokens(text)
    return max((len(base._trailing_digits(tokens[:end])[0]) for end in range(1, len(tokens) + 1)), default=0)


def load_data():
    pauses = [json.loads(l) for l in (R / "lk_pauses.jsonl").read_text(encoding="utf-8").splitlines()]
    st = list(csv.DictReader(open(R / "smart_turn.csv", encoding="utf-8")))
    lk = list(csv.DictReader(open(R / "lk_turns.csv", encoding="utf-8")))
    assert len(pauses) == len(st) == len(lk)
    out = []
    for rec, s, l in zip(pauses, st, lk):
        assert rec["id"] == s["id"] == l["id"]
        st_p = json.loads(s["detail"])
        lk_p = iter(json.loads(l["detail"]))
        items = [{"gap": p["gap"], "st": st_p[k][1], "lk": next(lk_p)[1] if p["text"] else 0.0, "text": p["text"]}
                 for k, p in enumerate(rec["pauses"])]
        out.append({"id": rec["id"], "dataset": rec["dataset"], "span": rec["span"],
                    "full": rec["full"], "pauses": items})
    return out


def scorer(mod, hint):
    fn = mod.number_in_progress
    params = inspect.signature(fn).parameters
    kw = next((k for k in HINT_KWARGS if k in params), None) if hint else None

    def holds(text, n_digits):
        return fn(text, **({kw: n_digits} if kw else {}))[0]
    return holds, kw


def score(nums, holds, max_hold):
    rows = {}
    for name, ends_turn in SETUPS.items():
        cells = []
        for ds in ("svarah", "lahaja", None):
            sub = [n for n in nums if ds is None or n["dataset"] == ds]
            before = sum(any(ends_turn(p, n["dataset"]) for p in n["pauses"]) for n in sub)
            after = sum(any(ends_turn(p, n["dataset"]) and (not holds(p["text"], n["len"]) or p["gap"] >= max_hold)
                            for p in n["pauses"]) for n in sub)
            cells.append((before, after, len(sub)))
        rows[name] = cells
    ends = [n for n in nums if n["span"] == 0]
    cost = sum(holds(n["full"], n["len"]) for n in ends), len(ends)
    return rows, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("others", nargs="*", help="directories holding another numhold.py")
    ap.add_argument("--hint", action="store_true", help="pass the true digit count to rules that accept it")
    args = ap.parse_args()

    base = load_rule(ROOT, "base")
    nums = load_data()
    for n in nums:
        n["len"] = true_length(base, n["full"])

    rules = [("this repo", base)] + [(Path(d).name, load_rule(d, f"r{i}")) for i, d in enumerate(args.others)]
    results = []
    for label, mod in rules:
        holds, kw = scorer(mod, args.hint)
        rows, cost = score(nums, holds, getattr(mod, "MAX_HOLD_S", 2.0))
        tag = f"{label} (hint via {kw})" if kw else label
        results.append((tag, rows, cost))

    print(f"{len(nums)} numbers. Share cut off mid-number. 'none' = no hold.\n")
    print(f"{'':22s}{'':8s}{'English':>10s}{'Hindi':>10s}{'All':>10s}")
    for name in SETUPS:
        first = results[0][1][name]
        print(f"{name:22s}{'none':8s}" + "".join(f"{100*b/t:>9.0f}%" for b, _, t in first))
        for i, (_, rows, _) in enumerate(results):
            print(f"{'':22s}{'rule ' + str(i + 1):8s}" + "".join(f"{100*a/t:>9.0f}%" for _, a, t in rows[name]))
    print("\nCost, holds at the real end of speech (each adds up to the hold time):")
    for i, (tag, _, (held, total)) in enumerate(results):
        print(f"  rule {i + 1}, {tag}: {held}/{total} = {100*held/total:.0f}%")


if __name__ == "__main__":
    main()
