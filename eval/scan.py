"""Find the clips in the public datasets that contain the things callers say on
phone calls: numbers, digit strings, amounts, dates and times.

Reads only the text columns (no audio), so it is fast. Writes
data/manifest.csv with one row per matching clip and prints a summary.

    python scan.py
"""
import collections
import csv
import glob
import re

import pyarrow.parquet as pq

# English number words, and Hindi number words in Devanagari and romanised form.
EN_NUM = r"\b(zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|lakh|lakhs|crore|crores|double|triple)\b"
HI_NUM = r"(शून्य|एक|दो|तीन|चार|पांच|पाँच|छह|छः|सात|आठ|नौ|दस|ग्यारह|बारह|तेरह|चौदह|पंद्रह|सोलह|सत्रह|अठारह|उन्नीस|बीस|तीस|चालीस|पचास|साठ|सत्तर|अस्सी|नब्बे|सौ|हज़ार|हजार|लाख|करोड़|डेढ़|ढाई|साढ़े|सवा|पौने)"
MONEY = r"(₹|\brs\.?\b|\brupees?\b|\binr\b|रुपये|रुपए|रुपया|लाख|करोड़|\blakhs?\b|\bcrores?\b)"
MONTHS = r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today)\b|(जनवरी|फरवरी|फ़रवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|कल|परसों|तारीख)"
DIGIT_WORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple|शून्य|एक|दो|तीन|चार|पांच|पाँच|छह|छः|सात|आठ|नौ)"
# Five or more single-digit words in a row: a number read out digit by digit.
DIGIT_RUN = r"(?:(?<![\wऀ-ॿ])" + DIGIT_WORD + r"(?![\wऀ-ॿ])\s+){4,}" + DIGIT_WORD + r"(?![\wऀ-ॿ])"
MONTH = r"(?:january|february|march|april|june|july|august|september|october|november|december|जनवरी|फरवरी|फ़रवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|सितम्बर|अक्टूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर)"
# A month next to a number (either order), a weekday, or tomorrow / day after.
DATE = r"(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?" + MONTH + r"|" + MONTH + r"\s+\d{1,2}|\S+\s+" + MONTH + r"|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|परसों|तारीख)"
TIME = r"\d{1,2}[:.]\d{2}|\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)(?!\w)|o'clock|(बजे)"


def tags_for(text):
    t = text.lower()
    tags = []
    digits = re.findall(r"\d", t)
    num_words = len(re.findall(EN_NUM, t)) + len(re.findall(HI_NUM, t))
    # A run of 5+ digits (possibly spaced) looks like a phone number, PIN, account or ID.
    if re.search(r"(\d[\s-]?){5,}", t) or re.search(DIGIT_RUN, t):
        tags.append("digit_string")
    if re.search(r"\b[a-z]{2,5}\d{2,}[a-z0-9]*\b", t):
        tags.append("alnum_id")
    if re.search(MONEY, t):
        tags.append("amount")
    if re.search(DATE, t) or re.search(TIME, t):
        tags.append("datetime")
    if digits or num_words >= 2:
        tags.append("number")
    return tags


def main():
    out = []
    summary = collections.Counter()
    for ds in ("svarah", "lahaja"):
        files = sorted(glob.glob(f"data/{ds}/data/*.parquet"))
        if not files:
            print(f"{ds}: no parquet files yet")
            continue
        for f in files:
            schema = pq.read_schema(f).names
            cols = [c for c in schema if c != "audio_filepath"]
            table = pq.read_table(f, columns=cols).to_pylist()
            for i, row in enumerate(table):
                text = row.get("verbatim") or row.get("text") or ""
                tags = tags_for(text)
                summary[(ds, "all")] += 1
                if not tags:
                    continue
                for tag in tags:
                    summary[(ds, tag)] += 1
                out.append({
                    "dataset": ds,
                    "file": f.replace("\\", "/"),
                    "row": i,
                    "tags": "|".join(tags),
                    "duration": row.get("duration"),
                    "scenario": row.get("scenario", ""),
                    "speaker_lang": row.get("native_language") or row.get("primary_language") or "",
                    "state": row.get("native_place_state", ""),
                    "text": text,
                })
    with open("data/manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()) if out else ["dataset"])
        w.writeheader()
        w.writerows(out)
    for ds in ("svarah", "lahaja"):
        total = summary[(ds, "all")]
        if not total:
            continue
        print(f"\n{ds}: {total} clips")
        for tag in ("number", "digit_string", "alnum_id", "amount", "datetime"):
            print(f"  {tag:13s} {summary[(ds, tag)]}")
    print(f"\nwrote data/manifest.csv with {len(out)} matching clips")


if __name__ == "__main__":
    main()
