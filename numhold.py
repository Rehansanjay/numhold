"""Number-aware turn hold: don't end the caller's turn while they are giving a number.

Sits in front of any end-of-turn decision (silence timer, Pipecat Smart Turn,
LiveKit's turn detector). When that decision says "the caller has finished",
ask `number_in_progress(transcript_so_far)`. If it says hold, keep listening
until either the caller speaks again or `MAX_HOLD_S` of silence has passed.

It works on the transcript, so it needs no audio model and adds no latency to
turns that have nothing to do with numbers.
"""
import re

MAX_HOLD_S = 2.0

# Lead-ins that announce a number, and how many digits that kind of number has.
# A tuple means any of these lengths is a complete answer.
CUES = [
    (r"(mobile|phone|contact|whatsapp|cell)( number| no)?|मोबाइल|फ़ोन|फोन|फ़ोन नंबर", (10, 11, 12)),
    (r"pin ?code|pin|पिन ?कोड|पिनकोड|पिन", (6,)),
    (r"otp|o t p|ओटीपी|one time password", (4, 6)),
    (r"aadhaar|aadhar|आधार", (12,)),
    (r"uan|u a n|pran|p r a n|prn|यूएएन|यूएन|प्रान", (12,)),
    (r"ifsc", (11,)),
    # Generic: "number", "id", "account" with no known length. Treat 10 digits as the
    # shortest plausible complete answer.
    (r"number|no|id|i d|account|tracking|application|reference|order|नंबर|नम्बर|आईडी|खाता|कोड|code", (10,)),
]
FILLER = r"(is|it's|its|hai|है|ka|का|ki|की|ke|के|my|mera|मेरा|meri|मेरी|the|was|:|-)"

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "शून्य": "0", "ज़ीरो": "0", "जीरो": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4",
    "पांच": "5", "पाँच": "5", "छह": "6", "छः": "6", "छे": "6", "चे": "6", "सात": "7", "आठ": "8", "नौ": "9",
}
REPEAT = {"double": 2, "triple": 3, "डबल": 2, "ट्रिपल": 3}
DEV = str.maketrans("०१२३४५६७८९", "0123456789")


def _tokens(text):
    text = text.lower().translate(DEV).replace("।", " ")
    text = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", text)
    return re.findall(r"[\wऀ-ॣ०-ॿ']+|[:\-]", text)


def _trailing_digits(tokens):
    """Digits at the end of the transcript, and the index where they start."""
    digits, i, pending_repeat = "", len(tokens), 1
    while i > 0:
        t = tokens[i - 1]
        if t.isdigit():
            digits = t + digits
        elif t in DIGIT_WORDS:
            digits = DIGIT_WORDS[t] + digits
        elif t in REPEAT and digits:
            digits = digits[0] * (REPEAT[t] - 1) + digits
        else:
            break
        i -= 1
    return digits, i


def _cue_before(tokens, end):
    """The expected lengths for the lead-in just before position `end`, if there is one."""
    j = end
    while j > 0 and re.fullmatch(FILLER, tokens[j - 1]):
        j -= 1
    tail = " ".join(tokens[max(0, j - 3):j])
    for pattern, lengths in CUES:
        if re.search(rf"(^|\s)({pattern})$", tail):
            return lengths
    return None


def _asked_for(question):
    """The expected lengths if the agent's last question asked for a number, else None."""
    q = " ".join(_tokens(question))
    for pattern, lengths in CUES:
        if re.search(rf"(^|\s)({pattern})(\s|$)", q):
            return lengths
    return None


def number_in_progress(text, asked=""):
    """(hold, reason) for the transcript the agent has at the moment the caller paused.

    `asked` is the agent's last question. On a real call the agent usually says
    "what's your mobile number?" and the caller answers with digits alone, so the
    lead-in that tells us how long the number should be is in the question.
    """
    tokens = _tokens(text)
    if not tokens:
        return False, ""
    digits, start = _trailing_digits(tokens)
    if not digits:
        lengths = _cue_before(tokens, len(tokens))
        return (True, "lead-in, no digits yet") if lengths else (False, "")
    lengths = _cue_before(tokens, start)
    if lengths is None and asked:
        lengths = _asked_for(asked)
        if lengths and len(digits) < min(lengths):
            return True, f"{len(digits)} of {min(lengths)}+ digits (agent asked for it)"
    if lengths is None:
        # Digits with no lead-in: hold only while it is too short to be any common number.
        return (len(digits) < 4, "short digit run, no lead-in") if len(digits) < 4 else (False, "")
    if len(digits) < min(lengths):
        return True, f"{len(digits)} of {min(lengths)}+ digits"
    return False, ""


if __name__ == "__main__":
    for s in ["my phone number is", "my phone number is 98400", "my phone number is 9840012345",
              "pin code six zero zero", "pin code 600041", "otp is 4471", "मेरा मोबाइल नंबर नौ आठ चार",
              "can you help me", "I want 2 tickets", "tracking id double nine five seven"]:
        print(f"{s!r:45} -> {number_in_progress(s)}")
