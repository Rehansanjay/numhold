# Indian voice agents don't cut you off mid-number. They're just slow.

People read phone numbers and PIN codes in chunks: "nine eight four zero one ... two three four
five six". A voice agent tuned to reply quickly hears that pause, decides the caller has finished,
and answers half a number.

I measured how often that happens on recorded speech, then called real Indian voice agents to see
whether it happens on real phone lines. It mostly doesn't, and the reason is the interesting part.

## On recorded speech: up to 39% of numbers cut off

341 numbers read by Indian English and Hindi speakers (AI4Bharat Svarah and Lahaja, at phone
quality, 8 kHz). At every pause inside or just before the number, each setup was asked "has the
caller finished?" A number counts as cut off if any of its pauses ends the turn.

| Turn detection | Numbers cut off |
|---|---|
| LiveKit turn detector, default settings | 8% |
| LiveKit turn detector, tuned to reply fast | **39%** |
| Pipecat Smart Turn v3 | 30% |
| Silence timer, 0.5 s | 18% |

So the faster you tune an agent, the more numbers it breaks.

## On real phone lines: 1 cut-off in 16 numbers

On 12 Nov I phoned the public Indian demo numbers of five voice-AI products (8 calls) with
[voice-agent-probe](https://github.com/Rehansanjay/voice-agent-probe). In each call the probe read a
mobile number and a PIN code, each with a 1-second pause in the middle, and timed what the agent did.

- **1 of 16 numbers was cut off:** one agent started talking 0.79 s into the pause in a PIN code.
- **The other 15 were heard whole, because the agents waited a long time:** after the mobile number
  ended, the median reply came **2.1 s** later (1.7 to 4.7 s).

That's the trade-off the recorded-speech numbers predict: the agents that don't cut callers off are
the ones that wait two seconds after everything, numbers or not. Callers feel that wait on every
turn, not just the ones with numbers in them.

One exception is worth noting: one agent replied to the PIN in about 0.6 s and still waited through
the pause. So fast and safe is possible. It just isn't the default.

## A third option: be fast, and wait only for numbers

[numhold](https://github.com/Rehansanjay/numhold) is a small check in front of the turn detector.
When the detector says "finished" but the transcript clearly has a number still coming (5 digits
after "my mobile number is", or after the agent asked for one), it waits for the caller to carry on,
up to 2 s. Everything else gets the detector's own answer.

| Turn detection | Cut off without | With numhold |
|---|---|---|
| LiveKit turn detector, tuned fast | 39% | 12% |
| Pipecat Smart Turn v3 | 30% | 8% |
| Silence timer, 0.5 s | 18% | 6% |

It costs a hold on 6% of real turn ends. On a live call with my own Pipecat agent, the version
without it cut me off mid-number several times; with it, the mobile number and PIN arrived whole.
It works with Pipecat and LiveKit Agents.

## Limits

- 8 real calls, to demo lines, not production traffic. Read this as "what a careful caller would
  notice", not as a survey.
- The probe's lines are in English and several of these agents open in Hindi, so they may have
  been more cautious than with a caller in their own language.
- One pause length (1.0 s). Longer pauses would catch more agents.
- The recorded-speech test is read speech, not live calls, in English and Hindi only.

If you run a voice agent that takes numbers over the phone, I'm happy to call it with a paused
number and send you the recording and timings.
