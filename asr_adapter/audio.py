"""Pure audio helpers for the ASR provider: chunking, silence trimming, loop detection.

Everything here works on 16 kHz mono float32 arrays and is deterministic, so it is
unit-tested without a model.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 10          # 100 ms analysis frames
VOICE_RMS = 0.01                   # a frame above this (about -40 dBFS) counts as sound
MIN_VOICED_FRACTION = 0.02         # a chunk with less voiced audio than this is silence
PAD_FRAMES = 3                     # keep 0.3 s of context around trimmed speech


@dataclass(frozen=True)
class Chunk:
    """A stretch of audio to transcribe, positioned in the source recording (seconds)."""

    start: float
    end: float
    samples: np.ndarray


def frame_rms(audio: np.ndarray) -> np.ndarray:
    """RMS of each whole 100 ms frame (a trailing partial frame is ignored)."""
    n = len(audio) // FRAME
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[: n * FRAME].reshape(n, FRAME).astype(np.float32)
    return np.sqrt((frames ** 2).mean(axis=1))


def plan_cuts(rms: np.ndarray, *, target_s: float = 30.0, slack_s: float = 4.0) -> list[int]:
    """Frame indices that split the recording into ~target_s pieces, each cut placed
    at the quietest frame within +/-slack_s of the target so words are not sliced."""
    per_s = SAMPLE_RATE / FRAME
    target, slack = int(target_s * per_s), int(slack_s * per_s)
    cuts, pos = [0], 0
    while len(rms) - pos > target + slack:
        lo, hi = pos + target - slack, pos + target + slack
        pos = lo + int(np.argmin(rms[lo:hi]))
        cuts.append(pos)
    return cuts


def make_chunks(audio: np.ndarray, *, target_s: float = 30.0, slack_s: float = 4.0) -> list[Chunk]:
    """Split into ~target_s chunks, trim each to its voiced span, and drop silent ones.

    Trimming matters: a model shown a mostly-silent clip tends to loop on filler
    text, and the trimmed bounds also make the reported timestamps tighter."""
    rms = frame_rms(audio)
    cuts = plan_cuts(rms, target_s=target_s, slack_s=slack_s)
    ends = cuts[1:] + [len(rms)]
    chunks: list[Chunk] = []
    for lo, hi in zip(cuts, ends):
        span = rms[lo:hi]
        if len(span) == 0:
            continue
        voiced = np.flatnonzero(span > VOICE_RMS)
        if len(voiced) / len(span) < MIN_VOICED_FRACTION:
            continue
        first = max(0, int(voiced[0]) - PAD_FRAMES)
        last = min(len(span), int(voiced[-1]) + 1 + PAD_FRAMES)
        start_sample, end_sample = (lo + first) * FRAME, (lo + last) * FRAME
        if hi == len(rms) and last == len(span):
            end_sample = len(audio)     # keep the trailing partial frame of the recording
        chunks.append(Chunk(
            start=start_sample / SAMPLE_RATE,
            end=end_sample / SAMPLE_RATE,
            samples=audio[start_sample:end_sample],
        ))
    return chunks


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def is_degenerate(text: str, *, gram: int = 6, min_repeats: int = 5, coverage: float = 0.4) -> bool:
    """True if the text is dominated by a looping phrase (a decoding failure, not speech).

    Counts how much of the text is made of 6-character windows that occur at least
    five times.  Speech repeats short words ("对对对") but does not repeat its
    phrases like that over 40% of a clip; a looping decoder does exactly that, and
    a long loop phrase is caught because every one of its windows repeats."""
    n = _norm(text)
    if len(n) < gram * min_repeats:
        return False
    counts = Counter(n[i:i + gram] for i in range(len(n) - gram + 1))
    repeated = sum(c for c in counts.values() if c >= min_repeats)
    return repeated / sum(counts.values()) >= coverage


def join_text(pieces: list[str]) -> str:
    """Join chunk texts; add a space only between two ASCII-word edges."""
    out = ""
    for piece in (p.strip() for p in pieces if p and p.strip()):
        if out and out[-1].isascii() and out[-1].isalnum() and piece[0].isascii() and piece[0].isalnum():
            out += " "
        out += piece
    return out


def token_budget(seconds: float) -> int:
    """Upper bound on generated tokens for a clip, so a loop cannot run away."""
    return int(seconds * 8) + 64
