from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .types import AlignmentError


@dataclass
class CTCAlignment:
    path: np.ndarray  # state indices, one per acoustic frame
    frame_scores: np.ndarray
    score: float


@dataclass
class WordSpan:
    start_frame: int
    end_frame: int  # exclusive
    score: float


def append_star_column(log_probs: np.ndarray, star_penalty: float = 0.0) -> tuple[np.ndarray, int]:
    """Append a synthetic <star> emission column and return its token id.

    The star token absorbs audio at transcript edges that is not represented by
    the transcript (silence, noise, non-speech events). ``star_penalty=0`` gives
    it log-probability 0, matching the widely used MMS CTC aligner; larger
    values make star frames progressively less attractive.
    """
    if log_probs.ndim != 2:
        raise AlignmentError(f"log_probs must be [T, C], got shape {log_probs.shape}")
    arr = np.asarray(log_probs, dtype=np.float32)
    star_id = int(arr.shape[1])
    column = np.full((arr.shape[0], 1), -float(star_penalty), dtype=np.float32)
    return np.concatenate([arr, column], axis=1), star_id


def _min_required_frames(targets: np.ndarray) -> int:
    repeats = int(np.sum(targets[1:] == targets[:-1])) if targets.size > 1 else 0
    return int(targets.size + repeats)


def ctc_forced_align(
    log_probs: np.ndarray,
    targets: Sequence[int] | np.ndarray,
    blank: int,
) -> CTCAlignment:
    """Viterbi CTC forced alignment (pure NumPy, deterministic).

    States are the CTC-extended sequence B, l1, B, l2, ..., B. At each frame a
    path may stay, advance from the previous state, or skip a blank between two
    *different* adjacent labels. No timestamp interpolation is performed.
    """
    lp = np.asarray(log_probs, dtype=np.float32)
    if lp.ndim != 2:
        raise AlignmentError(f"log_probs must be [T, C], got shape {lp.shape}")
    T, C = lp.shape
    tgt = np.asarray(targets, dtype=np.int64)
    L = int(tgt.size)
    if L == 0:
        raise AlignmentError("Empty CTC target sequence")
    if not (0 <= blank < C):
        raise AlignmentError(f"blank id {blank} outside log_probs vocabulary size {C}")
    if np.any(tgt == blank):
        raise AlignmentError("CTC target sequence must not contain the blank id")
    if np.any(tgt < 0) or np.any(tgt >= C):
        raise AlignmentError("CTC target ids outside log_probs vocabulary")
    need = _min_required_frames(tgt)
    if T < need:
        raise AlignmentError(
            f"Audio has {T} CTC frames but the transcript needs at least {need} frames."
        )

    S = 2 * L + 1
    state_token = np.empty(S, dtype=np.int64)
    state_token[0::2] = blank
    state_token[1::2] = tgt
    emit = lp[:, state_token]  # [T, S]

    skip_allowed = np.zeros(S, dtype=bool)
    for s in range(3, S, 2):
        i = (s - 1) // 2
        if tgt[i] != tgt[i - 1]:
            skip_allowed[s] = True

    dp = np.full(S, -np.inf, dtype=np.float32)
    dp[0] = emit[0, 0]
    dp[1] = emit[0, 1]
    backptr = np.zeros((T, S), dtype=np.int8)

    for t in range(1, T):
        stay = dp
        advance = np.full(S, -np.inf, dtype=np.float32)
        advance[1:] = dp[:-1]
        skip = np.full(S, -np.inf, dtype=np.float32)
        if np.any(skip_allowed):
            src = np.flatnonzero(skip_allowed) - 2
            skip[skip_allowed] = dp[src]
        cand = np.stack([stay, advance, skip], axis=0)
        choice = np.argmax(cand, axis=0).astype(np.int8)
        best = cand[choice, np.arange(S)]
        backptr[t] = choice
        dp = best + emit[t]
        if not np.any(np.isfinite(dp)):
            raise AlignmentError("CTC trellis became invalid (no finite path).")

    final_state = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if not np.isfinite(dp[final_state]):
        raise AlignmentError("No valid CTC alignment path for this transcript/audio window.")
    path = np.empty(T, dtype=np.int32)
    state = int(final_state)
    path[T - 1] = state
    for t in range(T - 1, 0, -1):
        c = int(backptr[t, state])
        source = state if c == 0 else (state - 1 if c == 1 else state - 2)
        if source < 0:
            raise AlignmentError("CTC backtracking produced an invalid source state.")
        state = source
        path[t - 1] = state

    labels = labels_from_path(path, tgt, blank)
    if not np.array_equal(labels, tgt):
        raise AlignmentError("Internal CTC path does not collapse to the requested targets.")

    frame_scores = emit[np.arange(T), path]
    return CTCAlignment(path=path, frame_scores=frame_scores, score=float(dp[final_state]))


def labels_from_path(path: np.ndarray, targets: np.ndarray, blank: int) -> np.ndarray:
    """Apply the CTC collapse rule: merge consecutive repeats, then drop blanks.

    A repeated label separated by an explicit blank state must remain two
    labels; removing blanks first would incorrectly merge them.
    """
    emitted: list[int] = []
    for state in np.asarray(path, dtype=np.int64):
        if state % 2 == 0:
            emitted.append(int(blank))
        else:
            emitted.append(int(targets[(state - 1) // 2]))
    merged: list[int] = []
    for token in emitted:
        if not merged or merged[-1] != token:
            merged.append(token)
    return np.asarray([t for t in merged if t != blank], dtype=np.int64)


def word_spans_from_path(
    alignment: CTCAlignment,
    word_index: Sequence[Optional[int]],
    n_words: int,
    n_targets: int,
) -> list[WordSpan]:
    """Map aligned character states back to original word intervals.

    ``word_index[i]`` is the original word index for target token ``i``; star
    tokens use ``None``. Adjacent blank frames are split at their midpoint, so
    word intervals are chronological and non-overlapping without fabricating
    any frame that the CTC path did not emit.
    """
    if len(word_index) != n_targets:
        raise AlignmentError(
            f"word_index has {len(word_index)} entries but the CTC target sequence has {n_targets}."
        )
    frames: list[list[int]] = [[] for _ in range(n_words)]
    path = alignment.path
    for t, state in enumerate(path):
        state = int(state)
        if state % 2 == 1:
            token_pos = (state - 1) // 2
            if token_pos >= len(word_index):
                raise AlignmentError("CTC state points outside the target sequence.")
            widx = word_index[token_pos]
            if widx is not None:
                if not 0 <= widx < n_words:
                    raise AlignmentError(f"Invalid word index {widx} in CTC targets.")
                frames[widx].append(t)

    raw: list[tuple[int, int]] = []
    for i, f in enumerate(frames):
        if not f:
            raise AlignmentError(
                f"CTC alignment produced no frames for word {i}; refusing to invent timestamps."
            )
        raw.append((min(f), max(f) + 1))

    expanded: list[tuple[int, int]] = []
    for i, (start, end) in enumerate(raw):
        new_start = start
        new_end = end
        if i > 0:
            prev_end = raw[i - 1][1]
            if start > prev_end:
                new_start = (prev_end + start) // 2
        if i + 1 < len(raw):
            next_start = raw[i + 1][0]
            if next_start > end:
                new_end = (end + next_start) // 2
        if new_start >= new_end:
            new_start, new_end = start, end
        expanded.append((new_start, new_end))

    spans: list[WordSpan] = []
    prev_end = 0
    for start, end in expanded:
        start = max(start, prev_end)
        end = max(end, start + 1)
        score = float(np.mean(alignment.frame_scores[start:end]))
        spans.append(WordSpan(start_frame=start, end_frame=end, score=score))
        prev_end = end
    return spans
