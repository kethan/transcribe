from __future__ import annotations

import json
import random
import time
from typing import Any, Optional

from .types import AudioSpan, StageError, TimedWord, TranscriptUnit, jsonable


class BodhanError(StageError):
    pass


class BodhanRetryableError(BodhanError):
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class BodhanFatalError(BodhanError):
    pass


class BodhanPayloadTooLarge(BodhanError):
    pass


def _response_error(resp: Any) -> str:
    try:
        body = resp.json()
    except Exception:
        return (resp.text or "")[:400]
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        err = body["error"]
        return f"{err.get('code', 'unknown_error')}: {err.get('message', '')} (request_id={err.get('request_id')})"
    return json.dumps(jsonable(body), ensure_ascii=False)[:400]


def _parse_retry_after(resp: Any) -> Optional[float]:
    value = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class BodhanClient:
    """Minimal client for the documented Bodhan speech-to-text route.

    Only ``file``, ``model`` and optional ``language`` are sent. We do not send
    undocumented ``response_format``/``timestamp_granularities`` fields.
    """

    def __init__(self, config: Config):
        self.config = config
        if not config.bodhan_api_key:
            raise BodhanFatalError(
                "BODHAN_API_KEY is not set. Export it in the environment; credentials are never stored in code."
            )

    def _post_once(self, span: AudioSpan, chunk_id: str) -> dict[str, Any]:
        try:
            import requests
        except Exception as exc:  # pragma: no cover - installation issue
            raise BodhanFatalError("The 'requests' package is required for Bodhan uploads.") from exc

        data: dict[str, str] = {"model": self.config.bodhan_model}
        if self.config.language:
            data["language"] = self.config.language
        url = f"{self.config.bodhan_base_url}/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.config.bodhan_api_key}"}
        with span.path.open("rb") as fh:
            files = {"file": (span.path.name, fh, "audio/wav")}
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=(10, self.config.bodhan_timeout),
                )
            except requests.exceptions.RequestException as exc:
                raise BodhanRetryableError(f"network/upload failure: {exc}") from exc

        status = int(resp.status_code)
        if status == 413:
            raise BodhanPayloadTooLarge(
                f"HTTP 413 for {chunk_id}: {_response_error(resp)}; splitting the audio span."
            )
        if status in {408, 425, 429, 500, 502, 503, 504}:
            raise BodhanRetryableError(
                f"HTTP {status} for {chunk_id}: {_response_error(resp)}",
                retry_after=_parse_retry_after(resp),
            )
        if status == 415:
            raise BodhanFatalError(
                f"HTTP 415 for {chunk_id}: Bodhan could not measure the WAV duration. "
                f"Verify the generated PCM16 WAV. {_response_error(resp)}"
            )
        if status == 401:
            raise BodhanFatalError(
                f"HTTP 401 for {chunk_id}: invalid or missing BODHAN_API_KEY. {_response_error(resp)}"
            )
        if not (200 <= status < 300):
            detail = _response_error(resp)
            # Some gateways report an over-duration request as 400. Detect it
            # instead of silently dropping a chunk.
            if "duration" in detail.lower() and ("30" in detail or "too long" in detail.lower()):
                raise BodhanPayloadTooLarge(f"HTTP {status}: {detail}")
            raise BodhanFatalError(f"HTTP {status} for {chunk_id}: {detail}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise BodhanFatalError(f"Bodhan returned non-JSON success response: {exc}") from exc
        if not isinstance(body, dict):
            raise BodhanFatalError("Bodhan success response is not a JSON object.")
        if "error" in body:
            raise BodhanFatalError(f"Bodhan returned an error object on success: {_response_error(resp)}")
        return body

    def request(self, span: AudioSpan, chunk_id: str) -> dict[str, Any]:
        for attempt in range(self.config.bodhan_max_retries + 1):
            try:
                return self._post_once(span, chunk_id)
            except BodhanRetryableError as exc:
                if attempt >= self.config.bodhan_max_retries:
                    raise BodhanFatalError(
                        f"Bodhan request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                if exc.retry_after is not None:
                    delay = exc.retry_after
                else:
                    delay = min(60.0, self.config.bodhan_retry_base * (2**attempt))
                delay += random.uniform(0.0, 0.25)
                print(
                    f"[Bodhan] {chunk_id}: retry {attempt + 1}/{self.config.bodhan_max_retries} "
                    f"in {delay:.1f}s ({exc})"
                )
                time.sleep(delay)
        raise BodhanFatalError("Unreachable Bodhan retry state")


def canonical_text(text: str) -> str:
    return "".join(text.split())


def _word_text(obj: dict[str, Any]) -> Optional[str]:
    if "word" in obj:
        return str(obj["word"])
    if "text" in obj:
        return str(obj["text"])
    return None


def _collect_timed_words(
    objects: list[Any], span: AudioSpan, context: str
) -> Optional[list[TimedWord]]:
    """Return word timestamps only if *every* usable entry is reliably timed."""
    words: list[TimedWord] = []
    for obj in objects:
        if not isinstance(obj, dict):
            return None
        if obj.get("type") == "spacing" or (isinstance(obj.get("text"), str) and obj["text"].isspace()):
            continue
        text = _word_text(obj)
        if text is None:
            return None
        start = obj.get("start", obj.get("start_time", obj.get("begin")))
        end = obj.get("end", obj.get("end_time"))
        if start is None or end is None:
            return None
        try:
            s = float(start)
            e = float(end)
        except (TypeError, ValueError):
            return None
        if s < -0.05 or e > span.duration + 0.05:
            raise BodhanFatalError(
                f"{context}: word timestamp [{s}, {e}] outside the {span.duration:.3f}s chunk."
            )
        s = max(0.0, s)
        e = min(span.duration, e)
        if not (s < e):
            raise BodhanFatalError(f"{context}: non-positive word duration [{s}, {e}].")
        words.append(TimedWord(text=text, start=span.start + s, end=span.start + e))
    if not words:
        return None
    words.sort(key=lambda w: (w.start, w.end))
    return words


def _clamped_segment_window(seg: dict[str, Any], span: AudioSpan, context: str) -> tuple[float, float]:
    try:
        s = float(seg["start"])
        e = float(seg["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BodhanFatalError(f"{context}: invalid segment start/end.") from exc
    s = max(0.0, min(span.duration, s))
    e = max(s, min(span.duration, e))
    if not (s < e):
        raise BodhanFatalError(f"{context}: empty segment window after clamping.")
    return span.start + s, span.start + e


def parse_bodhan_response(
    raw: dict[str, Any],
    span: AudioSpan,
    chunk_index: int,
    chunk_key: str,
    warnings: list[str],
) -> tuple[list[TranscriptUnit], str, bool]:
    """Parse Bodhan text from either backend and optional extra timestamp shapes.

    The hosted API's published 200 schema contains only ``text``; the local HF
    model also returns a text string. If a response additionally contains
    ``words`` or ``segments[].words`` with start/end, we use those times;
    otherwise forced alignment is required. No request asks for undocumented
    fields and no missing timestamp is interpolated.
    """
    text = raw.get("text") if isinstance(raw.get("text"), str) else ""
    segments = raw.get("segments")
    top_words = raw.get("words")
    units: list[TranscriptUnit] = []
    has_word_times = False

    def make_unit(
        i: int,
        unit_text: str,
        start: float,
        end: float,
        words: Optional[list[TimedWord]],
        source: str,
    ) -> TranscriptUnit:
        return TranscriptUnit(
            chunk_index=chunk_index,
            unit_index=i,
            key=f"{chunk_key}:u{i}",
            text=unit_text,
            start=start,
            end=end,
            span=span,
            words=words,
            source=source,
        )

    if isinstance(segments, list) and segments:
        segment_texts: list[str] = []
        all_word_timed = True
        all_segment_timed = True
        timed_words: list[Optional[list[TimedWord]]] = []
        windows: list[Optional[tuple[float, float]]] = []
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                all_word_timed = False
                all_segment_timed = False
                segment_texts.append("")
                timed_words.append(None)
                windows.append(None)
                continue
            seg_text = str(seg.get("text", "")) if seg.get("text") is not None else ""
            segment_texts.append(seg_text)
            ws = seg.get("words")
            tw = (
                _collect_timed_words(ws, span, f"chunk {chunk_key} segment {i}")
                if isinstance(ws, list)
                else None
            )
            timed_words.append(tw)
            if tw is None:
                all_word_timed = False
            if "start" in seg and "end" in seg:
                windows.append(_clamped_segment_window(seg, span, f"chunk {chunk_key} segment {i}"))
            else:
                all_segment_timed = False
                windows.append(None)

        if all_word_timed:
            has_word_times = True
            for i, tw in enumerate(timed_words):
                assert tw is not None
                word_text = " ".join(w.text for w in tw)
                unit_text = word_text or segment_texts[i]
                units.append(
                    make_unit(
                        i,
                        unit_text,
                        start=span.start,
                        end=span.end,
                        words=tw,
                        source="bodhan.segments.words",
                    )
                )
        elif all_segment_timed:
            for i, win in enumerate(windows):
                assert win is not None
                units.append(
                    make_unit(
                        i,
                        segment_texts[i],
                        start=win[0],
                        end=win[1],
                        words=None,
                        source="bodhan.segments.timed_text",
                    )
                )
        else:
            combined = " ".join(t for t in segment_texts if t) or text
            units.append(
                make_unit(
                    0, combined, span.start, span.end, None, "bodhan.segments.text"
                )
            )
    elif isinstance(top_words, list) and top_words:
        tw = _collect_timed_words(top_words, span, f"chunk {chunk_key}")
        if tw is not None:
            has_word_times = True
            unit_text = " ".join(w.text for w in tw)
            units.append(
                make_unit(0, unit_text, span.start, span.end, tw, "bodhan.words")
            )
        else:
            units.append(make_unit(0, text, span.start, span.end, None, "bodhan.text"))
    else:
        units.append(make_unit(0, text, span.start, span.end, None, "bodhan.text"))

    chunk_text = " ".join(u.text for u in units if u.text.strip())
    if has_word_times and text and canonical_text(text) != canonical_text(chunk_text):
        warnings.append(
            f"chunk {chunk_key}: Bodhan word list and text field differ; using the timed word texts "
            "as the authoritative transcript."
        )
    return units, chunk_text, has_word_times
