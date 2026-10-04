"""Small, bounded acoustic summaries shared by corpus feature routes."""

from __future__ import annotations

import math


def summarize_sound(
    sound, pitch, words: list[dict] | None = None,
    *, window_start: float | None = None, window_end: float | None = None,
) -> dict:
    """Summarize a clip or a bounded interval; undefined values become JSON null.

    ``words`` use the same local timeline as ``sound``.  A bounded interval is
    useful when alignment needs context padding but the acoustic outcome must
    describe only the annotated utterance.
    """
    import numpy as np

    lower = 0.0 if window_start is None else max(0.0, float(window_start))
    upper = float(sound.duration) if window_end is None else min(float(sound.duration), float(window_end))
    if upper <= lower:
        raise ValueError("Acoustic summary window must have positive duration")

    frequencies = np.asarray(pitch.selected_array["frequency"], dtype=float)
    pitch_times = np.asarray(pitch.xs(), dtype=float)
    frequencies = frequencies[(pitch_times >= lower) & (pitch_times <= upper)]
    voiced = frequencies[np.isfinite(frequencies) & (frequencies > 0)]
    intensity_object = sound.to_intensity()
    intensity = np.asarray(intensity_object.values, dtype=float).ravel()
    intensity_times = np.asarray(intensity_object.xs(), dtype=float)
    intensity = intensity[(intensity_times >= lower) & (intensity_times <= upper)]
    intensity = intensity[np.isfinite(intensity)]

    def number(value) -> float | None:
        value = float(value)
        return round(value, 3) if math.isfinite(value) else None

    def stat(values, function) -> float | None:
        return number(function(values)) if values.size else None

    result = {
        "duration_sec": number(upper - lower),
        "pitch_mean_hz": stat(voiced, np.mean),
        "pitch_median_hz": stat(voiced, np.median),
        "pitch_std_hz": stat(voiced, np.std),
        "pitch_p05_hz": stat(voiced, lambda x: np.percentile(x, 5)),
        "pitch_p95_hz": stat(voiced, lambda x: np.percentile(x, 95)),
        "pitch_min_hz": stat(voiced, np.min),
        "pitch_max_hz": stat(voiced, np.max),
        "voiced_frame_fraction": number(voiced.size / frequencies.size) if frequencies.size else 0.0,
        "intensity_mean_db": stat(intensity, np.mean),
        "intensity_std_db": stat(intensity, np.std),
    }
    result["pitch_robust_range_hz"] = (
        number(result["pitch_p95_hz"] - result["pitch_p05_hz"])
        if result["pitch_p95_hz"] is not None and result["pitch_p05_hz"] is not None
        else None
    )
    if words is not None:
        intervals = sorted((max(lower, float(w["start"])),
                            min(upper, float(w["end"]))) for w in words)
        intervals = [(start, end) for start, end in intervals if end > start]
        articulation_sec = 0.0
        covered_until = lower
        for start, end in intervals:
            articulation_sec += max(0.0, end - max(start, covered_until))
            covered_until = max(covered_until, end)
        duration = upper - lower
        result.update({
            "word_count": len(intervals),
            "words_per_sec": number(len(intervals) / duration) if duration > 0 else None,
            "articulation_sec": number(articulation_sec),
            "articulation_fraction": number(articulation_sec / duration) if duration > 0 else None,
        })
    return result
