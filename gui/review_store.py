"""Manual clinical review: epoch flags, free-text annotations, CSV export to the stimulator.

Persisted as a JSON sidecar next to the results so it survives a re-run and can be picked
up at the patient's next visit. The pipeline's own outputs are never modified in place —
the augmented table is written as a new file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Epoch statuses the clinicians asked for.
STATUS_OK = "ok"
STATUS_BAD = "bad"                      # "не нравится"
STATUS_FALSE_POSITIVE = "false_positive"
STATUS_COMPLEX = "complex"              # "сложный ответ"
STATUSES = (STATUS_OK, STATUS_BAD, STATUS_FALSE_POSITIVE, STATUS_COMPLEX)

STATUS_LABEL = {
    STATUS_OK: "OK",
    STATUS_BAD: "Bad epoch",
    STATUS_FALSE_POSITIVE: "False positive",
    STATUS_COMPLEX: "Complex response",
}


@dataclass
class Annotation:
    """A manually marked time window with a text note."""
    condition: str
    detection_index: int   # -1 = free-standing (not tied to a detection)
    tmin: float            # seconds, absolute in the condition's raw
    tmax: float
    text: str
    channel: str = ""


@dataclass
class Review:
    # keyed by "condition|detection_index"
    statuses: dict[str, str] = field(default_factory=dict)
    annotations: list[Annotation] = field(default_factory=list)

    @staticmethod
    def key(condition: str, index: int) -> str:
        return f"{condition}|{index}"

    def status(self, condition: str, index: int) -> str:
        return self.statuses.get(self.key(condition, index), STATUS_OK)

    def set_status(self, condition: str, index: int, status: str) -> None:
        k = self.key(condition, index)
        if status == STATUS_OK:
            self.statuses.pop(k, None)
        else:
            self.statuses[k] = status

    def annotations_for(self, condition: str, index: int | None = None) -> list[Annotation]:
        out = [a for a in self.annotations if a.condition == condition]
        if index is not None:
            out = [a for a in out if a.detection_index == index]
        return out

    def add_annotation(self, ann: Annotation) -> None:
        self.annotations.append(ann)

    def remove_annotation(self, ann: Annotation) -> None:
        if ann in self.annotations:
            self.annotations.remove(ann)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "statuses": self.statuses,
            "annotations": [asdict(a) for a in self.annotations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Review":
        r = cls(statuses=dict(d.get("statuses", {})))
        r.annotations = [Annotation(**a) for a in d.get("annotations", [])]
        return r


class ReviewStore:
    def __init__(self, output_root: Path) -> None:
        self.path = Path(output_root) / "review" / "review.json"
        self.review = Review()
        if self.path.exists():
            try:
                self.review = Review.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
            except Exception:
                self.review = Review()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.review.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ------------------------------------------------------------------ #
    def export_response_csv(
        self, path: Path, times: np.ndarray, values: np.ndarray, channel: str, note: str = ""
    ) -> None:
        """Export one response waveform for the stimulator (functional electrical stimulation).

        Time is re-centred on the start of the exported window, matching the convention the
        pipeline uses for its own burst-envelope exports.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.asarray(times, dtype=float)
        v = np.asarray(values, dtype=float)
        n = min(t.size, v.size)  # a window clipped at the edge of the recording returns fewer samples
        if n == 0:
            raise ValueError("The selected window contains no samples.")
        t, v = t[:n], v[:n]
        df = pd.DataFrame({"time_s": t - t[0], "amplitude_uV": v})
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"# channel: {channel}\n")
            if note:
                fh.write(f"# note: {note}\n")
            df.to_csv(fh, index=False)

    def augment_metrics(self, metrics: pd.DataFrame, condition_col: str = "Configuration") -> pd.DataFrame:
        """Add the manual review columns to the pipeline's metrics table.

        Returns a NEW frame — the pipeline's own CSV is left untouched, so the automated
        result stays reproducible and the manual layer sits beside it.
        """
        if metrics.empty:
            return metrics
        out = metrics.copy()
        epoch_col = "Epoch" if "Epoch" in out.columns else None
        if epoch_col is None:
            return out

        def _status(row) -> str:
            return self.review.status(str(row[condition_col]), int(row[epoch_col]))

        def _notes(row) -> str:
            anns = self.review.annotations_for(str(row[condition_col]), int(row[epoch_col]))
            return " | ".join(a.text for a in anns)

        out["Review status"] = out.apply(_status, axis=1)
        out["Review notes"] = out.apply(_notes, axis=1)
        return out
