"""Which of the three Neurosoft protocols a text ``curves`` export holds.

Applies ONLY to the Neurosoft station's .txt exports. Per A. Militskova, the
file name carries the stimulation level (Th9-10, Th12-L1, C3-4 ...) followed by
a token naming the test, and each test wants a different deliverable:

  1. RECRUITMENT  ("RC", "rec curve", "kr rec", "кр рек", "КР")
     Recruitment curve, up to 100 mA. Every curve is one stimulus of increasing
     intensity. Deliverable: how the amplitude grows with stimulus strength —
     the same picture the .mat tests produce.

  2. JENDRASSIK   ("JM", "ендр", "Ендрассик")
     Jendrassik manoeuvre: response change under voluntary contraction of other
     muscles, usually one or two intensities with many repeats. Deliverable:
     the curves themselves, plus amplitude groups with their means and spreads.

  3. PAIRED       ("2 stim", "2ст", "двойная стим", "парная", "paired")
     Paired-pulse test: spinal inhibition as a function of the pause between the
     two stimuli. Deliverable: as for (2), grouped by inter-stimulus interval.

Anything else in these folders (H-reflex, SEP/ССВП, TMS, n. ulnaris) is out of
scope and keeps the generic path.

Naming is inconsistent: Latin and Cyrillic mix inside one token (``Тh11-12``,
``JМ``), case varies, and Cyrillic homoglyphs stand in for Latin letters. So the
Latin tokens are matched on a homoglyph-FOLDED copy of the name, and the
Cyrillic tokens on the plain lower-cased name — folding would destroy them
(``Ендр`` -> ``Eндр``).
"""

from __future__ import annotations

import re
from pathlib import Path

RECRUITMENT = "recruitment"
JENDRASSIK = "jendrassik"
PAIRED = "paired"

#: Human-readable scenario names for logs and figure titles.
SCENARIO_LABELS = {
    RECRUITMENT: "Recruitment curve",
    JENDRASSIK: "Jendrassik manoeuvre",
    PAIRED: "Paired stimulation",
}

#: Cyrillic letters visually identical to Latin ones, so "Тh11-12 JМ" folds to
#: "th11-12 jm" and one Latin pattern covers every spelling.
_HOMOGLYPHS = str.maketrans(
    "АВЕКМНОРСТУХасеорхуі",
    "ABEKMHOPCTYXaceopxyi",
)

# Latin tokens — matched on the folded name.
_PAIRED_LAT = re.compile(r"2\s*stim|\bpaired\b|\bparied\b|double\s*stim")
_JM_LAT = re.compile(r"\bjm\b|jendrassik")
_RC_LAT = re.compile(r"\brc\b|rec\s*curve|recru[it]tment|recrutment|\bkr\b")

# Cyrillic tokens — matched on the plain lower-cased name.
_PAIRED_CYR = re.compile(r"\b2\s*ст|двойн\w*\s*стим\w*|парн\w*\s*стим|\bпарн")
_JM_CYR = re.compile(r"ендр")
_RC_CYR = re.compile(r"\bкр\b|кр\s*рек\w*|кривой\s*рекрут")


def _fold(name: str) -> str:
    """Lower-case *name* with Cyrillic homoglyphs mapped onto Latin."""
    return name.translate(_HOMOGLYPHS).lower()


# Stimulation intensities written into the name: "80 и 90мА", "90 and 100mA",
# "45mA", "100 мА". Anchored on the mA unit — a bare number is a spinal level
# (Th11-12) or a date, not a current.
_MA = r"(?:m[aа]|м[aа])"
#: "и" / "and" / "amd" (a typo that occurs in these names) / "+" / "/" / ","
_JOIN = r"(?:и|and|amd|\+|/|,)"
_TWO_INTENSITIES = re.compile(rf"(\d{{2,3}})\s*{_JOIN}\s*(\d{{2,3}})\s*{_MA}\b")
_ONE_THEN_MORE = re.compile(rf"(\d{{2,3}})\s*{_MA}\b\s*{_JOIN}\s*(\d{{2,3}})\b")
#: A joined pair with no unit at all ("ендр 90 и 100"). Safe without the mA
#: anchor only because both numbers must be >= 20: spinal levels never are
#: (Th1-12, L1-5, C3-7), so a level cannot be mistaken for a current.
_BARE_PAIR = re.compile(rf"\b([2-9]\d|1\d\d)\s*{_JOIN}\s*([2-9]\d|1\d\d)\b")
#: A lone number needs the unit — on its own it could be anything.
_SINGLE_INTENSITY = re.compile(rf"(\d{{2,3}})\s*{_MA}\b")


def intensities_from_name(path: str | Path) -> list[int]:
    """Stimulation intensities (mA) named in the file, low -> high.

    A Jendrassik run is done "at one or two intensities", and the name is the
    only place that number is recorded — the export itself carries no amplitude
    labels. It decides how many response-amplitude clusters to look for: at each
    intensity the patient gets a block of plain test stimuli and a block with the
    manoeuvre, so two named intensities mean four groups, not two.

    Returns [] when the name says nothing, which is the common case.
    """
    name = _fold(Path(path).stem)
    found: list[int] = []
    for rx in (_TWO_INTENSITIES, _ONE_THEN_MORE, _BARE_PAIR):
        for m in rx.finditer(name):
            found += [int(m.group(1)), int(m.group(2))]
        if found:
            break
    if not found:
        found = [int(m.group(1)) for m in _SINGLE_INTENSITY.finditer(name)]
    # 0 mA is not an intensity; drop duplicates and keep the order low -> high.
    return sorted({v for v in found if 0 < v <= 200})


def scenario_from_name(path: str | Path) -> str | None:
    """Scenario implied by the file name, or None when no token is present.

    Paired is tested first: ``Т11-12 2ст`` also contains an RC-looking ``ст``
    fragment, and the paired token is the more specific of the two.
    """
    stem = Path(path).stem
    raw = stem.lower()
    folded = _fold(stem)

    if _PAIRED_LAT.search(folded) or _PAIRED_CYR.search(raw):
        return PAIRED
    if _JM_LAT.search(folded) or _JM_CYR.search(raw):
        return JENDRASSIK
    if _RC_LAT.search(folded) or _RC_CYR.search(raw):
        return RECRUITMENT
    return None


def detect_scenario(
    path: str | Path,
    data=None,
    sfreq: float | None = None,
) -> tuple[str, dict]:
    """Resolve the scenario for one Neurosoft export.

    The file name wins whenever it carries a token — that is the convention the
    protocol is actually recorded in. The signal is consulted only when the name
    is silent (a bare ``Th11-12.txt``): a stimulus artifact that MOVES from curve
    to curve means paired-pulse, a fixed one means a recruitment sweep (the
    common default for unlabelled files).

    A name/signal disagreement is reported in the info dict so the run log can
    show it instead of silently picking one.
    """
    named = scenario_from_name(path)
    info: dict = {"from_name": named, "from_signal": None, "conflict": False}

    signal_paired = False
    if data is not None and sfreq:
        from .condition import is_condition_paradigm

        signal_paired, cond_info = is_condition_paradigm(data, float(sfreq))
        info["signal"] = cond_info
        info["from_signal"] = PAIRED if signal_paired else None

    if named is not None:
        # A moving artifact on a file named RC/JM is worth flagging: it usually
        # means the export mixes protocols, or a muscle channel is louder than
        # the stimulus channel and is being read as the artifact.
        info["conflict"] = bool(signal_paired and named != PAIRED)
        info["scenario"] = named
        info["reason"] = "file name"
        return named, info

    if signal_paired:
        info["scenario"] = PAIRED
        info["reason"] = "moving stimulus artifact"
        return PAIRED, info

    info["scenario"] = RECRUITMENT
    info["reason"] = "no protocol token in the name; fixed artifact -> recruitment sweep"
    return RECRUITMENT, info
