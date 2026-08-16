"""Which of the five Neurosoft protocols a text ``curves`` export holds.

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
     Paired-pulse test. NOTE: in this dataset these exports show a single
     stimulus artifact fixed at ~1 ms in a 100 ms window — no second pulse is
     visible even at an 8 % threshold or in the average across curves — so the
     inter-stimulus interval cannot be recovered from the file. They get the same
     deliverable as (2).

  4. CONDITION    (recognised from the SIGNAL, not the name)
     A. Militskova: "необходимо считать ответ после артефакта прямо на кривой,
     сам ответ как бы сдвигается по кривой". The station triggers on a fixed
     event and the stimulus follows at a programmed delay that steps from curve
     to curve, so the artifact — and the response behind it — slides along a
     longer epoch (250 ms rather than 100). Names carry the conditioning
     modality instead of the protocol (ТМС, ткмс, нерв, ulnaris, "в сочетании"),
     which is why this one is detected from the data. Deliverable: response
     measured RELATIVE to the artifact, grouped by inter-stimulus interval.

  5. HREFLEX     ("H reflex", "Н рефлекс", or a muscle name on its own)
     Peripheral-nerve stimulation with TWO responses on the same curve: the
     direct motor response (M) and, ~25-30 ms behind it, the monosynaptic
     reflex response (H). Everything the other scenarios measure once is
     measured twice here — onset, P1, P2 and PTP for M and for H separately —
     because the pair is the measurement: H grows and then RECEDES as M grows,
     and a detector that takes "the response" on such a curve reports whichever
     of the two happens to be larger at that intensity, silently mixing the two
     down one column. In the FCBRN dataset that is 75 of the 415 files, and 23
     of them carry exactly that complaint in the processing journal.

Anything else in these folders (SEP/ССВП, TMS, n. ulnaris) is out of scope and
keeps the generic path.

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
CONDITION = "condition"
HREFLEX = "hreflex"

#: Human-readable scenario names for logs and figure titles.
SCENARIO_LABELS = {
    RECRUITMENT: "Recruitment curve",
    JENDRASSIK: "Jendrassik manoeuvre",
    PAIRED: "Paired stimulation",
    CONDITION: "Condition test",
    HREFLEX: "H-reflex",
}

#: Every scenario a caller may pin, in one place so the CLI, the GUI selector and
#: ``run_pipeline``'s validation cannot drift apart.
SCENARIOS = (RECRUITMENT, JENDRASSIK, PAIRED, CONDITION, HREFLEX)

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

_HR_LAT = re.compile(r"h[\s\-_]*reflex|\bhreflex\b")

# Cyrillic tokens — matched on the plain lower-cased name.
_PAIRED_CYR = re.compile(r"\b2\s*ст|двойн\w*\s*стим\w*|парн\w*\s*стим|\bпарн")
_JM_CYR = re.compile(r"ендр")
_RC_CYR = re.compile(r"\bкр\b|кр\s*рек\w*|кривой\s*рекрут")
#: "Н рефлекс" with a Cyrillic Н — never folded, or it would read as Latin "H".
_HR_CYR = re.compile(r"\bн[\s\-_]*рефл\w*|\bн[\s\-_]*reflex")

#: The muscle a peripheral-nerve run was recorded from, standing in for the
#: protocol. Half of these files are named after nothing else — ``сол лев.txt``,
#: ``GM Right JM.txt``, ``челюсть сол прав.txt`` — because to the clinician who
#: named them "soleus" already means the H-reflex study; nothing else is
#: recorded from a single muscle over a stimulation ramp of a peripheral nerve.
#:
#: Read off the dataset rather than guessed: these patterns select exactly the
#: 75 files the processing journal marks as H-reflex out of all 415, with no
#: false positive and none missed.
#:
#: Split Latin/Cyrillic like every other token here, and for the same reason:
#: folding maps ``с`` and ``о`` onto Latin, so ``сол`` folds to ``col`` and a
#: single pattern over the folded name cannot see it.
_HR_MUSCLE_LAT = re.compile(r"\bsol\b|soleus|\bgm\b|\bfcu\b")
_HR_MUSCLE_CYR = re.compile(r"\bсол\w*|гастри\w*|камбал\w*")


def _stem(path: str | Path) -> str:
    """Bare name without the .txt extension.

    Not ``Path.stem``: these names are full of dots — the crop label the pipeline
    builds replaces every hyphen with one, so ``Т11-12 ендрассик 80 и 90мА``
    becomes ``Т11.12 ендрассик 80 и 90мА`` and ``Path.stem`` reads everything
    after ``Т11`` as an extension and throws the intensities away.
    """
    name = Path(path).name
    return name[:-4] if name.lower().endswith(".txt") else name


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
    name = _fold(_stem(path))
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

    H-reflex comes before Jendrassik and recruitment, and beats both. Those two
    name a PROTOCOL — how the stimuli were delivered — while H-reflex names what
    is on the curve, and a file is routinely both (``Н рефлекс СОЛ лев
    Ендрассик``, ``H reflex rec curve сол прав``). Which one the run must follow
    is settled by what would be lost: measured as a Jendrassik file, the M and H
    responses collapse into one column and the comparison the file exists for is
    gone. The protocol is not lost — it is kept in ``protocol_from_name`` and
    drives the deliverable.
    """
    stem = _stem(path)
    raw = stem.lower()
    folded = _fold(stem)

    if _PAIRED_LAT.search(folded) or _PAIRED_CYR.search(raw):
        return PAIRED
    if (_HR_LAT.search(folded) or _HR_CYR.search(raw)
            or _HR_MUSCLE_LAT.search(folded) or _HR_MUSCLE_CYR.search(raw)):
        return HREFLEX
    if _JM_LAT.search(folded) or _JM_CYR.search(raw):
        return JENDRASSIK
    if _RC_LAT.search(folded) or _RC_CYR.search(raw):
        return RECRUITMENT
    return None


def protocol_from_name(path: str | Path) -> str:
    """How the stimuli were delivered, for a file whose scenario is H-reflex.

    An H-reflex run is still driven by one of the ordinary protocols — a ramp of
    intensities, or a block with the Jendrassik manoeuvre against a block
    without. ``scenario_from_name`` deliberately answers HREFLEX for both, so the
    protocol is read separately and decides the deliverable: a ramp gets the
    recruitment curves, a Jendrassik run additionally gets its amplitude groups.
    """
    stem = _stem(path)
    raw = stem.lower()
    folded = _fold(stem)
    if _JM_LAT.search(folded) or _JM_CYR.search(raw):
        return JENDRASSIK
    return RECRUITMENT


def detect_scenario(
    path: str | Path,
    data=None,
    sfreq: float | None = None,
) -> tuple[str, dict]:
    """Resolve the scenario for one Neurosoft export.

    A stepping artifact settles it first: that shift IS the Condition test, and
    its files are named after the conditioning modality rather than the protocol,
    so the name cannot answer for them. Otherwise the name wins — that is the
    convention the protocol is recorded in — and a name with no token falls back
    to a recruitment sweep, the common case for a bare ``Th11-12.txt``.

    A name/signal disagreement is reported in the info dict so the run log can
    show it instead of silently picking one.
    """
    named = scenario_from_name(path)
    info: dict = {"from_name": named, "from_signal": None, "conflict": False,
                  "protocol": protocol_from_name(path)}

    moving = False
    n_isi = 1
    if data is not None and sfreq:
        from .condition import group_conditions, is_condition_paradigm

        moving, cond_info = is_condition_paradigm(data, float(sfreq))
        info["signal"] = cond_info
        _, isi = group_conditions(cond_info["positions_ms"])
        n_isi = len(isi)
        info["n_isi"] = n_isi
        info["from_signal"] = CONDITION if (moving and n_isi > 1) else None

    # An H-reflex named in the file beats the signal. The signal argument for
    # Condition is that its files are named after the conditioning modality and
    # so cannot be asked — but "Н рефлекс СОЛ лев" names neither a modality nor
    # a conditioning stimulus, it names the study. These files do sometimes read
    # as a stepping artifact (a stimulator whose delay drifts, a broad M-wave
    # taken for an artifact), which is how four of them were classified as
    # Condition tests and had to be re-run by hand.
    if named == HREFLEX:
        info["conflict"] = bool(moving and n_isi > 1)
        info["scenario"] = HREFLEX
        info["reason"] = "file name (H-reflex wins over the signal)"
        return HREFLEX, info

    # The Condition test is DEFINED by the shift, so the signal decides it —
    # its files are named after the conditioning modality (ТМС, нерв, ulnaris),
    # never after the protocol, so the name cannot be asked.
    if moving and n_isi > 1:
        info["conflict"] = bool(named is not None and named != CONDITION)
        info["scenario"] = CONDITION
        info["reason"] = f"artifact steps across curves ({n_isi} intervals)"
        return CONDITION, info

    if named is not None:
        info["scenario"] = named
        info["reason"] = "file name"
        return named, info

    info["scenario"] = RECRUITMENT
    info["reason"] = "no protocol token in the name; fixed artifact -> recruitment sweep"
    return RECRUITMENT, info
