# EMG-SCS-flow

EMG processing for spinal cord stimulation (SCS/EES) protocols, for researchers and clinicians:
load a recording → parse the protocol annotations → detect responses → export tables, plots and
epoch files.

It runs two ways:

- **`python3 -m emgflow`** — the interactive toolbox
- **`python3 run.py <file>`** — the scripted pipeline, for reproducible batch runs.

The GUI does not reimplement anything: it calls the same `run_pipeline(...)`, renders the
pipeline's own outputs, and can export a runner script that reproduces a session headlessly.

> This repository is in active development. Always validate outputs against your clinical or
> research ground truth before making decisions.

---

## Two analysis modes

The two modes are near-disjoint code paths and answer different questions. Pick with
`startstop_mode=True/False`, or `--startstop` / `--no-startstop`.

| | **Stimulation-induced (SIR)** | **StartStop** |
|---|---|---|
| Input | a recruitment sweep: electrode configuration `X+Y` stimulated at a ramp of amplitudes | a protocol of named conditions, each with `start` … `stop` markers |
| Unit of work | a **crop** = one `(config, amplitude)` block | a **condition** |
| Detection | templates matched, then per-epoch onset / P1 / P2 | pre-computed template bank matched against every channel |
| Extras | recruitment across amplitudes | **spontaneous-EMG analysis**: RMS envelopes, bursts, antagonist co-activation |
| Question it answers | how does the response grow with stimulation amplitude? | what happened during this movement/task, and how active was each muscle? |

A StartStop run produces **no** SIR results, and no longer creates the SIR folder tree either —
see [Output](#output) for how the layout stays flat. Output roots produced before that change may
still carry empty trees from both modes.

---

## Input

- `.mat` (LabChart), `.fif`, `.edf`, and `.txt` text **curves** exports (see below).
- **Annotations are required** (except for curves exports). SIR mode expects labels encoding configuration and amplitude;
  StartStop expects `start`/`stop` markers (Cyrillic `старт`/`стоп` and typos are handled) plus a
  free-text label naming each condition.
- The stimulation channel must be identifiable as an artifact channel (a name containing `art`),
  or named explicitly in `ARTCHAN`. SIR needs it to centre epochs on the stimulus; StartStop
  tolerates its absence.
- Amplitude labels are treated as **opaque strings**: `2`, `2,0` and `02` are different crops and
  are never silently normalised.

Two LabChart quirks are handled: an empty placeholder channel that breaks the naive loader, and
headerless "MATLAB Level 4" exports without a `dataend` key — those are reconstructed block by
block, merged, and re-run from a temporary `.fif`.

### Text "curves" exports (pre-cut epochs)

Files from the clinical EP station (`N - curves count`, then one header + `Values, Volt` block per
curve) are **already epoched**: the station triggers on the stimulus, so each curve is one epoch
with sample 0 at the trigger. They are detected by a header sniff, not by extension alone, and
take a shortened SIR path — **artifact peak-finding and epoching are skipped** and the curves go
straight into PASS1/PASS2. Everything downstream (templates, onset/P1/P2 detection, gates, the
force/suppress system, plots, Excel) is the normal SIR code.

Consequences of what the format does *not* carry:

- **No channel names.** Channels are numbered by column position (`ch1`..`chN`). Column order does
  not imply muscle identity — that has to come from the recording protocol.
- **No configuration/amplitude labels.** The whole file is one crop: configuration = file stem,
  amplitude = `all`.
- **No artifact channel**, so `ARTCHAN` stays empty and no channel is excluded from the EMG set.
- **No pre-stimulus data**, so the standard windows do not fit and are adapted (only where the
  caller left them at their defaults, so explicit values still win): epoch = the curve's own span,
  response = `TEXT_CURVES_RESP_TMIN..TEXT_CURVES_RESP_TMAX` (2–40 ms), baseline = the station's
  constant pre-artifact pad. That pad is the only pre-artifact data there is; its mean is an exact
  DC estimate, and assuming a zero baseline instead would bias every peak amplitude by the
  channel's DC offset.
- **Raw filtering is skipped.** A 50 Hz notch at 20 kHz has a transition band far longer than one
  100 ms epoch, so filtering the concatenated curves would ring across their edges. These exports
  arrive filtered from the station.

The CLI, the scripted `run_pipeline` call and the GUI all take the same path — the GUI only adds
the file dialog filter and mirrors the adapted windows into its settings panel, so the values shown
are the values that run.

#### The three Neurosoft protocols

Per A. Militskova, the file name carries the stimulation level (`Th9-10`, `Th12-L1`, `C3-4` …)
followed by a token naming the test, and each test wants a different deliverable. `src/neurosoft.py`
resolves which one a file is *before* any analysis runs, and only that scenario's outputs are
produced — no template diagnostics, no per-amplitude box-plots, no averaging plot.

| Scenario | Name tokens | Deliverable (under `results/`) |
| --- | --- | --- |
| **Recruitment curve** | `RC`, `rec curve`, `kr rec`, `кр рек`, `КР` | `Recruitment/` — amplitude vs curve number per channel, top-N and amplitude-group box-plots, per-curve tables |
| **Jendrassik manoeuvre** | `JM`, `ендр`, `Ендрассик` | `Jendrassik/` — every curve drawn, grouped by amplitude with group mean ± SD, group statistics |
| **Paired stimulation** | `2 stim`, `2ст`, `двойная стим`, `парная`, `paired` | `Paired stimulation/` (same as Jendrassik), or the Condition-test outputs when the ISI is recoverable |

Matching is case-insensitive and tolerates the Cyrillic/Latin homoglyph mixing these names are full
of (`Тh11-12`, `JМ`). A file with **no** token falls back to the signal: a moving stimulus artifact
means paired-pulse, a fixed one means a recruitment sweep — the common case for bare `Th11-12.txt`.
When the name and the signal disagree the name wins and the run log says so.

Both plot panels (`Plots with grid and markers`, `Plots without grid and markers`) change shape for
these files: **the black mean line is dropped** and each curve is coloured by its position in the
sweep instead — **the later the curve, the darker it is** (light = first/weakest → dark =
last/strongest). Averaging the curves would destroy exactly the effect under study, since each curve
is already one stimulus. The y-axis is fitted to the post-artifact window, letting the stimulus
artifact (±10 mV against responses of ~0.1 mV) run off-scale.

The amplitude-group figures put **all channels on one figure** (`curves_by_amplitude_group.png`),
one subplot per channel with the groups overlaid — the groups are read by comparing them, and that
means having them side by side rather than in one file per channel.

In the GUI the scenario surface is one **interactive** tab that renames itself after whichever
scenario the run produced ("Recruitment curve" / "Jendrassik manoeuvre" / "Paired stimulation"),
since the three are mutually exclusive. It is driven by the per-curve metrics and the saved epochs,
not by the exported PNGs: pick a channel, click a point on the response-vs-curve plot (or drag the
slider) to pull that curve up bold with its markers, switch the colouring between curve order and
amplitude group, and read the group mean/SD table beside it.

The **Crop review** tab also draws these files properly: its recruitment panel falls back to the
curve number when the export carries no amplitude labels (previously it went blank), and the epoch
stack is coloured by curve order with the mean dropped, the same as the exported panels.

**Paired stimulation splits in two.** When the artifact position varies across curves the ISI is
real and recoverable, and the run goes to the Condition-test analysis (waterfalls, amplitude vs ISI,
persistence). Every tSCS `двойная стимуляция` export in the FCBRN dataset instead carries one
artifact per curve fixed at t=0 — the interval simply is not in the file — so those fall back to the
per-curve + amplitude-group deliverable and the log states why.

---

## Stimulation-induced (SIR) mode

Each `(config, amplitude, channel)` yields an evoked response with three markers — **onset**,
**P1** (first response peak, drawn red) and **P2** (the opposite rebound, green) — plus latency
and amplitude metrics.

Two passes:

1. **PASS 1 — templates.** The artifact-aligned epoch mean is matched against the template bank
   (and, with `SIR_TEMPLATE_PER_AMPLITUDE`, per amplitude rather than per configuration). The best
   match defines the expected onset / P1 / P2 latencies.
2. **PASS 2 — detect, gate, mark.** Peaks are searched near those latencies in each epoch, then
   filtered by amplitude thresholds and by channel-level consistency (valid fraction, median
   epoch–template correlation, inter-trial correlation, artifact-correlation rejection).

### Manual corrections

Real recordings need per-file tuning, and the pipeline exposes it as module-level globals — the
same ones the GUI writes when you click on a plot:

| Global | Effect |
|---|---|
| `SIR_FORCE_POS_P1_AFTER` / `SIR_FORCE_NEG_P1_AFTER` | force P1 onto a deflection of the chosen polarity; value is `min_lat` or `(min_lat, max_lat)` |
| `SIR_FORCE_KEYS` | whitelist a `(config, amp, channel)` — bypass the consistency gates |
| `SIR_SUPPRESS_KEYS` | force-drop a false positive no threshold separates |
| `SIR_EXCLUDE_CHANNELS` / `SIR_EXCLUDE_CONFIGS` | drop a whole channel / configuration |
| `SIR_P1_DOMINANT` | P1 = the dominant deflection instead of the template's first marker |
| `SIR_ALIGN_ONSET`, `SIR_ALIGN_SKIP_XCORR` | centre epochs on the artifact's onset; skip the cross-correlation that locks onto the largest artifact peak |

Keys have three precedence levels — `"channel"` < `("config","channel")` < `("config","amp","channel")` —
and corrections are **additive**: a correction changes only the `(config, amp, channel)` it names.

---

## StartStop mode

Conditions are split into `start` segments (the `stop` spans are collected but not analysed) and
concatenated. Detection is **template matching against a pre-computed bank** (`templates/`, 26
templates at 2 kHz, centre sample 200 = *t* = 0): each template is tried at several time scales and
both polarities, per channel, by sliding normalised cross-correlation.

Three levels of gate, in order:

1. **Leakage rejection** — a detection whose ±40 ms window around P1 correlates with more than
   three other channels is stimulus or movement leakage, not a muscle response.
2. **Channel wipe** — a channel whose detections do not correlate with *each other* is firing on
   irregular bursts, not on a stereotyped response.
3. **Per-epoch consistency** — inside a kept channel, individual inconsistent detections are
   dropped.

Every rejected candidate is written out **with its reason**, which is what makes "why was nothing
found here?" answerable.

### Spontaneous EMG (runs inside StartStop)

Independent of peak detection — the "how active is this muscle" view:

- windowed RMS and mean amplitude, plus a smooth RMS envelope;
- **burst detection** on the envelope: a channel has bursts only if the envelope peak clears both a
  signal-to-noise ratio *and* an absolute floor in µV — the ratio alone cannot separate an active
  muscle from a low-amplitude noisy channel;
- per-burst envelopes exported as `.txt`, **re-centred on the burst midpoint** — the form the
  stimulator wants for functional electrical stimulation;
- **antagonist co-activation**: for each pair in `ANTAGONIST_PAIRS`, bursts of the two muscles are
  merged into episodes and plotted as an **L-shape** (one point per episode, normalised RMS of
  muscle A against muscle B) — a reciprocal pattern hugs the axes, co-contraction fills the corner.

A high-pass is applied **inside this analysis only**: it is needed against baseline drift, but it
distorts the stereotyped peak shapes that detection relies on.

---

## Output

By default results go **next to the recording, in a folder named `<subject> <file>`** —
`(Ж14)/Th11-12.txt` → `(Ж14)/(Ж14) Th11-12/`. The subject prefix is the containing folder's name,
which in this dataset is the subject code; it is redundant while you sit inside that folder and
essential the moment you do not, since half the subjects have a `Th11-12.txt` and the results
folders would otherwise be indistinguishable in the GUI's run list or once copied out. (The prefix
is skipped when the file name already starts with it.) `--output-dir` overrides the whole thing.

Two things keep the tree flat:

- Directories are created **lazily**, as something is written into them. A SIR run no longer leaves
  an empty `StartStop analysis/` tree behind (and vice versa), and a scenario that emits no
  box-plots leaves no `Boxplots/` folder. An existing folder means that analysis actually ran.
- The **`<mode>/` level is dropped when it would be the only one**. It exists to keep two analyses
  apart inside one output root, and a run produces exactly one of them — so results land straight in
  `results/`. The level comes back only if the root already holds a different analysis, where the
  names would collide (both modes write `Excel/Large_dataset_emg_response_metrics.csv`). Readers
  accept either layout, so older output roots still open.

```
<output_root>/
├── data/                        original + preprocessed .fif  (SIR also: annot_crops_fif/)
└── results/                     ← one run: no mode folder. Mixed root: nested under
    │                              Stimulation-induced responses/ | StartStop analysis/ |
    │                              Condition test/
    │  SIR:
    ├── Stimulus-centered epochs/     <config>_<amp>-epo.fif
    ├── Excel/                        per-epoch metrics + summary stats
    ├── Plots with / without grid and markers/, Plots grouped by amplitude/
    ├── Boxplots/, Templates/, Template overlays/
    ├── Recruitment/ | Jendrassik/ | Paired stimulation/   (Neurosoft .txt: one of the three)
    │  StartStop:
    ├── Excel/                        metrics, summary, and three diagnostic tables:
    │                                 accepted anchors, discarded anchors + reason, channel QC
    ├── Detections raw/               <condition>_detections_raw.fif — the recording with one
    │                                 annotation per detection (channels merged: "ECR L+TR R")
    ├── Raw epochs/, Plots …/, Boxplots/, Templates/
    └── Spontaneous EMG/<condition>/
        ├── Excel/                    summary, per-window detail, bursts, co-activation episodes
        ├── Envelopes/                per-burst RMS envelopes (.txt)
        └── Plots/                    overview with bursts, envelope overlays, L-shapes
```

Key files: `Excel/Large_dataset_emg_response_metrics.csv` (per-epoch metrics) and
`Excel/Summary_stats_by_config_amp_channel.csv`.

Note these metric CSVs reach 50–120 MB because of the `Time series` column, which stores every
epoch's waveform. Read them with `usecols` if you only need the latencies.

Amplitudes in the tables are in **volts**, latencies in **seconds**.

---

## Settings

Everything tunable lives in `src/constants.py`, grouped and documented: filtering, artifact
detection, epoch and response windows, re-referencing, the SIR template/alignment/gate parameters,
the StartStop matching and gate thresholds, and the spontaneous-EMG and burst parameters. Tune
there, re-run on a known validation set, and compare.

Caveats worth knowing before you go looking for a knob:

- `pipeline.py` does `from .constants import (...)`, binding **by value at import**. A runner that
  overrides a setting must patch `src.pipeline.<NAME>`, not `src.constants.<NAME>`.
- `SPONTANEOUS_EMG_WINDOW_MS`, `_ENV_HOP_MS` and `_UV_SCALE` are bound as *default arguments* of
  `_run_spontaneous_emg_analysis`, whose caller never passes them — a module-global patch does not
  reach them.
- The leakage thresholds (±40 ms, corr 0.7, >3 channels) are hardcoded locals.
- `STARTSTOP_ONSET_TMIN`, `STARTSTOP_ONSET_TMAX` and `STARTSTOP_RESP_TMAX` are imported but never
  read — changing them does nothing.

---

## Layout

```
src/
├── pipeline.py     run_pipeline(...), both modes, detection, marker logic, per-file globals
├── constants.py    every tunable, with its default and the reasoning behind it
├── annotations.py  protocol parsing: config/amplitude labels, start/stop segments
├── detection.py    onset and peak primitives shared by both modes
├── plotting.py     panels, overlays, boxplots, spontaneous plots, L-shapes
├── io_utils.py     output tree
templates/          the 26-template bank (.npy + onset/peak indices in .npz)
emgflow/            the interactive toolbox  → emgflow/README.md
run.py              CLI
```

## Requirements

Python 3.9+, `pip install -r requirements.txt`. The toolbox additionally needs `PySide6`.
