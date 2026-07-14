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

A StartStop run produces **no** SIR results — but `build_output_dirs` creates both folder trees
on every run, so an empty `Stimulation-induced responses/` folder proves nothing about which
mode ran.

---

## Input

- `.mat` (LabChart), `.fif`, `.edf`.
- **Annotations are required.** SIR mode expects labels encoding configuration and amplitude;
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

```
<output_root>/
├── data/<mode folder>/          original + preprocessed .fif  (SIR also: annot_crops_fif/)
└── results/
    ├── Stimulation-induced responses/
    │   ├── Stimulus-centered epochs/     <config>_<amp>-epo.fif
    │   ├── Excel/                        per-epoch metrics + summary stats
    │   ├── Plots with / without grid and markers/, Plots grouped by amplitude/
    │   ├── Boxplots/, Templates/, Template overlays/
    └── StartStop analysis/
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
