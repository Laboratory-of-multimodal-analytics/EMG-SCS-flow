# EMG-SCS-flow GUI

Interactive front end for the pipeline. Replaces the two manual loops:

- the **runner-per-file** loop in SIR mode (edit globals in a script → run → open PNGs → repeat),
- the **command-line** loop the clinicians were shown in StartStop mode.

```bash
cd /path/to/EMG-SCS-flow
python -m gui
```

Needs `PySide6` on top of `requirements.txt`.

## What it is (and is not)

The GUI **never reimplements detection**. It calls the real `run_pipeline(...)` and renders
the pipeline's own outputs, so what you see is exactly what a scripted run produces. Every
interactive edit is stored as the same module globals the pipeline already reads — which is
why **File → Export runner script** can emit a standalone `.py` that reproduces the session
headlessly.

## Opening work

- **Open recording…** — pick a `.mat` / `.fif` / `.edf`, choose the mode, **Run**. The progress
  bar is driven by the pipeline's own tqdm loops (crops in SIR, conditions in StartStop), so it
  shows real counts rather than spinning.
- **Open results folder…** — open an output root that already has results, **without re-running**.
  The mode is detected from the *content* (both mode folders exist on every run, so an empty
  `Stimulation-induced responses/` proves nothing). If the GUI produced the run, its
  `review/session.json` is restored with it, edits and all.
- **Processed recordings…** — scan a folder (e.g. `Coding/results/`) and list every run that
  already has results: mode, crops or conditions, metrics-table size, whether spontaneous EMG
  ran, and when. Double-click to open. The scan runs in the background; the *first* scan of a
  Google Drive folder is slow because Drive fetches every directory listing over the network,
  after which it is instant.

## The surfaces

| Tab | Mode | What you do |
|---|---|---|
| **Crop review (SIR)** | Stimulation-induced | Walk the `(config, amplitude)` crops. Crops with **0 detections are red**. Click a peak to force P1; reject false positives; drag out a response and turn it into a template. Recruitment curves sit beside the plot. |
| **Epoch browser (StartStop)** | StartStop | Three views of a condition: **Detection** (one response in context), **Overlay** (all detections of a channel superimposed, as in SIR — that is how you see whether a response is stereotyped or the detector is firing on noise), and **Segment** (the whole condition with bursts shaded and their RMS envelopes drawn on top). Flag, annotate, export. |
| **Spontaneous EMG** | StartStop | RMS envelopes with the detected bursts; export a burst envelope. |
| **Raw** | any | Scroll through the whole recording. Overlay protocol annotations, detections, bursts and envelopes; jump straight to any detection or burst; or fit the entire segment into one window. |

`Settings` is generated from the lever registry, grouped as in `constants.py`, and swaps
its contents with the mode.

### How things are drawn

Markers are **points, one per epoch, on that epoch's own trace** — onset blue, P1 red, P2
green, exactly as `plotting.plot_epochs_panel` draws them. A line at the median would hide
how consistent the detection actually was across trials. (If a run leaves `Onset latency`
empty, as SIR runs currently do, there are simply no blue points.)

The SIR channel column **fits every channel on screen without scrolling**: SIR epochs are
short, so the time axis can be squeezed and the column kept narrow. Each row is scaled to
its own post-stimulus response, because the stimulus artifact and pre-stimulus drift are
often an order of magnitude larger and would flatten the deflection being judged.

Recruitment is shown as **curves, not a table**: peak-to-peak against amplitude, one line
per channel, with a 95 % CI band across the epochs of each crop (a single-trial point gets
a zero-width band rather than a fabricated interval). A histogram view is a dropdown away.
The pipeline still writes its own summary tables — this is for looking.

### Crop review — Markers mode

| Action | Effect | Global it writes |
|---|---|---|
| click a peak | force P1 onto it, polarity from the dropdown (or the sign under the cursor) | `SIR_FORCE_POS_P1_AFTER` / `SIR_FORCE_NEG_P1_AFTER` |
| Shift+click, or double-click a table row | reject as a false positive | `SIR_SUPPRESS_KEYS` |
| Ctrl+click | whitelist — bypass the consistency gates | `SIR_FORCE_KEYS` |
| Alt+click | exclude the whole channel | `SIR_EXCLUDE_CHANNELS` |
| right-click | clear the forced marker | — |

**Rejection is immediate and everywhere at once**: the markers vanish from the plot, the channel
row in the table drops to 0 detections and reads `rejected`, the crop's count in the list falls,
and the recruitment table and the exported metrics agree. No re-run needed for the numbers to
match what you see — and because it is stored as `SIR_SUPPRESS_KEYS`, the next real run
reproduces the same decision.

Edits are written at the most specific `(config, amp, channel)` precedence level, so they are
**additive**: forcing or rejecting on one crop cannot alter a crop you did not name. Clicking left
of `t = 0` forces a peak *before* the stimulus, which is allowed and sometimes correct.

“Bind P1 to ±3 ms” writes `(min_lat, max_lat)` so P1 cannot drift to a stronger later peak;
unchecked, it writes a bare `min_lat` and P1 becomes the dominant deflection at or after the click.

A forced marker is drawn explicitly: the bound window (or the minimum latency) is shaded in
purple, labelled with the latency in ms, and an arrow shows which way P1 was forced to point.

### Crop review — Template mode

Switch to **Template**, drag over the response you want found (the selection stays on screen),
and press *Make template from selection…*. An editor opens showing the template with its
auto-placed onset / P1 / P2 — **click anywhere on the trace to move whichever marker is
selected**, since the whole point of drawing a template by hand is that you disagree with the
automatic answer somewhere. It warns if the markers fall out of order.

The selected span of the epoch mean is resampled onto the bank's native 2 kHz grid
(index 200 = `t = 0`), flattened outside the selection, and saved into a session-local bank
under `<output>/templates/`; the pipeline is pointed at it — **both modes** load their bank
through the same function, so a template made here drives StartStop matching too.

*Detect by my templates only* gives a bank containing nothing but your templates — that is how you
say “find me this response and nothing else”. Unchecked, yours are added on top of the 26 stock
ones. The bank path is carried in `session.json` and in the exported runner script, so a run made
with a custom template stays reproducible.

## Re-running

“Apply edits and re-run” re-runs **the whole file**. The pipeline has no per-crop entry point,
so a genuinely incremental re-run would need `src/pipeline.py` refactored — this is the honest
limitation, not a stub. (Rejecting a false positive does *not* need a re-run; making a template
does, since detection itself changes.)

## Files it writes

Inside the output root, in a `review/` folder that is never touched by the pipeline:

- `session.json` — the settings + edits that produced the current outputs (written after each run)
- `review.json` — clinician flags and free-text annotations, so they survive a re-run and the
  patient's next visit
- exported response CSVs (`time_s`, `amplitude_uV`)
- `metrics_with_review.csv` via *File → Export reviewed metrics* — the pipeline's own metrics
  table plus `Review status` / `Review notes`, written as a **new** file so the automated result
  stays reproducible

## Things worth knowing

- `pipeline.py` does `from .constants import (...)`, binding by value at import. The GUI
  therefore patches `src.pipeline.<NAME>`; patching `src.constants` would do nothing.
- `SPONTANEOUS_EMG_WINDOW_MS`, `_ENV_HOP_MS` and `_UV_SCALE` are bound as *default arguments*
  of `_run_spontaneous_emg_analysis` and its caller never passes them, so no global patch can
  reach them. The GUI rebinds the function's defaults (`gui/session.py: rebind_default`, with
  the constant→parameter map, since the names differ: `window_ms`, `hop_ms`, `uv`).
- The leakage thresholds (±40 ms, corr 0.7, >3 channels) and the `borderline_valid_very_low_corr`
  rule are hardcoded locals in `pipeline.py`, so they are **not** exposed as controls.
- The metrics CSVs run to **50–120 MB** because of the `Time series` column. The scanner never
  reads them (it works from directory metadata alone), and loading one skips that column at parse
  time — waveforms come from the epoch `.fif` files instead.
- `STARTSTOP_ONSET_TMIN` / `_ONSET_TMAX` / `_RESP_TMAX` are imported by the pipeline but never
  read — deliberately not shown, since a control for them would be a lie.
- `Detections raw/<cond>_detections_raw.fif` holds the concatenated start segment, but its
  annotation onsets stay in the **original** recording's time base. `results.py` subtracts
  `raw.first_time`; without that every marker lands past the end of the file.
- Amplitude labels are opaque strings (`2` ≠ `2,0` ≠ `02`) and are never normalised.

## Layout

```
gui/
├── settings_spec.py   # the lever registry: every knob, its type, group, and HOW it reaches the pipeline
├── session.py         # settings + edits; applies them; JSON round-trip; runner-script export
├── runner.py          # runs run_pipeline off the UI thread; streams its log and tqdm progress
├── results.py         # reads what the pipeline wrote; finds already-processed runs on disk
├── templates.py       # builds a bank-format template from a selected response
├── review_store.py    # clinician flags / annotations / CSV export (JSON sidecar)
├── main_window.py     # shell: file, mode switch, tabs, progress, log
└── widgets/
    ├── settings_panel.py
    ├── sir_viewer.py
    ├── startstop_viewer.py
    ├── spontaneous_viewer.py
    └── database_dialog.py
```
