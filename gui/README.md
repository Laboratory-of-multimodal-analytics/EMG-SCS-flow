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

## The three surfaces

| Tab | Mode | What you do |
|---|---|---|
| **Crop review (SIR)** | Stimulation-induced | Walk the `(config, amplitude)` crops. Crops with **0 detections are red** in the list. Click a peak to force P1 onto it. |
| **Epoch browser (StartStop)** | StartStop | Walk the detected responses, flag bad / false-positive / complex ones, drag out a window and annotate it, export the waveform to CSV for the stimulator. |
| **Spontaneous EMG** | StartStop | RMS envelopes with the detected bursts; export a burst envelope. |

`Settings` is generated from the lever registry, grouped as in `constants.py`, and swaps
its contents with the mode.

### Crop review — mouse map

| Action | Effect | Global it writes |
|---|---|---|
| click a peak | force P1 onto it, polarity from the dropdown (or the sign under the cursor) | `SIR_FORCE_POS_P1_AFTER` / `SIR_FORCE_NEG_P1_AFTER` |
| Shift+click | suppress this detection | `SIR_SUPPRESS_KEYS` |
| Ctrl+click | whitelist — bypass the consistency gates | `SIR_FORCE_KEYS` |
| Alt+click | exclude the whole channel | `SIR_EXCLUDE_CHANNELS` |
| right-click | clear the forced marker | — |

Edits are written at the most specific `(config, amp, channel)` precedence level, so they are
**additive**: forcing a peak on one crop cannot alter a crop you did not name. Clicking left of
`t = 0` forces a peak *before* the stimulus, which is allowed and sometimes correct.

“Bind P1 to a ±3 ms window” writes `(min_lat, max_lat)` so P1 cannot drift to a stronger later
peak; unchecked, it writes a bare `min_lat` and P1 becomes the dominant deflection at or after
the click.

## Re-running

“Apply edits and re-run” re-runs **the whole file**. The pipeline has no per-crop entry point,
so a genuinely incremental re-run would need `src/pipeline.py` refactored — this is the honest
limitation of v1, not a stub.

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
├── runner.py          # runs run_pipeline off the UI thread, streams its log
├── results.py         # reads what the pipeline wrote (crops, detections, envelopes, QC tables)
├── review_store.py    # clinician flags / annotations / CSV export (JSON sidecar)
├── main_window.py     # shell: file, mode switch, tabs, log
└── widgets/
    ├── settings_panel.py
    ├── sir_viewer.py
    ├── startstop_viewer.py
    └── spontaneous_viewer.py
```
