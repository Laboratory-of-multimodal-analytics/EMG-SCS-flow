# EMG-SCS-flow GUI

Interactive front end for the pipeline. Replaces the two manual loops:

- the **runner-per-file** loop in SIR mode (edit globals in a script → run → open PNGs → repeat),
- the **command-line** loop the clinicians were shown in StartStop mode.

```bash
cd /path/to/EMG-SCS-flow
python -m emgflow
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
- **Process several…** — a queue of recordings processed one after another: add files, or a
  whole folder (results folders, `*_annotations.txt` and non-Neurosoft text files are skipped).
  Each row shows what the file is, its mode (SIR / StartStop; Neurosoft exports always SIR) and
  whether it already has results; those are skipped unless you untick that. A recording
  processed before re-runs with its own saved `review/session.json` (settings and hand edits);
  any other takes the Settings tab, without the hand edits of the recording on screen. Neurosoft
  exports get their windows fitted to the file and keep their previous scenario. Results go next
  to each recording or into one chosen folder. A failure is marked in its row and the batch goes
  on; "Stop after this recording" ends it between files. Double-click a finished row to open it.

## The surfaces

| Tab | Mode | What you do |
|---|---|---|
| **Crop review (SIR)** | Stimulation-induced | Walk the `(config, amplitude)` crops. Crops with **0 detections are red**. Click a peak to force P1; reject false positives; drag out a response and turn it into a template. Recruitment curves sit beside the plot. |
| **Epoch browser (StartStop)** | StartStop | Three views of a condition: **Detection** (one response in context), **Overlay** (all detections of a channel superimposed, as in SIR — that is how you see whether a response is stereotyped or the detector is firing on noise), and **Segment** (the whole condition with bursts shaded and their RMS envelopes drawn on top). Flag, annotate, export. |
| **Spontaneous EMG** | StartStop | RMS envelopes with the detected bursts; export a burst envelope. |
| **Raw** | any | Scroll through the whole recording. Overlay protocol annotations, detections, bursts and envelopes; jump straight to any detection or burst; or fit the entire segment into one window. |

`Settings` is generated from the lever registry, grouped as in `constants.py`, and swaps
its contents with the mode. In stimulation-induced mode its first group is **Detector**: the network
(`unet`, the default) or the templates (`templates`). The template groups (Templates (PASS1),
Gates (PASS2), Markers) act on the template detector only; **Low-confidence curves** holds the
thresholds that decide which curves each detector flags for checking.

### Low-confidence curves

The scenario tab lists, under the channels, every curve the detector is unsure about, with its
reason, and marks them on the response-vs-curve plot with a hollow violet square. Clicking one —
its point or its line in the list — brings it up below and opens the **peak editor**: the curve
at full resolution, the channel's other curves in grey behind it, and its current markers. Each
click places the chosen marker and moves on (onset → P1 → P2); P1 and P2 snap to the extremum
within ±0.5 ms on the side of the click (untick to place freely). "Сохранить пики" stores them
for this curve, "Ответа нет" marks it a false response, "Отмена" leaves it as it is. The same
window opens on any curve from "Поставить пики вручную…". Saved peaks show as hollow diamonds
until "Пересчитать по правкам" applies them; a curve carries one correction at a time (false,
missed or hand peaks), and a curve already corrected shows it in the list. "Только выбранный
канал" narrows the list to the channel on screen. The header says which detector produced the
run.

What the reasons mean:

| Reason | Detector | Meaning |
|---|---|---|
| `сети расходятся в пике` | network | the ensemble's nets place P1 with an SD above the threshold (2 ms; only on curves with a response) |
| `две волны` | network | the P1 heatmap has a second peak almost as high as the first (only on curves with a response) |
| `p(ответ) на границе` | network | p(response) close to the model's threshold (off by default) |
| `сети расходятся в ответе` | network | the nets split on whether there is a response at all (off by default) |
| `не прошло: …` | network | the network sees a response but its peaks fail the doctors' criteria (off by default) |
| `не похоже на шаблон` | templates | a detected response that does not look like the template it was found with |
| `похоже на шаблон, но ответ не найден` | templates | a curve with no detection that does look like the template: a response the gates may have lost |


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

### Crop review — channel decisions

| Action | Effect | Global it writes |
|---|---|---|
| Shift+click, or double-click a table row | reject as a false positive | `SIR_SUPPRESS_KEYS` |
| Ctrl+click | whitelist — bypass the consistency gates | `SIR_FORCE_KEYS` |
| Alt+click | exclude the whole channel | `SIR_EXCLUDE_CHANNELS` |

What is edited here is which channels count, not where their response is. Placing P1 by hand
used to live here too — click a peak, pick a polarity from a dropdown, bind it to a window —
and it is gone: a template drawn over the response says the same thing better, since it carries
the whole shape and its markers instead of one latency and an arrow. Draw one in Template mode,
or place onset/P1/P2 directly on the scenario and Condition tabs.

**Rejection is immediate and everywhere at once**: the markers vanish from the plot, the channel
row in the table drops to 0 detections and reads `rejected`, the crop's count in the list falls,
and the recruitment table and the exported metrics agree. No re-run needed for the numbers to
match what you see — and because it is stored as `SIR_SUPPRESS_KEYS`, the next real run
reproduces the same decision.

Edits are written at the most specific `(config, amp, channel)` precedence level, so they are
**additive**: rejecting on one crop cannot alter a crop you did not name.

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
emgflow/
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
