"""Shared constants and small helpers for the EMG pipeline."""

# Default mode when caller does not pass startstop_mode.
STARTSTOP_MODE = False
# Artifact channel selection: None -> auto-detect by channel names.
ARTCHAN = None
# Multiplier for artifact-channel std to detect stimulus events.
THRESH = 4
# Width range (in ms) for artifact peak detection. None to use THRESH-based detection.
ART_PEAK_WIDTH_MS = None
# Height range (in V) for artifact peak detection. None to use THRESH-based detection.
ART_PEAK_HEIGHT = None

# Main band-pass low cutoff for raw preprocessing (Hz). None to skip filtering.
RAW_BANDPASS_L_FREQ = 60
# Main band-pass high cutoff for raw preprocessing (Hz). None to skip filtering.
RAW_BANDPASS_H_FREQ = 300

# Enable artifact-reference subtraction before other processing.
ARTIFACT_REREF = False
# Enable common average reference across non-artifact channels.
CAR_REREF = False
# Enable side-specific CAR (left/right grouping) if implemented.
LATERAL_CAR_REREF = False

# Convert MAT signal units by dividing by 1000.
MAT_DIVIDE_BY_1000 = True


# Reject StartStop channels that look like artifact leakage.
STARTSTOP_LEAKAGE_CORR_REJECTION = True

# Minimum template-match score to accept a candidate event.
STARTSTOP_TM_SCORE_THR = 0.75
# Minimum correlation to template to keep a match.
STARTSTOP_TM_TEMPLATE_CORR_THR = 0.75
# Time-scale factors tested during template matching.
STARTSTOP_TM_SCALES = (0.95, 1.0, 1.05, 1.1, 1.2)
# Minimum temporal distance between StartStop events (ms).
STARTSTOP_MIN_DIST_MS = 50.0
# StartStop epoch start time relative to event (s).
STARTSTOP_EPOCH_TMIN = -0.1
# StartStop epoch end time relative to event (s).
STARTSTOP_EPOCH_TMAX = 0.1

# Minimum spacing between peaks in template-match score (ms).
STARTSTOP_TM_MATCH_PEAK_MIN_DIST_MS = 50.0
# Minimum prominence of score peaks (0 disables prominence filtering).
STARTSTOP_TM_MATCH_PEAK_PROMINENCE = 0.02  # min prominence of score peak (0 = disabled)
# Max accepted matches per channel (0 means no explicit cap).
STARTSTOP_TM_MAX_MATCHES_PER_CHANNEL = 0

# Keep top-k candidate matches before filtering (0 means keep all).
STARTSTOP_TM_TOP_K_CANDIDATES = 0

# Minimum fraction of valid epochs required to keep a channel.
STARTSTOP_CHANNEL_MIN_VALID_FRAC = 0.7   # min fraction of epochs with valid p1 for channel to be kept
# Minimum median epoch-template correlation required per channel.
STARTSTOP_CHANNEL_MIN_MEDIAN_CORR = 0.7  # min median epoch-template corr; with low valid_frac → wipe channel
# Minimum detections per channel/condition; below this, detections are dropped.
STARTSTOP_MIN_DETECTIONS_PER_CHANNEL = 5  # if a channel has fewer detections in a condition, discard them (0 = disabled)
# Skip amplitude/PTP posthoc wipes in StartStop mode.
STARTSTOP_TM_NO_POSTHOC_CHECKS = True    # Skip PTP/amplitude wipes; leakage and channel stability still apply
# Resampling target for StartStop template construction (Hz).
STARTSTOP_TM_TEMPLATE_SFREQ = 2000.0
# Center sample index for StartStop template alignment.
STARTSTOP_TM_TEMPLATE_CENTER_SAMPLE = 200
# Baseline window start for StartStop amplitude/noise estimation (s).
STARTSTOP_BASELINE_TMIN = -0.05
# Baseline window end for StartStop amplitude/noise estimation (s).
STARTSTOP_BASELINE_TMAX = -0.02
# Allowed onset-search start in StartStop mode (s).
STARTSTOP_ONSET_TMIN = -0.02
# Allowed onset-search end in StartStop mode (s).
STARTSTOP_ONSET_TMAX = 0.01
# Latest expected response time in StartStop mode (s).
STARTSTOP_RESP_TMAX = 0.03
# Similarity-comparison window start (s) for StartStop channel QC.
STARTSTOP_SIM_TMIN = -0.02
# Similarity-comparison window end (s) for StartStop channel QC.
STARTSTOP_SIM_TMAX = 0.02

# Stimulation-induced epoch start relative to detected stimulus event (s).
EPOCH_TMIN = -0.05
# Stimulation-induced epoch end relative to detected stimulus event (s).
EPOCH_TMAX = 0.1

# Baseline start for stimulation-induced per-epoch metrics (s).
BASELINE_TMIN = -0.04
# Baseline end for stimulation-induced per-epoch metrics (s).
BASELINE_TMAX = -0.01

# Earliest response time allowed for peak search (s).
RESP_TMIN = -0.01
# Latest response time allowed for peak search (s).
RESP_TMAX = 0.02

# Enable artifact-correlation channel-level rejection in stimulation mode.
STIM_EPOCH_ARTIFACT_CORR_REJECTION = False
# Absolute correlation threshold vs artifact channel means.
STIM_EPOCH_ARTIFACT_ABS_CORR_THR = 0.5
# Maximum allowed deviation of epoch onset from template onset (s).
STIM_ONSET_MAX_DEV_S = 0.003

# Time-scale factors tested when matching pre-computed templates in SIR mode.
# Keep scales near 1.0 so the synthesized template preserves the shape of the
# original .npy templates (small scales compress them into ringing).
SIR_TM_SCALES = (0.3, 0.5, 0.75, 1.0, 1.25, 1.5)
# Minimum correlation to accept a template match in SIR mode.
SIR_TM_MIN_CORR = 0.7
# When True, fit a separate template per (config, amplitude, channel)
# instead of one template per (config, channel) — useful when the
# response shape changes meaningfully with stim amplitude.
SIR_TEMPLATE_PER_AMPLITUDE = True
# Minimum absolute amplitude for individual detected peaks (uV).
STIM_PEAK_AMP_MIN_UV = 3.0
# Minimum PTP (|P1-P2|) required for a valid response (uV).
STIM_PTP_MIN_UV = 5.0
# Minimum absolute P1 amplitude required (uV).
STIM_P1_ABS_MIN_UV = 3.0

# Minimum number of detected stimulus events needed to build templates.
MIN_VALID_EPOCHS = 5

# Minimum inter-trial median pairwise Pearson correlation in the response
# window required to keep a channel. None disables the gate.
STIM_MIN_INTER_TRIAL_CORR = 0.7

# Minimum median correlation between each epoch and the matched config-level
# template in the response window — channel-level rejection gate.
STIM_CHANNEL_MIN_MEDIAN_CORR = 0.5
# Minimum fraction of epochs with a valid p1 — channel-level rejection gate.
STIM_CHANNEL_MIN_VALID_FRAC = 0.3
