"""Shared constants and small helpers for the EMG pipeline."""

STARTSTOP_MODE = False
ARTCHAN = None
THRESH = 4

RAW_BANDPASS_L_FREQ = 20
RAW_BANDPASS_H_FREQ = 180

ARTIFACT_REREF = True
CAR_REREF = True
LATERAL_CAR_REREF = False

MAT_DIVIDE_BY_1000 = False


STARTSTOP_RMS_WIN_MS = 4.0
STARTSTOP_RMS_K = 4.0
STARTSTOP_MIN_DIST_MS = 50.0
STARTSTOP_PLOT_WIN_S = 0.1
STARTSTOP_EPOCH_TMIN = -0.1
STARTSTOP_EPOCH_TMAX = 0.1
STARTSTOP_TM_SCORE_THR = 0.75
STARTSTOP_TM_TEMPLATE_CORR_THR = 0.75
STARTSTOP_TM_SCALES = (0.95, 1.0, 1.05, 1.1, 1.2)
STARTSTOP_TM_MATCH_PEAK_MIN_DIST_MS = 50.0
STARTSTOP_TM_MATCH_PEAK_PROMINENCE = 0.02  # min prominence of score peak (0 = disabled)
STARTSTOP_TM_MAX_MATCHES_PER_CHANNEL = 0
STARTSTOP_TM_TOP_K_CANDIDATES = 0
STARTSTOP_LEAKAGE_CORR_REJECTION = True
STARTSTOP_CHANNEL_MIN_VALID_FRAC = 0.7   # min fraction of epochs with valid p1 for channel to be kept
STARTSTOP_CHANNEL_MIN_MEDIAN_CORR = 0.7  # min median epoch-template corr; with low valid_frac → wipe channel
STARTSTOP_MIN_DETECTIONS_PER_CHANNEL = 5  # if a channel has fewer detections in a condition, discard them (0 = disabled)
STARTSTOP_TM_NO_POSTHOC_CHECKS = True    # Skip PTP/amplitude wipes; leakage and channel stability still apply
STARTSTOP_TM_TEMPLATE_SFREQ = 2000.0
STARTSTOP_TM_TEMPLATE_CENTER_SAMPLE = 200
STARTSTOP_TM_TEMPLATE_TMIN = -0.1
STARTSTOP_TM_TEMPLATE_TMAX = 0.1
STARTSTOP_BASELINE_TMIN = -0.05
STARTSTOP_BASELINE_TMAX = -0.02
STARTSTOP_ONSET_TMIN = -0.02
STARTSTOP_ONSET_TMAX = 0.01
STARTSTOP_RESP_TMAX = 0.03
STARTSTOP_SIM_TMIN = -0.02
STARTSTOP_SIM_TMAX = 0.02

LOWPASS_CUTOFF_HZ = 180.0

EPOCH_TMIN = -0.05
EPOCH_TMAX = 0.1

BASELINE_TMIN = -0.04
BASELINE_TMAX = 0.0

RESP_TMIN = 0.01
RESP_TMAX = 0.04

MIN_VALID_EPOCHS = 5 #very conservative


def get_prominence_k(file_name: str, ch_name: str) -> int:
    # if file_name in ["14+15-_8.fif", "14+15-_9.fif"]:
    #     return 30 if ch_name == "Flex U R" else 10
    # if file_name in ["8+9-_2.fif"]:
    #     return 30 if ch_name == "Ext U R" else 10
    # if file_name in ["1+2-_8.fif"]:
    #     return 40 if ch_name == "Ext U L" else 10
    return 10
