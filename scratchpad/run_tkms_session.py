"""Runner for the ТКМС Т12-Л1 session — applies the clinician's per-channel SIR
marker corrections to the two single-shock text-curves files, and lets the third
(paired-pulse) file auto-route to the Condition test.

Each single-shock file collapses to ONE crop (config = file stem, amp = "all"),
so corrections are keyed by (config, channel). Force dicts set P1 polarity and a
(min_lat, max_lat) search window in seconds; onset (foot) and P2 (opposite
rebound) follow automatically. Suppress keys drop a channel's detections.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import src.pipeline as P
from src.pipeline import run_pipeline

DOWNLOADS = Path("/Users/dkleeva/Downloads")
RESULTS = Path(
    "/Users/dkleeva/Library/CloudStorage/GoogleDrive-dkleeva@gmail.com/My Drive/"
    "💻 Professional/🚶🏻‍♂️Verticalization/Coding/results/ТКМС Т12-Л1 session"
)

# config labels as the pipeline parses them from the crop filename
CFG1 = "Т11.12 ендрассик 80 и 90мА"
CFG2 = "Т12.Л1 Ендр"


def reset_corrections():
    P.SIR_FORCE_POS_P1_AFTER = {}
    P.SIR_FORCE_NEG_P1_AFTER = {}
    P.SIR_SUPPRESS_KEYS = set()
    P.SIR_FORCE_KEYS = set()
    P.SIR_FORCE_MARKER_LAT = {}
    P.SIR_EXCLUDE_CHANNELS = set()
    P.SIR_FORCE_P2_WIN_MS = 999.0


def run_file1():
    """Т11-12 ендрассик 80 и 90мА."""
    reset_corrections()
    # red P1 down (works on the mean), onset near 0.02, green up after
    P.SIR_FORCE_NEG_P1_AFTER = {
        (CFG1, "ch2"): (0.019, 0.024),
        (CFG1, "ch8"): (0.015, 0.021),   # red dip after 0.015, onset before it
    }
    # ch3 left as-is (correct); drop ch4/ch5/ch7
    P.SIR_SUPPRESS_KEYS = {
        (CFG1, "all", "ch4"),
        (CFG1, "all", "ch5"),
        (CFG1, "all", "ch7"),
    }
    P.SIR_FORCE_KEYS = {(CFG1, "all", "ch8")}   # red dip on the artifact-recovery ramp
    # ch1 (red up + green down) and ch6 (red down + green up) cancel in the mean,
    # so place them explicitly by latency+polarity and sample each sweep there.
    P.SIR_FORCE_MARKER_LAT = {
        (CFG1, "ch1"): {"onset": 0.017, "p1": (0.0194, +1), "p2": (0.0223, -1)},
        (CFG1, "ch6"): {"onset": 0.020, "p1": (0.0242, -1), "p2": (0.0298, +1)},
    }
    P.SIR_FORCE_P2_WIN_MS = 12.0
    run_pipeline(DOWNLOADS / "Т11-12 ендрассик 80 и 90мА.txt",
                 output_dir=RESULTS / "Т11-12 ендрассик 80 и 90мА")


def run_file2():
    """Т12-Л1 Ендр."""
    reset_corrections()
    P.SIR_FORCE_POS_P1_AFTER = {
        (CFG2, "ch5"): (0.018, 0.022),   # red up ~0.02, green down earlier
    }
    P.SIR_FORCE_NEG_P1_AFTER = {
        (CFG2, "ch2"): (0.019, 0.024),   # red down, green up after
        (CFG2, "ch4"): (0.015, 0.021),   # onset after 0.015, red dip
        (CFG2, "ch8"): (0.013, 0.019),   # red dip before 0.02, onset before, green up after
    }
    # ch3/ch6/ch7 left as-is
    P.SIR_FORCE_KEYS = {
        (CFG2, "all", "ch4"),   # red dip on the artifact-recovery ramp
        (CFG2, "all", "ch8"),   # red dip on the artifact-recovery ramp
    }
    # ch1: red up + green down; the green rebound cancels in the mean, so place
    # both explicitly and sample each sweep (per-sweep peaks are clean here).
    P.SIR_FORCE_MARKER_LAT = {
        (CFG2, "ch1"): {"onset": 0.017, "p1": (0.0198, +1), "p2": (0.0233, -1)},
    }
    P.SIR_FORCE_P2_WIN_MS = 12.0
    run_pipeline(DOWNLOADS / "Т12-Л1 Ендр.txt",
                 output_dir=RESULTS / "Т12-Л1 Ендр")


def run_condition():
    """ткмс т12-л1 80мА — auto-routes to Condition test (no corrections)."""
    reset_corrections()
    run_pipeline(DOWNLOADS / "ткмс т12-л1 80мА.txt",
                 output_dir=RESULTS / "ткмс т12-л1 80мА")


if __name__ == "__main__":
    import time
    for name, fn in [("condition", run_condition), ("file1", run_file1), ("file2", run_file2)]:
        t = time.time()
        fn()
        print(f"=== {name} done {time.time() - t:.0f} s", flush=True)
    print("ALL DONE", flush=True)
