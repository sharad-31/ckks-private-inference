"""
config.py - single place for every setting of the MIMIC tabular pipeline.

Two things are normally changed (or set through environment variables):
    MIMIC_DATASET = "mimic4" or "mimic3"
    N_VITALS      = 10 -> grid 12x10   |   5 -> grid 12x5
Everything else (item ids, file and column names, kernel size, windows, CKKS slots,
output file names) adjusts automatically.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path for common imports
ROOT_REPO = Path(__file__).resolve().parent.parent
if str(ROOT_REPO) not in sys.path:
    sys.path.insert(0, str(ROOT_REPO))

from common.ckks_context import PRESETS

DATASET  = os.environ.get("MIMIC_DATASET", "mimic4")   # <<<<< "mimic3" or "mimic4"
N_VITALS = int(os.environ.get("N_VITALS", 10))          # <<<<< 10 -> 12x10, 5 -> 12x5

assert DATASET in ("mimic3", "mimic4"), "DATASET must be mimic3 or mimic4"

# ---------------- time axis (48h in 4h bins = 12 rows) ----------------
WINDOW_H  = 48
BIN_HOURS = 4
N_BINS    = WINDOW_H // BIN_HOURS            # 12

# ---------------- vitals (column order = priority) ----------------
# itemids per dataset.  MIMIC-III has CareVue + MetaVision ids, MIMIC-IV only MetaVision.
VITAL_IDS = {
    "heart_rate": {"mimic4": [220045],         "mimic3": [211, 220045]},
    "resp_rate":  {"mimic4": [220210],         "mimic3": [618, 615, 220210, 224690]},
    "spo2":       {"mimic4": [220277],         "mimic3": [646, 220277]},
    "sbp":        {"mimic4": [220179, 220050], "mimic3": [51, 455, 220179, 220050]},
    "temp_c":     {"mimic4": [223762, 223761], "mimic3": [676, 678, 223762, 223761]},
    "dbp":        {"mimic4": [220180, 220051], "mimic3": [8368, 8441, 220180, 220051]},
    "mbp":        {"mimic4": [220181, 220052], "mimic3": [52, 456, 220181, 220052]},
    "gcs_motor":  {"mimic4": [223901],         "mimic3": [454, 223901]},
    "fio2":       {"mimic4": [223835],         "mimic3": [190, 3420, 223835]},
    "glucose":    {"mimic4": [225664, 220621], "mimic3": [807, 811, 1529, 3744, 3745, 225664, 220621]},
}
VITALS_ALL    = {v: ids[DATASET] for v, ids in VITAL_IDS.items()}
TEMP_F_ITEMS  = {223761, 678}                # Fahrenheit itemids -> converted to C
RANGES = {   # outside range -> treated as missing
    "heart_rate": (20, 250), "resp_rate": (4, 70), "spo2": (50, 100),
    "sbp": (40, 300), "temp_c": (30, 43), "dbp": (20, 200), "mbp": (30, 250),
    "gcs_motor": (1, 6), "fio2": (21, 100), "glucose": (20, 1000),
}
assert 1 <= N_VITALS <= len(VITALS_ALL), f"N_VITALS must be 1..{len(VITALS_ALL)}"
VITALS = dict(list(VITALS_ALL.items())[:N_VITALS])     # vitals used in the grid
STATIC_VITALS = ["heart_rate", "resp_rate", "spo2", "sbp", "temp_c"]   # for built-in static features

# ---------------- CNN ----------------
GRID      = (N_BINS, N_VITALS)               # (12, 10) or (12, 5)
KERNEL_H  = 3
KERNEL_W  = N_VITALS                         # full width kernel
N_WINDOWS = N_BINS - KERNEL_H + 1            # 10 sliding windows
CHANNELS  = 4
TARGET    = "los7"                           # "los7" or "mortality"
SEED      = 42
EPOCHS, BATCH, LR, WD, PATIENCE = 60, 256, 1e-3, 1e-3, 10

# ---------------- paths ----------------
ROOT   = Path(__file__).resolve().parent
RAW    = ROOT / "data" / "raw" / DATASET     # put the MIMIC csv / csv.gz files here
PROC   = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORT = ROOT / "reports"
# Static features (one row per stay_id). Step 10 builds a built-in set automatically.
STATIC_CSV  = Path(os.environ.get("STATIC_CSV", PROC / f"static_builtin_{DATASET}.csv"))
STATIC_DROP = ["subject_id", "hadm_id", "split", "los7", "mortality", "y"]

TAG = f"{DATASET}_{N_BINS}x{N_VITALS}_{TARGET}"        # every output file carries this

for p in (RAW, PROC, MODELS, REPORT):
    p.mkdir(parents=True, exist_ok=True)


def find_table(name):
    """case-insensitive lookup: name.csv / name.csv.gz / NAME.csv / NAME.csv.gz"""
    for stem in (name.lower(), name.upper()):
        for ext in (".csv", ".csv.gz"):
            p = RAW / f"{stem}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"{name}.csv(.gz) not found in {RAW}")
