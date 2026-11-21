"""
config.py - one place for every setting (image version: NIH Malaria cell images, parasitized vs uninfected).

Normally you change nothing.  Optional (env variables or edit here):
    IMG     image side after resize (default 28)      KERNEL (7)   STRIDE (3)   CHANNELS (4)
With IMG=28, KERNEL=7, STRIDE=3 the CNN has 8 x 8 = 64 sliding windows of 49 pixels:
64 windows x 64 (49 padded to a power of two) = 4096 slots = exactly one standard ciphertext.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path for common imports
ROOT_REPO = Path(__file__).resolve().parent.parent
if str(ROOT_REPO) not in sys.path:
    sys.path.insert(0, str(ROOT_REPO))

from common.ckks_context import PRESETS
from common.utils import next_pow2

DATASET  = os.environ.get("IMG_DATASET", "malaria")
IMG      = int(os.environ.get("IMG", 28))
KERNEL   = int(os.environ.get("KERNEL", 7))
STRIDE   = int(os.environ.get("STRIDE", 3))
CHANNELS = int(os.environ.get("CHANNELS", 4))
assert (IMG - KERNEL) % STRIDE == 0, "(IMG - KERNEL) must be divisible by STRIDE"
SIDE      = (IMG - KERNEL) // STRIDE + 1          # windows per side
N_WINDOWS = SIDE * SIDE

CLASSES = {"Uninfected": 0, "Parasitized": 1}      # folder name -> label (1 = positive class)
SEED = 42
EPOCHS, BATCH, LR, WD, PATIENCE = 80, 128, 1e-3, 1e-3, 10

ROOT   = Path(__file__).resolve().parent
RAW    = ROOT / "data" / "raw" / DATASET           # put the 'Parasitized' and 'Uninfected' folders here
PROC   = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORT = ROOT / "reports"
TAG    = f"{DATASET}_{IMG}" + ("" if (CHANNELS, KERNEL, STRIDE) == (4, 7, 3) else f"_c{CHANNELS}k{KERNEL}s{STRIDE}")

for p in (RAW, PROC, MODELS, REPORT):
    p.mkdir(parents=True, exist_ok=True)
