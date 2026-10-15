"""Make `src/` importable when the scripts are run directly (no installation needed)."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

PRIVATE = ROOT / "data" / "private"     # the school's data and the tables derived from it: never committed
