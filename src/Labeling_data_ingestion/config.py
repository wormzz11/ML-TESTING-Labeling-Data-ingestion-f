from pathlib import Path
from dataclasses import dataclass
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "ground_truth_fallback" / "combined_truth.csv"
TO_FILTER_PATH = ROOT / "data" / "certain_auto" / "auto_labeled_all.csv"

@dataclass
class ThresholdConfig:
    high_neg: float = 0.2
    high_pos: float = 0.26
    