from __future__ import annotations

import sys
from pathlib import Path


LOCOMO_DIR = Path(__file__).resolve().parents[1]
if str(LOCOMO_DIR) not in sys.path:
    sys.path.insert(0, str(LOCOMO_DIR))

