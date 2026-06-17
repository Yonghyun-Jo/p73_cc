"""scripts/ 를 sys.path 에 넣어 robot_gui.* 를 import 가능하게."""
import sys
from pathlib import Path

# tests → robot_gui → scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
