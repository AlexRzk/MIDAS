"""
pytest configuration for MIDAS tests.
"""
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "features"))
sys.path.insert(0, str(project_root / "processor"))
