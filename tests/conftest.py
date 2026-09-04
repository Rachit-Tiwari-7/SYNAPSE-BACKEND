import sys
import os

# Ensure repo root is on sys.path so 'backend' shim and 'app' resolve cleanly in pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import backend
except ImportError:
    pass
