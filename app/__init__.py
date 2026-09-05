"""
SynapseOS — Central Healthcare Backend & Multi-Agent Swarm.
"""
import sys
import types

__version__ = "1.0.0"

# Support both `app...` and `backend.app...` imports seamlessly in standalone backend deployments
if "backend" not in sys.modules:
    _backend_mod = types.ModuleType("backend")
    _backend_mod.app = sys.modules[__name__]
    sys.modules["backend"] = _backend_mod
    sys.modules["backend.app"] = sys.modules[__name__]
