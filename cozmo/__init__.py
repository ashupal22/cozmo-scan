"""cozmo-scan: measured floor plans, damage and repair scope from iPhone captures."""

__version__ = "0.1.0"

import os as _os

# Depth Anything 3's ray-pose solver uses linear algebra Apple's GPU lacks. The fallback must be set before torch starts,
# and any entry point imports this package first.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
