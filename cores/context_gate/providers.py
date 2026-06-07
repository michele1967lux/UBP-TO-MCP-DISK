"""providers.py — stub for adapter-embedded providers.

Phase 10.D (W6-B01-03): the 3-file module pattern expects
`providers.py` alongside `adapter.py`. This module embeds its
provider logic directly inside `adapter.py` (or in a sibling helper
such as `operations.py`). This file exists to satisfy structural
audits and to give an explicit hook point if providers are later
extracted.
"""
from __future__ import annotations

# Intentionally empty. See adapter.py for the operation handlers.
