#!/usr/bin/env python3
"""
Sign-Ease Flask entrypoint. Implementation: api.py (10-frame consensus,
post-softmax 1.2x HELP/YES boost, hard neutral below 5%, soft hold 5–15%).
Run from this directory:  python app.py
Default port 8001 (macOS often uses 5000 for AirPlay; set PORT=5000 if you free that port).
"""

from __future__ import annotations

import os

from api import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8001")), debug=True)
