"""
OGD Search Prototype Application

Thin wrapper that launches the thesis-aligned portal analysis prototype.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    if __package__ in (None, ""):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from portal_analysis_app import main as portal_analysis_main
    else:
        from .portal_analysis_app import main as portal_analysis_main

    portal_analysis_main()


if __name__ == "__main__":
    main()
