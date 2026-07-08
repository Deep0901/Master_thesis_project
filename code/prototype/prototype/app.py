"""
OGD Search Prototype Application

Thin wrapper that launches the thesis phase-2 retrieval prototype.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    if __package__ in (None, ""):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from swiss_ogd_portal import main as prototype_main
    else:
        from .swiss_ogd_portal import main as prototype_main

    prototype_main()


if __name__ == "__main__":
    main()
