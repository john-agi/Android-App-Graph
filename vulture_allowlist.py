"""Vulture allowlist for the 90% blocking run.

Every entry names code that Vulture cannot see a use for inside src/ and
tests/ but that is used elsewhere. Each entry carries a one-line reason.
An entry marks that name as used everywhere in src/ and tests/, so prefer
deleting or renaming over allowlisting a generic name. Remove an entry as
soon as its reason stops being true.
"""

from typing import Any

_: Any = None  # placeholder for the ``_.name`` form written by --make-whitelist

# _.example_method  # called by keyword from AITK through aitk_files/android_app_graph_v2.py
