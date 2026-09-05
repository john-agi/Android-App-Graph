"""Shim: AITK loads translators only from ``aitk/translators/<name>.py``."""

from android_app_graph.adapters.aitk_translator import UIKobeV2Translator, register

__all__ = ["UIKobeV2Translator", "register"]
