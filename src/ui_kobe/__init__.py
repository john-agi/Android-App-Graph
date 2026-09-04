from importlib.metadata import version

from ui_kobe.kobe import Kobe

__version__ = version("ui-kobe")

__all__ = ["Kobe", "__version__"]
