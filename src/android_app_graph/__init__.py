from importlib.metadata import version

from android_app_graph.kobe import Kobe

__version__ = version("android-app-graph")

__all__ = ["Kobe", "__version__"]
