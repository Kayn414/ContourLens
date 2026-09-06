import importlib.metadata

try:
    __version__ = importlib.metadata.version("source")
except importlib.metadata.PackageNotFoundError:
    pass
