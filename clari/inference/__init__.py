from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clari.inference.export import export_cifs
    from clari.inference.rank import rank
    from clari.inference.sample import ClariSampler, sample, save

__all__ = ["ClariSampler", "sample", "save", "rank", "export_cifs"]

_LAZY = {
    "export_cifs": "clari.inference.export",
    "rank": "clari.inference.rank",
    "ClariSampler": "clari.inference.sample",
    "sample": "clari.inference.sample",
    "save": "clari.inference.sample",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
