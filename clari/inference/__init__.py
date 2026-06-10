__all__ = ["ClariSampler", "save", "rank", "export_cifs"]

_LAZY = {
    "ClariSampler": "clari.inference.sample",
    "save": "clari.inference.sample",
    "rank": "clari.inference.rank",
    "export_cifs": "clari.inference.export",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        # Overwrite the submodule binding the import system creates, so e.g.
        # `clari.inference.rank` resolves to the function, not the rank module.
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
