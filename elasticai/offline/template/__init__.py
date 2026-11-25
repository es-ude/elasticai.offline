import os
import pkgutil
import re
from importlib import import_module

package_paths = globals().get("__path__", [os.path.dirname(__file__)])
module_pattern = re.compile(r"^pipeline_v\d+$")


_lazy = {}
for _finder, _name, _ispkg in pkgutil.iter_modules(package_paths):
    if module_pattern.match(_name):
        _class_name = "".join(part.capitalize() for part in _name.split("_"))
        _lazy[_class_name] = _name

__all__ = sorted(_lazy)


def __getattr__(attr):
    if attr in _lazy:
        module = import_module(f".{_lazy[attr]}", package=__name__)
        try:
            cls = getattr(module, attr)
        except AttributeError:
            raise AttributeError(f"Modul {_lazy[attr]!r} enthält keine Klasse {attr!r}") from None
        globals()[attr] = cls  # cachen, damit __getattr__ nur einmal läuft
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


def __dir__():
    return sorted(set(globals()) | set(_lazy))
