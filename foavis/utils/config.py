import time
import uuid

from dotmap import DotMap


class StrictDotMap(DotMap):
    """Dot access to the config; a missing key raises instead of silently creating it."""

    def __getattr__(self, name):
        if name not in self:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return super().__getattr__(name)


def create_folder_name():
    """Run directory name: <date>_<time>_<random id>."""
    return f"{time.strftime('%Y_%m_%d_%H_%M_%S')}_{uuid.uuid4().hex[:8]}"


def flatten(d, prefix=""):
    """{'a': {'b': 1}} -> {'a.b': 1} (for logging)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out
