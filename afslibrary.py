import os
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # carpeta donde vive este fichero


def list_py_files(folder_path: Path):
    """Return a list of .py filenames (without extensions) in the given folder, excluding __init__.py."""
    if not folder_path.is_dir():
        return []
    with os.scandir(folder_path) as entries:
        return [
            entry.name[:-3]
            for entry in entries
            if entry.is_file()
            and entry.name.endswith(".py")
            and entry.name != "__init__.py"
        ]


def import_module(module_files, module_name):
    """
    Dynamically imports modules trying relative (package) first, then absolute.
    """
    for file in module_files:
        rel = f".{module_name}.{file}"
        absn = f"{module_name}.{file}"
        try:
            module = importlib.import_module(rel, package=__package__ or "pypricing")
        except (ImportError, ModuleNotFoundError, ValueError):
            module = importlib.import_module(absn)
        globals().update(
            {k: getattr(module, k) for k in dir(module) if not k.startswith("_")}
        )


# data
folder = ROOT / "data"
module_names = list_py_files(folder)
import_module(module_names, "data")

# pricing
folder = ROOT / "pricing"
module_names = list_py_files(folder)
import_module(module_names, "pricing")

# risk
folder = ROOT / "risk"
module_names = list_py_files(folder)
import_module(module_names, "risk")
