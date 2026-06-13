import os
import pathlib
import uuid

PACKAGE_ROOT = pathlib.Path(__file__).parent

ASSETS_DIR = PACKAGE_ROOT / "assets"
DATA_DIR = pathlib.Path(os.environ.get("CLARI_DATA_DIR", pathlib.Path.cwd() / "data"))
RESULTS_DIR = pathlib.Path(os.environ.get("CLARI_RESULTS_DIR", pathlib.Path.cwd() / "results"))
LOG_DIR = pathlib.Path(os.environ.get("CLARI_LOG_DIR", pathlib.Path.cwd() / "logs"))


def resolve_results_path(path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.is_absolute() or path.exists() or len(path.parts) != 1:
        return path
    return RESULTS_DIR / path


def random_checkpoint_dir():
    rand_dir = DATA_DIR / "checkpoints" / str(uuid.uuid4())
    assert not rand_dir.exists()
    return str(rand_dir)
