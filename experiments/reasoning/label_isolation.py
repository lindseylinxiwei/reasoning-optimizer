"""Keep ground-truth files out of the container that runs candidate pipelines.

Candidate pipelines can contain agent-written code operators, which DocETL runs with plain
`exec` and full builtins. Stripping label fields from what the agent sees (label_guard) does
not stop such code from opening a ground-truth file on disk, so the Modal image that runs the
search must not contain any. This module decides, file by file, what goes into that image.

The rule fails closed. From `experiments/` the search image gets only:
  - Python files,
  - pipeline YAMLs (files under a `pipelines*` directory),
  - input splits under `reasoning/data/train/` whose records carry no label field.
Every other file (other splits, ground-truth files, experiment outputs and runs, plots, notes)
is left out. From `docetl/` everything is kept except data files (.json/.jsonl/.csv/...) that
carry label fields or cannot be read.

Evaluation, which needs the labels, runs in a separate Modal function on the full image
(experiments/reasoning/remote_eval.py). The search container sets DOCETL_LABEL_FREE=1 and
evaluation code checks it to call that function instead of reading ground truth locally.
"""
import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_label_guard():
    """docetl/mcts/label_guard.py, loaded by path: importing docetl.mcts would import mcts.py,
    which imports experiments.reasoning.utils, which imports this module."""
    import importlib.util
    import sys

    name = "docetl.mcts.label_guard"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location("_label_guard_standalone", REPO_ROOT / "docetl" / "mcts" / "label_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_label_guard = _load_label_guard()
LABEL_FIELDS = _label_guard.LABEL_FIELDS
_SUSPICIOUS = _label_guard._SUSPICIOUS
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DOCETL_DIR = REPO_ROOT / "docetl"

LABEL_FREE_ENV = "DOCETL_LABEL_FREE"

DATA_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".parquet", ".pkl", ".pickle",
                 ".xlsx", ".xls", ".feather", ".arrow", ".sqlite", ".sqlite3", ".db", ".npz", ".npy"}
PARSEABLE_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv"}
MAX_SCAN_BYTES = 300_000_000          # larger files are treated as labeled (not scanned)

# Every label field of every workload, plus "GT ..." style keys.
ALL_LABEL_KEYS: Set[str] = set().union(*LABEL_FIELDS.values())
_GT_KEY = re.compile(r"^gt\b|ground[_ ]?truth", re.IGNORECASE)
# File names that mark a ground-truth file whatever its contents (e.g. CUAD's clause table).
_GT_NAME = re.compile(r"(^|[_\-. ])gt([_\-. ]|$)|ground[_\-]?truth|master_clauses|(^|[_\-. ])labels?([_\-. ]|$)",
                      re.IGNORECASE)

TRAIN_DIR = ("reasoning", "data", "train")
SKIP_DIR_NAMES = {"outputs", "__pycache__", ".venv", "node_modules"}


def label_free_container() -> bool:
    """True inside the search container, where ground-truth files are absent."""
    return os.environ.get(LABEL_FREE_ENV) == "1"


def is_label_key(key: str) -> bool:
    return key in ALL_LABEL_KEYS or bool(_SUSPICIOUS.search(key)) or bool(_GT_KEY.search(key))


def _keys(obj, depth: int = 0, out: Optional[Set[str]] = None) -> Set[str]:
    out = set() if out is None else out
    if depth > 4:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            _keys(v, depth + 1, out)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, depth + 1, out)
    return out


def label_keys_in(path: Path) -> Optional[Set[str]]:
    """Label keys found in a data file, or None if the file could not be read (treat as labeled)."""
    try:
        if path.stat().st_size > MAX_SCAN_BYTES:
            return None
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path) as f:
                keys = _keys(json.load(f))
        elif suffix in (".jsonl", ".ndjson"):
            keys = set()
            with open(path) as f:
                for line in f:
                    if line.strip():
                        _keys(json.loads(line), out=keys)
        elif suffix in (".csv", ".tsv"):
            with open(path, newline="", errors="replace") as f:
                keys = set(next(csv.reader(f, delimiter="\t" if suffix == ".tsv" else ","), []))
        else:
            return None
    except Exception:
        return None
    return {k for k in keys if is_label_key(k)}


def file_has_labels(path: Path) -> bool:
    """Fail closed: a data file has labels if its name marks it as ground truth, it holds a
    label key, or it cannot be read and checked."""
    if _GT_NAME.search(path.name):
        return True
    found = label_keys_in(path)
    return found is None or bool(found)


def _parts(rel: Path) -> tuple:
    return tuple(Path(rel).parts)


def ignore_for_search_experiments(rel: Path) -> bool:
    """Ignore callable for add_local_python_source("experiments"). `rel` is relative to experiments/."""
    parts = _parts(rel)
    if any(p.startswith(".") or p in SKIP_DIR_NAMES for p in parts):
        return True
    suffix = Path(rel).suffix.lower()
    if suffix == ".py":
        return False
    if suffix in (".yaml", ".yml"):
        return not any(p.startswith("pipelines") for p in parts[:-1])
    if suffix in DATA_SUFFIXES and parts[:-1] == TRAIN_DIR:
        return file_has_labels(EXPERIMENTS_DIR / rel)
    return True


def ignore_for_search_docetl(rel: Path) -> bool:
    """Ignore callable for add_local_python_source("docetl"). `rel` is relative to docetl/."""
    parts = _parts(rel)
    if any(p in SKIP_DIR_NAMES for p in parts) or Path(rel).name == ".DS_Store":
        return True
    if Path(rel).suffix.lower() in DATA_SUFFIXES:
        return file_has_labels(DOCETL_DIR / rel)
    return False


def label_free_train_copies(build_dir: Path) -> List[tuple]:
    """Label-free copies of the train splits that carry labels inline (BioDEX, MEDEC), so a
    search on those workloads still finds its input in the search image. Returns
    (local copy, path inside the image) pairs. A split whose labels are not all known label
    fields of its workload gets no copy: it stays out of the image."""
    strip_labels = _label_guard.strip_labels

    pairs = []
    train_dir = EXPERIMENTS_DIR.joinpath(*TRAIN_DIR)
    build_dir = Path(build_dir)
    for src in sorted(train_dir.glob("*.json")):
        if not file_has_labels(src) or src.stem not in LABEL_FIELDS:
            continue
        with open(src) as f:
            stripped = strip_labels(json.load(f), src.stem)
        if {k for k in _keys(stripped) if is_label_key(k)}:
            continue
        out = build_dir / src.name
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(stripped)
        if not out.exists() or out.read_text() != text:
            out.write_text(text)
        pairs.append((out, "/root/experiments/" + "/".join(TRAIN_DIR) + "/" + src.name))
    return pairs


def _walk(top: Path, skip_dirs: Iterable[str] = ()) -> Iterable[Path]:
    skip = set(skip_dirs)
    for root, dirs, files in os.walk(top):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            yield Path(root) / name


def scan_labeled_files(top: Path, skip_dirs: Iterable[str] = ("__pycache__", ".venv", "node_modules"),
                       suffixes: Iterable[str] = PARSEABLE_SUFFIXES | DATA_SUFFIXES) -> List[Path]:
    """Every data file under `top` that has label fields, a ground-truth name, or cannot be checked."""
    suffixes = set(suffixes)
    return sorted(p for p in _walk(top, skip_dirs) if p.suffix.lower() in suffixes and file_has_labels(p))


def search_image_files(top: Path, ignore) -> List[Path]:
    """Files under `top` that the search image would contain under the given ignore callable."""
    return sorted(p for p in _walk(top, ("__pycache__",)) if not ignore(p.relative_to(top)))
