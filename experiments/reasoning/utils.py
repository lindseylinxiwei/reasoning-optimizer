import modal
from typing import Dict, Any, List, Optional

APP_NAME = "docetl-experiments"
VOLUME_NAME = "docetl-ro-experiments"
VOLUME_MOUNT_PATH = "/mnt/docetl-ro-experiments"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True) 

# Base layers shared by both images
_base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_sync()
    .uv_pip_install("matplotlib", "Levenshtein", "nltk")
)

# Full image: the whole experiments package, including ground-truth files. Used by evaluation
# (experiments/reasoning/remote_eval.py) and by runners that do not execute agent-written
# candidate pipelines.
image = (
    _base_image
    .add_local_python_source("experiments", ignore=lambda p: (
        # Include othersystems only if it contains lotus_evaluation or pz_evaluation files
        ("othersystems" in str(p) and 
         "lotus_evaluation" not in str(p) and 
         "pz_evaluation" not in str(p)) or
        # Exclude large files by size (>300MB)
        (p.is_file() and p.stat().st_size > 300_000_000) or
        # Exclude specific file types
        p.suffix in {'.bin', '.sqlite3'} or
        # Exclude specific directories
        any(part in {'.venv', 'outputs', '__pycache__'} for part in p.parts) or
        # Exclude files starting with dot
        p.name.startswith('.') or
        # Exclude chroma directories
        '.chroma-' in str(p)
    ))
    .add_local_python_source("docetl", ignore=["**/.venv/*"])
)


def _ignore_for_eval_experiments(p):
    """Evaluation image: Python files plus experiments/reasoning/data (inputs and ground truth)."""
    if any(part.startswith(".") or part in {"__pycache__", ".venv", "outputs"} for part in p.parts):
        return True
    if p.suffix == ".py":
        return False
    return tuple(p.parts[:2]) != ("reasoning", "data")


# Evaluation image: holds the ground-truth files. Used only by experiments/reasoning/remote_eval.py,
# which mounts no volume, so nothing it reads is written where the search container can see it.
eval_image = (
    _base_image
    .add_local_python_source("experiments", ignore=_ignore_for_eval_experiments)
    .add_local_python_source("docetl", ignore=["**/.venv/*"])
)


def _build_search_image():
    """Label-free image for containers that execute candidate pipelines (the MCTS search).

    It holds no ground-truth file (see experiments/reasoning/label_isolation.py for the rule),
    so agent-written code operators have nothing to read. DOCETL_LABEL_FREE=1 makes evaluation
    call evaluate_remote / run_dataset_evaluation_remote on the full image instead."""
    from experiments.reasoning.label_isolation import (
        LABEL_FREE_ENV, ignore_for_search_docetl, ignore_for_search_experiments, label_free_train_copies,
    )

    img = (
        _base_image
        .env({LABEL_FREE_ENV: "1"})
        .add_local_python_source("experiments", ignore=ignore_for_search_experiments)
        .add_local_python_source("docetl", ignore=ignore_for_search_docetl)
    )
    if modal.is_local():
        import tempfile
        from pathlib import Path

        build_dir = Path(tempfile.gettempdir()) / "docetl_label_free_train"
        for local, remote in label_free_train_copies(build_dir):
            img = img.add_local_file(local, remote)
    return img


search_image = _build_search_image()


def create_original_query_result(
    success: bool,
    cost: float,
    output_file_path: Optional[str] = None,
    sample_output: Optional[List[Dict[str, Any]]] = None,
    accuracy: Optional[float] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """Create an original query result dictionary that's serializable for Modal."""
    return {
        "success": success,
        "cost": cost,
        "output_file_path": output_file_path,
        "sample_output": sample_output or [],
        "accuracy": accuracy,
        "error": error
    }

