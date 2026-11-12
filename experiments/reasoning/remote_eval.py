"""Evaluation as Modal functions on eval_image, for the label-free search container.

The search container (search_image, DOCETL_LABEL_FREE=1) holds no ground-truth file, so it
cannot score a plan itself. It sends the plan's output to these functions, which run on
eval_image (code plus experiments/reasoning/data, including the ground-truth files) and
return metrics. They mount no volume, so ground truth is never written anywhere the search
container can read. Only the pipeline output goes
in and only metrics (and, for the final evaluation, the plots and summary files it writes)
come back.

Any module whose Modal function runs a search must import this module at top level, so these
functions are registered on the app and hydrated inside the search container.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import modal

from experiments.reasoning.utils import app, eval_image

EVAL_TIMEOUT = 60 * 30


@app.function(image=eval_image, secrets=[modal.Secret.from_dotenv()], timeout=EVAL_TIMEOUT)
def evaluate_remote(dataset: str, mode: str, method_name: str, results_name: str, results_bytes: bytes) -> Dict[str, Any]:
    """Score one pipeline output; same return value as get_evaluate_func(dataset, mode)(...)."""
    from experiments.reasoning.evaluation.utils import get_evaluate_func

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(results_name).name
        path.write_bytes(results_bytes)
        return get_evaluate_func(dataset, mode=mode)(method_name, str(path))


@app.function(image=eval_image, secrets=[modal.Secret.from_dotenv()], timeout=EVAL_TIMEOUT * 2)
def run_dataset_evaluation_remote(
    dataset: str,
    items: List[Dict[str, Any]],
    ground_truth_path: Optional[str],
    method_name: str,
    root_cost: Optional[float],
) -> Tuple[list, Dict[str, bytes]]:
    """run_dataset_evaluation on copies of the outputs. `items` hold rel_path, content and the
    node fields. Returns (eval_results, files written under the output dir)."""
    from experiments.reasoning.evaluation.utils import run_dataset_evaluation

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        out.mkdir()
        inputs = set()
        objs = []
        for item in items:
            path = out / item["rel_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item["content"])
            inputs.add(path.resolve())
            info = item["info"]
            if item["is_node"]:
                objs.append(SimpleNamespace(
                    result_path=str(path), cost=info["cost"], visits=info["visits"], value=info["value"],
                    mcts_accuracy=info.get("mcts_accuracy"), on_frontier=info.get("on_frontier", False),
                    get_id=(lambda node_id=info["node_id"]: node_id),
                ))
            else:
                objs.append({**info, "file_path": str(path)})
        eval_results = run_dataset_evaluation(
            dataset=dataset, nodes_or_files=objs, output_path=out,
            ground_truth_path=ground_truth_path, method_name=method_name, root_cost=root_cost,
        )
        written = {
            str(p.relative_to(out)): p.read_bytes()
            for p in out.rglob("*") if p.is_file() and p.resolve() not in inputs
        }
        return eval_results, written
