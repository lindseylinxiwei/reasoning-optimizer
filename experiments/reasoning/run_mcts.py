#!/usr/bin/env python3
"""
MCTS Experiment Runner

This script runs the MCTS-based optimization for DocETL pipelines.
It's extracted from the MCTS folder to provide a clean experiment interface.
"""

# Load API keys from the repository-root .env before DocETL is imported (DocETL's own
# load_dotenv() would otherwise pick up a package-level .env first).
from pathlib import Path as _Path
from dotenv import load_dotenv as _load_dotenv
# On Modal the file sits at /root/run_mcts.py and the keys come from the Modal secret instead.
_here = _Path(__file__).resolve()
if len(_here.parents) > 2 and (_here.parents[2] / ".env").exists():
    _load_dotenv(_here.parents[2] / ".env")

import os
import json
import argparse
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from docetl.mcts import MCTS, Node, ParetoFrontier
from docetl.mcts.jev_selector import DEFAULT_JEV_MODEL, DEFAULT_INPUT_PRICE
from docetl.reasoning_optimizer.directives import (
    DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, ALL_DIRECTIVES
)
from experiments.reasoning.evaluation.utils import run_dataset_evaluation, get_evaluate_func, get_dataset_stats
# Modal integration
import modal
import yaml
from experiments.reasoning.utils import app, volume, VOLUME_MOUNT_PATH, search_image
# Registers evaluate_remote / run_dataset_evaluation_remote on the app, so the label-free search
# container can call them (it holds no ground-truth file; see label_isolation.py).
import experiments.reasoning.remote_eval  # noqa: F401

# Rewrites left out of the search unless asked for. "change model cost" is the generic entry;
# the search adds one model-change rewrite per model in change_model_cost.MODEL_COSTS itself.
DEFAULT_EXCLUDED_DIRECTIVES = ("hierarchical_reduce", "chunk_header_summary", "change model cost")

# Modal's maximum function timeout. A search that runs longer stops here; run the same command
# again and it resumes from the checkpoint on the volume.
SEARCH_TIMEOUT_S = 24 * 60 * 60


def select_directives(exclude=DEFAULT_EXCLUDED_DIRECTIVES, only=None):
    names = {d.name for d in ALL_DIRECTIVES}
    for n in list(exclude or ()) + list(only or ()):
        if n not in names:
            raise ValueError(f"Unknown directive {n!r}; known: {sorted(names)}")
    chosen = [d for d in ALL_DIRECTIVES if d.name not in set(exclude or ())]
    if only:
        chosen = [d for d in chosen if d.name in set(only)]
    return set(chosen)


def _names(csv: str | None):
    return [x.strip() for x in csv.split(",") if x.strip()] if csv else None



def _resolve_in_volume(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((Path(VOLUME_MOUNT_PATH) / p).resolve())


def _rewrite_pipeline_yaml_for_modal(orig_yaml_path: str, experiment_name: str) -> str:
    with open(orig_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_mount = Path(VOLUME_MOUNT_PATH)

    def make_abs(p: str) -> str:
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str(base_mount / p)

    # Rewrite pipeline output path(s)
    pipeline_cfg = cfg.get("pipeline", {})
    output_root = base_mount / "outputs" / experiment_name
    if isinstance(pipeline_cfg, dict):
        out = pipeline_cfg.get("output")
        if isinstance(out, dict):
            if isinstance(out.get("path"), str):
                original_name = Path(out["path"]).name
                output_root.mkdir(parents=True, exist_ok=True)
                out["path"] = str(output_root / original_name)
            if isinstance(out.get("intermediate_dir"), str):
                out["intermediate_dir"] = str(output_root / "intermediates")

    # Save rewritten YAML into volume tmp
    tmp_dir = base_mount / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    new_yaml_path = tmp_dir / f"{Path(orig_yaml_path).stem}_modal.yaml"
    with open(new_yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return str(new_yaml_path)


# retries: a crashed container is restarted with the same arguments and resumes from the checkpoint.
@app.function(image=search_image, secrets=[modal.Secret.from_dotenv()], volumes={VOLUME_MOUNT_PATH: volume},
              timeout=SEARCH_TIMEOUT_S, retries=modal.Retries(max_retries=2, initial_delay=60.0))
def run_mcts_remote(
    yaml_path: str,
    dataset_path: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
    experiment_name: str = "mcts_experiment",
    max_iterations: int = 100,
    exploration_weight: float = 1.414,
    model: str = DEFAULT_MODEL,
    dataset: str = "cuad",
    ground_truth_path: str | None = None,
    original_query_result: Dict[str, Any] | None = None,
    build_first_layer: Optional[bool] = False,
    selector_backend: str = "llm",
    jev_model: str = DEFAULT_JEV_MODEL,
    jev_input_price: float = DEFAULT_INPUT_PRICE,
    max_budget_usd: float | None = None,
    exclude_directives: str | None = ",".join(DEFAULT_EXCLUDED_DIRECTIVES),
    only_directives: str | None = None,
    cost_guard_fraction: float = 0.10,
    prior: Dict[str, Any] | None = None,
):
    os.environ["EXPERIMENT_OUTPUT_DIR"] = str(Path(VOLUME_MOUNT_PATH) / "outputs")
    resolved_output_dir = _resolve_in_volume(output_dir) if output_dir else None
    
    # Write a temporary YAML with dataset/output paths rewritten into the mounted volume
    modal_yaml_path = _rewrite_pipeline_yaml_for_modal(yaml_path, experiment_name)

    results = run_mcts_experiment(
        yaml_path=modal_yaml_path,
        dataset_path=dataset_path,
        data_dir=data_dir,
        output_dir=resolved_output_dir,
        experiment_name=experiment_name,
        max_iterations=max_iterations,
        exploration_weight=exploration_weight,
        model=model,
        dataset=dataset,
        ground_truth_path=ground_truth_path,
        original_query_result=original_query_result,
        build_first_layer=build_first_layer,
        selector_backend=selector_backend,
        jev_model=jev_model,
        jev_input_price=jev_input_price,
        max_budget_usd=max_budget_usd,
        exclude_directives=_names(exclude_directives),
        only_directives=_names(only_directives),
        checkpoint_hook=volume.commit,
        cost_guard_fraction=cost_guard_fraction,
        prior=prior,
    )
    volume.commit()
    return results


@app.local_entrypoint()
def modal_main_mcts(
    yaml_path: str,
    dataset_path: str,
    experiment_name: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
    max_iterations: int = 100,
    exploration_weight: float = 1.414,
    model: str = DEFAULT_MODEL,
    dataset: str = "cuad",
    ground_truth: str | None = None,
    build_first_layer: bool = False,
    selector_backend: str = "llm",
    jev_model: str = DEFAULT_JEV_MODEL,
    jev_input_price: float = DEFAULT_INPUT_PRICE,
    max_budget_usd: float = 0.0,
    exclude_directives: str = ",".join(DEFAULT_EXCLUDED_DIRECTIVES),
    only_directives: str = "",
    cost_guard_fraction: float = 0.10,
    prior_path: str = "",
):
    # Note: original_query_result cannot be passed via CLI, so it's not included here.
    # Use programmatic calls to run_mcts_remote if you need to pass original_query_result.
    # spawn, not remote: the search runs on Modal without waiting on this client, so a lost local
    # connection does not cancel it. Launch with `modal run --detach` so the app outlives the client;
    # follow it with `modal app logs <app id>` and find results on the volume.
    call = run_mcts_remote.spawn(
        yaml_path=yaml_path,
        dataset_path=dataset_path,
        data_dir=data_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        max_iterations=max_iterations,
        exploration_weight=exploration_weight,
        model=model,
        dataset=dataset,
        ground_truth_path=ground_truth,
        original_query_result=None,
        build_first_layer=build_first_layer,
        selector_backend=selector_backend,
        jev_model=jev_model,
        jev_input_price=jev_input_price,
        max_budget_usd=max_budget_usd or None,
        exclude_directives=exclude_directives,
        only_directives=only_directives or None,
        cost_guard_fraction=cost_guard_fraction,
        prior=json.load(open(prior_path)) if prior_path else None,
    )
    print(f"Search spawned on Modal (function call {call.object_id}); it keeps running if this client exits.")


def run_mcts_experiment(
    yaml_path: str,
    dataset_path: str,
    data_dir: str = None,
    output_dir: str = None, 
    experiment_name: str = "mcts_experiment",
    max_iterations: int = 100,
    exploration_weight: float = 1.414,
    model: str = DEFAULT_MODEL,
    dataset: str = "cuad",
    ground_truth_path: str | None = None,
    original_query_result: Dict[str, Any] | None = None,
    build_first_layer: Optional[bool] = False,
    selector_backend: str = "llm",
    jev_model: str = DEFAULT_JEV_MODEL,
    jev_input_price: float = DEFAULT_INPUT_PRICE,
    max_budget_usd: float | None = None,
    exclude_directives=DEFAULT_EXCLUDED_DIRECTIVES,
    only_directives=None,
    checkpoint_hook=None,
    cost_guard_fraction: float = 0.10,
    prior: Dict[str, Any] | None = None,
):
    """
    Run MCTS optimization experiment with specified parameters.
    
    Args:
        yaml_path: Path to the input YAML pipeline file
        dataset_path: Path to the dataset file for sample input data
        data_dir: Directory containing input data files
        output_dir: Directory to save experiment outputs
        experiment_name: Name for this experiment run
        max_iterations: Maximum MCTS iterations
        exploration_weight: UCB exploration parameter (c)
        model: LLM model to use for directive instantiation
    """
    # Set up environment
    if data_dir:
        os.environ['EXPERIMENT_DATA_DIR'] = data_dir
    # Determine output directory (env var, parameter, or default)
    if output_dir is None:
        output_dir = os.environ.get('EXPERIMENT_OUTPUT_DIR', DEFAULT_OUTPUT_DIR)
    # Create output directory
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"🌳 Running MCTS Optimization Experiment")
    print(f"=" * 50)
    print(f"Input Pipeline: {yaml_path}")
    print(f"Data Directory: {data_dir or 'default'}")
    print(f"Output Directory: {output_path}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Exploration Weight (c): {exploration_weight}")
    print(f"Model: {model}")
    print(f"Experiment: {experiment_name}")
    print(f"Dataset for evaluation: {dataset}")
    print()
    # Initialize MCTS
    print("🚀 Initializing MCTS...")
    
    # Load sample input data from dataset_path
    with open(dataset_path, 'r') as f:
        dataset_data = json.load(f)
    
    # Take only the first 5 documents
    if isinstance(dataset_data, list):
        sample_input_data = dataset_data[:5]
    else:
        sample_input_data = dataset_data
    
    
    # Document-length statistics shown to the agent and used for cost estimates, from this run's
    # input documents (label fields stripped).
    from docetl.mcts import param_grid
    from docetl.mcts.label_guard import strip_labels
    if isinstance(dataset_data, list):
        print(param_grid.configure_corpus(strip_labels(dataset_data, dataset)))

    available_actions = select_directives(exclude_directives, only_directives)
    print(f"Directives ({len(available_actions)}): {sorted(d.name for d in available_actions)}")
    # A directory with a checkpoint is a search to continue (a rerun after a crash or timeout).
    resume = (output_path / "checkpoint.json").exists()
    print(f"Budget: {'$%.2f' % max_budget_usd if max_budget_usd else 'none'}; "
          f"{'resuming from checkpoint' if resume else 'fresh search'}")
    
    # Get dataset statistics
    dataset_stats = get_dataset_stats(dataset, yaml_path)
    
    # Initialize MCTS
    mcts = MCTS(
        root_yaml_path=yaml_path,
        available_actions=available_actions,
        sample_input=sample_input_data,
        dataset_stats=dataset_stats,
        dataset_name=dataset,
        exploration_constant=exploration_weight,
        max_iterations=max_iterations,
        model=model,
        output_dir=str(output_path),
        original_query_result=original_query_result,
        build_first_layer=build_first_layer,
        selector_backend=selector_backend,
        jev_model=jev_model,
        jev_input_price=jev_input_price,
        max_budget_usd=max_budget_usd,
        resume=resume,
        checkpoint_hook=checkpoint_hook,
        cost_guard_fraction=cost_guard_fraction,
        prior=prior,
    )
    print(f"✅ MCTS initialized with root node: {yaml_path}")
    # Run MCTS optimization
    print(f"\n🔍 Running MCTS optimization for {max_iterations} iterations...")
    start_time = datetime.now()
    best_nodes = mcts.search()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"✅ MCTS optimization completed in {duration:.2f} seconds")
    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    # Prepare nodes with MCTS-specific attributes for evaluation
    nodes_for_evaluation = []
    for n in mcts.pareto_frontier.plans:
        # Add MCTS-specific attributes to the node
        n.mcts_accuracy = mcts.pareto_frontier.plans_accuracy.get(n)
        n.on_frontier = n in mcts.pareto_frontier.frontier_plans
        nodes_for_evaluation.append(n)
    # Use original query result cost if available, otherwise use MCTS root cost
    root_cost = original_query_result["cost"] if original_query_result and original_query_result["success"] else mcts.root.cost
    
    eval_results = run_dataset_evaluation(
        dataset=dataset,
        nodes_or_files=nodes_for_evaluation,
        output_path=output_path,
        ground_truth_path=ground_truth_path,
        method_name="docetl_mcts",
        root_cost=root_cost
    )
    # Save results
    results = {
        "experiment_name": experiment_name,
        "input_pipeline": yaml_path,
        "model": model,
        "selector_backend": selector_backend,
        "jev_model": jev_model if selector_backend == "jev" else None,
        "jev_input_price_per_million": jev_input_price if selector_backend == "jev" else None,
        "jev_selection_calls": len(mcts.selector_calls),
        "jev_selection_cost": sum(r["estimated_cost_usd"] for r in mcts.selector_calls),
        "total_search_cost": mcts.total_search_cost,
        "max_iterations": max_iterations,
        "exploration_weight": exploration_weight,
        "data_dir": data_dir,
        "output_dir": str(output_path),
        "dataset": dataset,
        "ground_truth": ground_truth_path,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "num_best_nodes": len(best_nodes) if best_nodes else 0,
        "total_nodes_explored": (len(mcts._all_nodes()) - 1) if hasattr(mcts, "_all_nodes") else None,
        "original_query_cost": original_query_result["cost"] if original_query_result and original_query_result["success"] else None,
        "original_query_success": original_query_result["success"] if original_query_result else None,
    }
    if eval_results:
        results["evaluation_file"] = str(output_path / "evaluation_metrics.json")
    # Save Pareto frontier if available
    if hasattr(mcts, 'pareto_frontier') and mcts.pareto_frontier.frontier_plans:
        pareto_file = output_path / "pareto_frontier.json"
        pareto_data = []
        
        for solution in mcts.pareto_frontier.frontier_plans:
            pareto_data.append({
                "accuracy": mcts.pareto_frontier.plans_accuracy.get(solution, None),
                "cost": getattr(solution, 'cost', None),
                "value": getattr(solution, 'value', 0),
                "config_path": str(getattr(solution, 'yaml_file_path', None)) if getattr(solution, 'yaml_file_path', None) is not None else None
            })
        
        with open(pareto_file, 'w') as f:
            json.dump(pareto_data, f, indent=2)
        
        results["pareto_frontier_file"] = str(pareto_file)
        print(f"📈 Pareto frontier saved to: {pareto_file}")
    # Save experiment summary
    summary_file = output_path / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📋 Experiment Summary:")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Best Configs Found: {results['num_best_nodes']}")
    print(f"   Summary saved to: {summary_file}")
    print(f"   All outputs in: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run MCTS reasoning optimization experiment", 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic MCTS run
  python run_mcts.py --yaml_path ./pipeline.yaml --experiment_name mcts_test
  
  # With custom parameters
  python run_mcts.py --yaml_path ./pipeline.yaml --data_dir ./data --output_dir ./results --max_iterations 200 --experiment_name mcts_deep
  
  # High exploration
  python run_mcts.py --yaml_path ./pipeline.yaml --exploration_weight 2.0 --experiment_name mcts_explore
        """
    )
    
    parser.add_argument("--yaml_path", type=str, required=True,
                       help="Path to the input YAML pipeline file")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the dataset file for sample input data")
    parser.add_argument("--data_dir", type=str,
                       help="Directory containing input data files (sets EXPERIMENT_DATA_DIR)")
    parser.add_argument("--output_dir", type=str,
                       help=f"Directory to save experiment outputs (default: EXPERIMENT_OUTPUT_DIR env var or {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Name for this experiment run")
    parser.add_argument("--max_iterations", type=int, default=100,
                       help="Maximum MCTS iterations (default: 100)")
    parser.add_argument("--exploration_weight", type=float, default=1.414,
                       help="UCB exploration parameter c (default: 1.414)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                       help=f"LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset", type=str, default="cuad", help="Dataset name for evaluation (default: cuad)")
    parser.add_argument("--ground_truth", type=str, help="Path to ground-truth file (if not default)")
    
    parser.add_argument("--selector_backend", choices=["llm", "jev"], default="llm",
                        help="Model backend for directive selection only")
    parser.add_argument("--jev_model", default=DEFAULT_JEV_MODEL)
    parser.add_argument("--jev_input_price", type=float, default=DEFAULT_INPUT_PRICE,
                        help="Jev input price in USD per million tokens (output is free)")
    parser.add_argument("--max_budget_usd", type=float, default=None)
    parser.add_argument("--exclude_directives", default=",".join(DEFAULT_EXCLUDED_DIRECTIVES))
    parser.add_argument("--only_directives", default=None)
    parser.add_argument("--cost_guard_fraction", type=float, default=0.10,
                        help="Skip a child whose estimated cost is above this share of the budget left")
    parser.add_argument("--prior_path", default=None,
                        help="JSON from an earlier sweep: evidence text, prior counts, seed view")
    args = parser.parse_args()
    
    result = run_mcts_experiment(
        yaml_path=args.yaml_path,
        dataset_path=args.dataset_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        max_iterations=args.max_iterations,
        exploration_weight=args.exploration_weight,
        model=args.model,
        dataset=args.dataset,
        ground_truth_path=args.ground_truth,
        selector_backend=args.selector_backend,
        jev_model=args.jev_model,
        jev_input_price=args.jev_input_price,
        max_budget_usd=args.max_budget_usd,
        exclude_directives=_names(args.exclude_directives),
        only_directives=_names(args.only_directives),
        cost_guard_fraction=args.cost_guard_fraction,
        prior=json.load(open(args.prior_path)) if args.prior_path else None,
    )
    
    print("\n🎉 MCTS experiment completed successfully!")


if __name__ == "__main__":
    main()