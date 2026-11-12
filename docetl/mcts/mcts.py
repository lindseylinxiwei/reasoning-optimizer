import json
import os
import re
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import litellm
import yaml
import threading

from docetl.reasoning_optimizer.directives import (
    DEFAULT_MODEL,
    DEFAULT_OPERATOR_MODEL,
    DIRECTIVE_GROUPS,
    MULTI_INSTANCE_DIRECTIVES,
    Directive,
)
from docetl.reasoning_optimizer.directives.agent_utils import (
    agent_completion,
    is_together_model,
    _extract_json_from_text,
)
from docetl.reasoning_optimizer.directives.jev_classify import is_jev_op, jev_compatible
from docetl.reasoning_optimizer.directives.agent_utils import track_agent_cost
from .tree_log import render_tree
from docetl.reasoning_optimizer.directives.jev_fusion import JevFusionDirective
from .jev_fit import JEV_FIT_TOKENS, SHORTENING_FOR_JEV, fits_jev
from docetl.reasoning_optimizer.directives.change_model_cost import (
    ChangeModelCostDirective,
    create_model_specific_directives,
    MODEL_COSTS,
    MODEL_STATS,
    first_layer_yaml_paths,
    FRONTIER_MODELS
)
from docetl.reasoning_optimizer.op_descriptions import *
from docetl.utils import extract_output_from_json
from .jev_selector import JevSelector, JevAPIError, DEFAULT_JEV_MODEL, DEFAULT_INPUT_PRICE
from .label_guard import label_free_yaml, sanitize_pipeline, sanitized_copy, strip_labels
from .Node import Node
from .ParetoFrontier import ParetoFrontier
from .mcts_utils import *
from . import param_grid as _param_grid
from .param_grid import GRIDS, DEFAULT_NOISE_THRESHOLD, ParamTable, Result, _key, estimate_child_cost, grid_prompt, prune
from .search_state import (
    CHECKPOINT, COSTS_LOG, NODES_LOG, PARAM_TABLE, append_jsonl, capture_applies, find_rewrite,
    install_apply_capture, ledger_totals, node_record, read_checkpoint, read_jsonl, reapply,
    write_checkpoint,
)
from pathlib import Path
from experiments.reasoning.utils import VOLUME_MOUNT_PATH

# Import evaluation function lookup
import sys
sys.path.append("../../experiments/reasoning")
try:
    from experiments.reasoning.evaluation.utils import get_evaluate_func
except ImportError:
    # Fallback import path
    sys.path.append(
        os.path.join(os.path.dirname(__file__), "../../experiments/reasoning")
    )
    from evaluation.utils import get_evaluate_func


class MCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation for DocETL query plan optimization.

    This class implements the four phases of MCTS with Pareto frontier integration:
    1. Selection: Choose the best child using UCB with Pareto value consideration
    2. Expansion: Add new children to the tree
    3. Simulation: Execute a pipeline to get value
    4. Backpropagation: Update node values up the tree

    The MCTS optimizes for both cost and accuracy using the Pareto frontier.
    """

    def __init__(
        self,
        root_yaml_path: str,
        available_actions: set[Directive],
        sample_input,
        dataset_stats: str,
        dataset_name: str,
        exploration_constant: float = 1.414,
        max_iterations: int = 20,
        max_time: Optional[float] = 600.0,
        expansion_count: int = 6,
        model=DEFAULT_MODEL,
        output_dir: Optional[str] = None,
        original_query_result: Optional[Dict[str, Any]] = None,
        build_first_layer: Optional[bool] = False,
        selector_backend: str = "llm",
        jev_model: str = DEFAULT_JEV_MODEL,
        jev_input_price: float = DEFAULT_INPUT_PRICE,
        max_consecutive_failures: int = 5,
        max_budget_usd: Optional[float] = None,
        noise_threshold: float = DEFAULT_NOISE_THRESHOLD,
        cost_guard_fraction: float = 0.10,
        use_param_grid: bool = True,
        explore_every: int = 3,
        explore_k: int = 3,
        prior: Optional[Dict[str, Any]] = None,
        resume: bool = False,
        checkpoint_hook=None,
    ):
        """
        Initialize the MCTS algorithm with Pareto frontier integration.

        Args:
            root_yaml_path: Path to the initial YAML configuration file
            available_actions: List of available actions for expansion
            exploration_constant: UCB exploration constant (default: sqrt(2))
            max_iterations: Maximum number of MCTS iterations
            max_time: Maximum time to run MCTS in seconds (None for no limit)
            sample_input: sample input data
            output_dir: Directory to save new pipeline files (None means same dir as original)
            original_query_result: Pre-executed original query result to avoid re-execution
            max_consecutive_failures: Abort the search after this many failed iterations in a row
            max_budget_usd: Stop the search once agent plus pipeline spending reaches this amount
            noise_threshold: Scores within this of each other are a tie in the parameter grid
            cost_guard_fraction: Skip a child whose estimated cost exceeds this share of the
                remaining budget
            use_param_grid: Fan out one child per selected value of a rewrite's main parameter
            explore_every: Every n-th agent decision is an exploration turn: the valid choices are
                limited to the explore_k least-chosen applicable rewrites (ties included). 0 turns it off.
            explore_k: How many of the least-chosen rewrites an exploration turn offers
            prior: Results from an earlier sweep, all optional:
                evidence_text: shown in every expansion prompt
                prior_counts: {rewrite: configurations already tested}; counted as earlier choices
                    when ranking rewrites for exploration turns and in the valid-choices order
                seed: {"directive": "take_head_tail", "head_words": N, "document_key": K}: before the
                    first iteration, each first-layer plan gets one child with this cut on every
                    operation that reads K (a fresh search only)
            resume: Rebuild the tree from <output_dir>/checkpoint.json when it exists
            checkpoint_hook: Called after every checkpoint (e.g. to commit a Modal volume)
        """

        if selector_backend not in {"llm", "jev"}:
            raise ValueError("selector_backend must be 'llm' or 'jev'")
        self.selector_backend = selector_backend
        self.jev_selector = (
            JevSelector(model=jev_model, input_price=jev_input_price)
            if selector_backend == "jev" else None
        )
        self.selector_calls = []
        # Ground truth must never reach the agent or a candidate pipeline: the root and every
        # node derived from it read label-free copies of the input data (see label_guard).
        root_yaml_path = label_free_yaml(root_yaml_path, dataset_name, output_dir)
        self.root = Node(root_yaml_path, c=exploration_constant)
        self.tree_lock = threading.RLock()
        self.max_concurrent_agents = 3
        
        # Start with base available actions
        self.available_actions = set(available_actions)

        # Add all possible model-specific directives to available actions
        all_models = list(MODEL_COSTS.get(dataset_name, {}).keys()) if dataset_name in MODEL_COSTS else []
        
        for current_model in all_models:
            model_specific_directive = create_model_specific_directives(current_model)
            self.available_actions.add(model_specific_directive)
        
        # Initialize root node's action_used to include all change model directives
        for op_name in self.root.op_dict.keys():
            for directive in self.available_actions:
                if isinstance(directive, ChangeModelCostDirective):
                    self.root.mark_action_used(op_name, directive)
        
        # Initialize action tracking for all actions (base + model-specific)
        self.action_rewards = {action: 0.0 for action in self.available_actions}
        self.action_cost_changes = {action: 0.0 for action in self.available_actions}
        self.action_accuracy_changes = {action: 0.0 for action in self.available_actions}
        self.action_counts = {action: 0.0 for action in self.available_actions}
        # (cost change, accuracy change) vs the parent for every child that ran successfully; the
        # per-rewrite stats shown to the agent are computed from this, so skipped or failed
        # children and repeated frontier updates don't distort them.
        self.action_outcomes: Dict[Any, List[Tuple[float, float]]] = {}
        # Every rewrite the agent chose, with its children's outcomes; shown when expanding other
        # plans that have the same operation, so parallel or sibling expansions don't repeat it blindly.
        self.attempts: List[Dict[str, Any]] = []
        self._attempt_local = threading.local()
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.iteration_count = 0
        self.expansion_count = expansion_count
        self.start_time = None
        self.model = model
        self.sample_input = strip_labels(sample_input, dataset_name)
        self.dataset_stats = dataset_stats
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        
        # Set up evaluation function and dataset metrics
        self.evaluate_func = get_evaluate_func(dataset_name, mode="train")
        self.dataset_metrics = {
            "cuad": "avg_f1",
            "blackvault": "avg_distinct_locations",
            "game_reviews": "weighted_score",
            "medec": "combined_score", 
            "sustainability": "combined_score",
            "biodex": "avg_rp_at_5",
        }
        # Set up log file path
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.log_path = os.path.join(self.output_dir, "mcts_tree_log.txt")
        else:
            self.log_path = "mcts_tree_log.txt"

        # Initialize log file (clear it)
        with open(self.log_path, "a" if resume else "w", encoding="utf-8") as f:
            f.write(f"MCTS search tree log\n")
            f.write(f"Root YAML: {root_yaml_path}\n")
            f.write(f"Max iterations: {max_iterations}\n")
            f.write(f"{'='*50}\n")

        # Initialize Pareto frontier
        self.pareto_frontier = ParetoFrontier(
            self.action_rewards, self.action_cost_changes, 
            self.action_accuracy_changes, dataset_name
        )

        # Create comprehensive directive mapping 
        self.directive_name_to_obj = {
            action.name: action for action in self.available_actions
        }
       
        # Failed iterations (an exception or no child created) are counted and logged rather than
        # retried silently; the search aborts after max_consecutive_failures in a row.
        self.failed_iteration_count = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = max_consecutive_failures
        
        # Track total cost for all completion calls during search
        self.total_search_cost = 0.0
        
        # Track if action space is exhausted for early termination
        self.action_space_exhausted = False

        # Budget, grid, logs and checkpoint (see search_state.py)
        self.max_budget_usd = max_budget_usd
        self.cost_guard_fraction = cost_guard_fraction
        self.use_param_grid = use_param_grid
        self.explore_every = explore_every
        self.explore_k = explore_k
        self.prior = prior or {}
        self.prior_counts: Dict[str, int] = {rewrite_family(k): int(v) for k, v in
                                             (self.prior.get("prior_counts") or {}).items()}
        self.declined: Counter = Counter()     # rewrites offered on exploration turns the agent declined
        self.checkpoint_hook = checkpoint_hook
        self.agent_cost = 0.0
        self.pipeline_cost = 0.0
        self.resumed = False
        self._iteration_stats = threading.local()
        self.reserved_cost = 0.0
        self.consecutive_all_skipped = 0
        self.max_consecutive_all_skipped = 5
        state_dir = self.output_dir or "."
        os.makedirs(state_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(state_dir, CHECKPOINT)
        self.nodes_log_path = os.path.join(state_dir, NODES_LOG)
        self.costs_log_path = os.path.join(state_dir, COSTS_LOG)
        can_resume = resume and os.path.exists(self.checkpoint_path)
        if not can_resume and os.path.exists(self.costs_log_path):
            raise RuntimeError(
                f"{state_dir} already holds a search (costs.jsonl exists). Pass resume=True to continue "
                "it, or use a new output directory.")
        self.param_table = ParamTable.load(os.path.join(state_dir, PARAM_TABLE), threshold=noise_threshold)
        for action in self.available_actions:
            if action.name in GRIDS or action.name == "arbitrary_rewrite":
                install_apply_capture(action)
        if can_resume:
            self._restore_checkpoint()
            return

        if build_first_layer:
            if not MODEL_COSTS.get(dataset_name):
                raise ValueError(
                    f"build_first_layer needs cached per-model runs for '{dataset_name}' in MODEL_COSTS "
                    "(change_model_cost.py) for models in AVAILABLE_MODELS; none were found.")
            print("🔄 Building first layer of the pipeline manually...")
            output_dir = str(Path(VOLUME_MOUNT_PATH) / "outputs")
            for model_yaml_path in first_layer_yaml_paths[dataset_name]:
                model_yaml_path = Path(output_dir) / f"{dataset_name}" / model_yaml_path
                child = Node(model_yaml_path)
                # Cached first-layer configs point at the labeled split; descendants copy this.
                sanitize_pipeline(child.parsed_yaml, dataset_name, self.output_dir)
                child_model = child.parsed_yaml["default_model"]
                child_model = child_model.replace("gemini/", "")
                child.cost = MODEL_COSTS[dataset_name][child_model]
                child.value = 0
                child.visits = 1
                action_name = "change to " + child_model
                action = self.directive_name_to_obj[action_name]
                child.latest_action = action
                child.sample_result = strip_labels(extract_output_from_json(model_yaml_path), dataset_name)
                
                # Mark all change model directives as used for first layer nodes
                # since they are already model-specific plans
                for op_name in child.op_dict.keys():
                    for directive in self.available_actions:
                        if isinstance(directive, ChangeModelCostDirective):
                            child.mark_action_used(op_name, directive)
                
                self.root.add_child(child)
                accuracy = MODEL_STATS[dataset_name][child_model]["acc"]
                self.pareto_frontier.add_plan_f1(child, accuracy)
            
            print(self.pareto_frontier.plans)
            self.root.visits = len(first_layer_yaml_paths[dataset_name])
            self.log_tree_to_file(0)
            self._checkpoint()      # a crash in the first iteration can still resume
            return
        
        # Use original query result if provided, otherwise execute root node
        if original_query_result and original_query_result.get("success"):
            print("🔄 Using pre-executed original query result for MCTS root node")
            # Set root node properties from original query result
            self.root.cost = original_query_result["cost"]
            self.root.sample_result = strip_labels(original_query_result.get("sample_output", []), dataset_name)

            # Update the root node's YAML to point to the original query result file
            if original_query_result.get("output_file_path"):
                print(
                    f"📁 Setting root node output path to: {original_query_result['output_file_path']}"
                )
                try:
                    self.root.parsed_yaml["pipeline"]["output"]["path"] = (
                        original_query_result["output_file_path"]
                    )
                    self.root.result_path = original_query_result["output_file_path"]
                except Exception as e:
                    print(f"⚠️ Could not update root node output path: {e}")
                    
            # Evaluate root node accuracy and add to pareto frontier
            root_accuracy = self.evaluate_node(self.root)
            affected_nodes, is_frontier_updated = self.pareto_frontier.add_plan_f1(self.root, root_accuracy)
            self.root.visits = 1
        else:
            print(
                "▶️ Executing root node for MCTS (original query result not available)"
            )
            # execute root node and add it to the pareto frontier
            cost, accuracy = self.simulate(self.root)
            with self.tree_lock:
                cost_to_add = max(cost,0)
                print(f"💰 Adding simulation cost: ${cost_to_add:.4f} (total before: ${self.total_search_cost:.4f})")
                self._add_cost("pipeline", cost_to_add)
            affected_nodes, is_frontier_updated = self.add_to_frontier(self.root, accuracy)
            with self.tree_lock:
                self.backpropagate(affected_nodes, self.root)
            self.root.visits = 1
        self._checkpoint()


    def evaluate_node(self, node: Node) -> float:
        """
        Evaluate a node's accuracy by running the evaluation function on its output.
        
        Args:
            node: Node object representing the plan
            
        Returns:
            The accuracy score for the node
        """
        if node.cost == -1:  # Handle error case
            return float("-inf")
            
        result_file_path = node.parsed_yaml["pipeline"]["output"]["path"]
        print("result_file_path", result_file_path)

        try:
            results = self.evaluate_func("docetl_preprint", result_file_path)

            # Extract the appropriate metric based on dataset
            primary_metric = self.dataset_metrics.get(self.dataset_name)
            if primary_metric and primary_metric in results:
                true_accuracy = results[primary_metric]
            else:
                # Fallback to first numerical value found if dataset unknown or metric missing
                true_accuracy = next(
                    (v for v in results.values() if isinstance(v, (int, float))), 0.5
                )
                
        except Exception as e:
            print(f"⚠️ Evaluation failed for node {node.get_id()}: {e}")
            # Log -inf occurrence for debugging
            node._log_inf_occurrence("evaluation_failure", str(e), node.yaml_file_path)
            # Return -inf when evaluation fails, same as unexecutable plans
            return float("-inf")
            
        # Guard against NaN values
        if true_accuracy is None or (isinstance(true_accuracy, float) and (true_accuracy != true_accuracy)):  # NaN check
            print(f"⚠️ Evaluation returned NaN for node {node.get_id()}, setting to -inf")
            # Log -inf occurrence for debugging
            node._log_inf_occurrence("nan_evaluation", f"Evaluation returned NaN or None: {true_accuracy}", node.yaml_file_path)
            return float("-inf")

        return true_accuracy

    def search(self):
        """
        Perform MCTS search to find the optimal query plan with concurrent agents.

        Returns:
            Tuple of (best_node, search_statistics)
        """
        self.start_time = time.time()
        if not self.resumed:
            self.iteration_count = 0
            if (getattr(self, "prior", None) or {}).get("seed"):
                self._seed_first_layer()

        print(f"Starting concurrent MCTS search with {self.max_iterations} iterations and {self.max_concurrent_agents} agents...")
        print(f"Root node cost: ${self.root.cost:.2f}")

        with ThreadPoolExecutor(max_workers=self.max_concurrent_agents) as executor:
            while self.should_continue():
                # Submit up to max_concurrent_agents tasks
                futures = []
                agents_to_submit = min(self.max_concurrent_agents, self.max_iterations - self.iteration_count)
                
                for _ in range(agents_to_submit):
                    future = executor.submit(self.mcts_iteration)
                    futures.append(future)
                
                # Wait for at least one agent to complete
                for future in as_completed(futures):
                    try:
                        outcome = future.result()
                        print(f"Agent completed with outcome={outcome}, current iteration_count={self.iteration_count}")
                        error = None
                    except JevAPIError:
                        raise
                    except Exception as e:
                        outcome = False
                        error = "".join(traceback.format_exception(e))
                    success = outcome is True
                    with self.tree_lock:
                        if outcome is None:
                            pass        # the selected node had no rewrite left; it is now excluded
                        elif outcome == "all_skipped":
                            self.consecutive_all_skipped += 1
                            self._log_failure("every child was skipped by the cost guard")
                        if success:
                            self.consecutive_all_skipped = 0
                            self.iteration_count += 1
                            self.consecutive_failures = 0
                            print(f"Updated iteration_count to {self.iteration_count}")
                        elif outcome is False and not self.action_space_exhausted:
                            self.failed_iteration_count += 1
                            self.consecutive_failures += 1
                            print(f"❌ Iteration failed ({self.consecutive_failures} in a row, "
                                  f"{self.failed_iteration_count} total): {error}")
                            self._log_failure(error)
                            if self.consecutive_failures >= self.max_consecutive_failures:
                                for f in futures:
                                    f.cancel()
                                self._checkpoint()
                                raise RuntimeError(
                                    f"Aborting MCTS: {self.consecutive_failures} iterations failed in a row. "
                                    f"Last error:\n{error}")
                        self._checkpoint()

                    # Check if we should continue after each completion
                    if not self.should_continue():
                        break
                
                # Cancel remaining futures if stopping
                if not self.should_continue():
                    for future in futures:
                        future.cancel()

        with self.tree_lock:
            self._checkpoint()

        # Final statistics
        print(f"\nConcurrent MCTS search completed!")
        print(f"Spent: ${self.total_search_cost:.2f} (agent ${self.agent_cost:.2f}, "
              f"pipelines ${self.pipeline_cost:.2f})")
        print(f"Total iterations: {self.iteration_count}")
        print(f"Failed iterations: {self.failed_iteration_count}")
        print(f"Pareto frontier size: {len(self.pareto_frontier)}")
        print(f"Frontier plans: {len(self.pareto_frontier.frontier_plans)}")
        print("pareto frontier yaml files: ")
        for plan in self.pareto_frontier.frontier_plans:
            print(plan.yaml_file_path)

        # Return all frontier plans
        frontier_plans = [
            summary["node"]
            for summary in self.pareto_frontier.get_all_plans_summary()
            if summary["is_frontier"]
        ]

        return frontier_plans


    def mcts_iteration(self):
        """Perform one complete MCTS iteration with thread safety."""
        thread_id = threading.current_thread().ident
        
        # 1. Selection: Find the best leaf node (requires tree lock)
        dual_expand = False
        with self.tree_lock:
            print(f"[Thread {thread_id}] SELECTION")
            leaf = self.select(self.root)
            print(f"[Thread {thread_id}] SELECTED NODE: {leaf.get_id()}")
            # Counted now so parallel iterations spread out; undone below if this iteration
            # evaluates no plan (every child skipped, no child created, or no rewrite left).
            bumps = 1
            if self.is_first_layer_node(leaf) and leaf.visits == 1:
                dual_expand = True
                self.increment_visits_up_tree(leaf)
                bumps = 2
            self.increment_visits_up_tree(leaf)


        # 2. Expansion (requires tree lock). A first-layer node visited for the first time is
        # expanded for both goals; any other node for its goal, falling back to the other goal
        # when its own has no rewrite left. A node with no rewrite left for either goal is marked
        # and never selected again; the search stops only when every leaf is in that state.
        goals = ["acc", "cost"] if dual_expand else None
        children_by_goal = {}
        errors = []
        with self.tree_lock:
            print(f"[Thread {thread_id}] EXPANSION")
            if goals is None:
                first = self.get_optimize_goal(leaf)
                goals = [first]
                fallback = "cost" if first == "acc" else "acc"
            else:
                fallback = None
            exhausted = getattr(leaf, "exhausted_goals", set())
            for goal in goals:
                try:
                    children_by_goal[goal] = self._expand_tracked(leaf, goal)
                except JevAPIError:
                    raise
                except RuntimeError as e:
                    if "Action space may be exhausted" not in str(e):
                        errors.append(f"expand({goal}): {e}")
                        continue
                    exhausted = exhausted | {goal}
                    if fallback and fallback not in exhausted:
                        try:
                            children_by_goal[fallback] = self._expand_tracked(leaf, fallback)
                        except RuntimeError as e2:
                            if "Action space may be exhausted" in str(e2):
                                exhausted = exhausted | {fallback}
                            else:
                                errors.append(f"expand({fallback}): {e2}")
            leaf.exhausted_goals = exhausted
            if exhausted >= {"acc", "cost"} and not children_by_goal:
                leaf.no_actions = True
                print(f"[Thread {thread_id}] node {leaf.get_id()} has no rewrite left; it will not be selected again")
                if all(getattr(n, "no_actions", False) for n in self._all_nodes()
                       if n is not self.root and not n.children and getattr(n, "cost", -1) != -1):
                    self.action_space_exhausted = True
                self._undo_visits(leaf, bumps)
                return None

        if not children_by_goal:
            with self.tree_lock:
                self._undo_visits(leaf, bumps)
            raise RuntimeError("no child was created: " + ("; ".join(errors) or "unknown reason"))

        # 3. Simulation
        self._iteration_stats.executed = 0
        self._iteration_stats.skipped = 0
        for goal, children in children_by_goal.items():
            print(f"HAS LEAF {goal.upper()} SIMULATION")
            self._simulate_children(children, goal)

        with self.tree_lock:
            self.log_tree_to_file(self.iteration_count + 1)

        if self._iteration_stats.executed == 0 and self._iteration_stats.skipped > 0:
            with self.tree_lock:
                self._undo_visits(leaf, bumps)
                self._block_if_unaffordable(leaf, [c for cs in children_by_goal.values() for c in cs])
            print(f"[Thread {thread_id}] every child was skipped by the cost guard")
            return "all_skipped"
        print(f"[Thread {thread_id}] MCTS iteration completed with {self._iteration_stats.executed} evaluated child(ren)")
        return True

    def _simulate_children(self, children: List[Node], goal_type: str) -> bool:
        """
        Simulate a list of children and handle multi-instance vs single instance logic.
        
        Args:
            children: List of child nodes to simulate
            goal_type: Type of goal ("acc" or "cost") for logging purposes
            
        Returns:
            True if frontier was updated, False otherwise
        """
        is_frontier_updated = False
        
        if len(children) > 1 and not any(getattr(c, "grid", None) or getattr(c, "coupled", False) for c in children):
            print(f"Handling {len(children)} multi-instance candidates")
            candidate_results = []
            
            # Simulate all candidates
            for num, candidate in enumerate(children):
                cost, accuracy = self.simulate(candidate)
                self._iteration_stats.executed += 1
                with self.tree_lock:
                    cost_to_add = max(cost, 0)
                    print(f"💰 Adding multi-instance {goal_type} candidate cost: ${cost_to_add:.4f} (total before: ${self.total_search_cost:.4f})")
                    self._add_cost("pipeline", cost_to_add)
                print(f"Multi-instance candidate {num} - Cost: ${cost:.2f}, Accuracy: {accuracy:.4f}")
                if cost != -1 and accuracy != float("-inf"):  # Valid plan
                    candidate_results.append((candidate, accuracy, cost))
                else:
                    print(f"Multi-instance candidate {num} failed during simulation")
            
            if candidate_results:
                # Select the best candidate based on accuracy
                best_candidate, best_accuracy, best_cost = max(candidate_results, key=lambda x: x[1])
                print(f"Selected best multi-instance candidate {best_candidate.get_id()} with accuracy {best_accuracy:.4f} and cost ${best_cost:.2f}")
                
                # Change the best candidate's ID back to a proper counter ID
                old_id = best_candidate.get_id()
                new_id = best_candidate.set_id_to_counter()
                print(f"Updated best candidate ID from {old_id} to {new_id}")
                
                # Delete ALL non-selected candidates (both failed and successful ones that weren't chosen)
                for candidate in children:
                    if candidate != best_candidate:
                        candidate.delete(selected_node_final_id=new_id)
                
                # Process only the best candidate (requires tree lock for backprop)
                with self.tree_lock:
                    affected_nodes, is_frontier_updated = self.add_to_frontier(best_candidate, best_accuracy)
                    self.backpropagate(affected_nodes, best_candidate)
                self._record_result(best_candidate, best_cost, best_accuracy)
            else:
                # If no candidates were successful, delete all of them
                print("No successful candidates found, deleting all multi-instance candidates")
                for candidate in children:
                    candidate.delete(selected_node_final_id=None)
                
        else:
            # Original logic for single instantiations (also grid children and coupled pairs, in order)
            not_ok = set()          # children skipped or failed: a coupled child of theirs is not run
            grid_runs: List[Tuple[Node, float, float]] = []
            for child in children:
                if child.parent in not_ok:
                    reason = "its shortening parent was skipped or failed"
                    self._iteration_stats.skipped += 1
                    print(f"⏭️ Skipping node {child.get_id()}: {reason}")
                    self._log_node(child, None, None, status="skipped", reason=reason)
                    with self.tree_lock:
                        child.delete()
                    not_ok.add(child)
                    continue
                skip_reason = self._cost_guard(child)
                if skip_reason:
                    self._iteration_stats.skipped += 1
                    print(f"⏭️ Skipping node {child.get_id()}: {skip_reason}")
                    self._log_node(child, None, None, status="skipped", reason=skip_reason)
                    with self.tree_lock:
                        child.delete()
                    not_ok.add(child)
                    continue
                try:
                    cost, accuracy = self.simulate(child)
                finally:
                    self._release_reservation(child)
                if cost == -1 or accuracy == float("-inf"):
                    not_ok.add(child)
                self._iteration_stats.executed += 1
                with self.tree_lock:
                    cost_to_add = max(cost, 0)
                    print(f"💰 Adding leaf {goal_type} simulation: ${cost_to_add:.4f} (total before: ${self.total_search_cost:.4f})")
                    self._add_cost("pipeline", cost_to_add)
                if getattr(child, "grid", None) and child not in not_ok:
                    grid_runs.append((child, cost, accuracy))      # added once all its siblings ran
                    continue
                if self._add_result(child, cost, accuracy):
                    is_frontier_updated = True
            for child, cost, accuracy in self._keep_pareto_siblings(grid_runs):
                if self._add_result(child, cost, accuracy):
                    is_frontier_updated = True

        return is_frontier_updated

    def _add_result(self, child: Node, cost: float, accuracy: float) -> bool:
        with self.tree_lock:
            affected_nodes, updated = self.add_to_frontier(child, accuracy)
            self.backpropagate(affected_nodes, child)
        self._record_result(child, cost, accuracy)
        return updated

    def _keep_pareto_siblings(self, runs: List[Tuple[Node, float, float]]) -> List[Tuple[Node, float, float]]:
        """Grid variants of one rewrite under one parent: keep those that are Pareto-optimal among
        themselves under the tie-to-cheaper rule (scores within the noise threshold tie, and the
        cheaper wins); drop the rest from the tree. Every value's result stays in the param table."""
        groups: Dict[Tuple, List[Tuple[Node, float, float]]] = {}
        for run in runs:
            child = run[0]
            groups.setdefault((child.parent.get_id() if child.parent else None, child.grid.get("directive")),
                              []).append(run)
        kept: List[Tuple[Node, float, float]] = []
        for group in groups.values():
            if len(group) == 1:
                kept += group
                continue
            results = [Result(value=str(c.get_id()), cost=cost, score=acc) for c, cost, acc in group]
            winner, dropped = prune(results, self.param_table.threshold)
            for child, cost, acc in group:
                if _key(str(child.get_id())) not in dropped:
                    kept.append((child, cost, acc))
                    continue
                g = child.grid
                reason = (f"dominated by sibling #{winner} (cost and score vs other {g.get('param')} values; "
                          f"scores within {self.param_table.threshold} tie and the cheaper wins)")
                print(f"✂️ Dropping node {child.get_id()} ({g.get('param')}={g.get('value')}): {reason}")
                self.param_table.record(g["directive"], g["op"], g["parent_model"], g["value"], cost, acc,
                                        str(child.get_id()))
                self._log_node(child, cost, acc, status="pruned", reason=reason)
                with self.tree_lock:        # out of the tree; its YAML and outputs stay on disk
                    if child.parent is not None and child in child.parent.children:
                        child.parent.children.remove(child)
        return kept

    def select(self, node: Node) -> Node:
        """
        Select the best child using the Node's best_child method.

        Args:
            node: Starting node for selection

        Returns:
            Selected leaf node
        """
        current = node

        while self.is_fully_explored(current):
            print(f"Node {current.get_id()} is fully explored, selecting best child")
            current = current.best_child()

        return current


    def is_fully_explored(self, node: Node) -> bool:
        """Check if a node has been fully explored based on visit count."""
        return is_fully_explored(node)

    def expansion_prompt_acc(self, node, action_options, input_query) -> tuple[str, str]:
        return create_expansion_prompt_acc(
            node, action_options, input_query, self.available_actions, 
            self.action_cost_changes, self.action_accuracy_changes, self.action_counts, self.sample_input, 
            self.root, node.yaml_file_path, self.dataset_name, self.pareto_frontier.plans_accuracy,
            choice_counts=self._choice_counts(), prior_counts=getattr(self, "prior_counts", {}),
        )

    def expansion_prompt_cost(self, node, action_options, input_query) -> tuple[str, str]:
        return create_expansion_prompt_cost(
            node, action_options, input_query, self.available_actions,
            self.action_cost_changes, self.action_accuracy_changes, self.action_counts, self.sample_input,
            self.root, node.yaml_file_path, self.dataset_name, self.pareto_frontier.plans_accuracy,
            choice_counts=self._choice_counts(), prior_counts=getattr(self, "prior_counts", {}),
        )

    def is_first_layer_node(self, node: Node) -> bool:
        """
        Check if a node is a first layer node (direct child of root with model change action).
        
        Args:
            node: Node to check
            
        Returns:
            True if this is a first layer node, False otherwise
        """
        # First layer nodes are direct children of root
        if node.parent != self.root:
            return False
        
        # First layer nodes have a model change action
        if not hasattr(node, 'latest_action') or node.latest_action is None:
            return False
        
        # Check if the action is a model change directive
        return isinstance(node.latest_action, ChangeModelCostDirective)

    def get_optimize_goal(self, node: Node) -> str:
        """
        Determine the optimization goal based on the node's accuracy relative to other plans.
        
        Args:
            node: Node to determine optimization goal for
            
        Returns:
            "cost" for nodes in the top 50% accuracy plans, "acc" otherwise
        """
        if not self.pareto_frontier.plans_accuracy:
            # If no plans exist yet, optimize for accuracy
            return "acc"
        
        # Get all accuracy values from tracked plans
        all_accuracies = list(self.pareto_frontier.plans_accuracy.values())
        
        # Filter out invalid accuracies (negative infinity indicates failed plans)
        valid_accuracies = [acc for acc in all_accuracies if acc != float("-inf")]
        
        if not valid_accuracies:
            # If no valid accuracies exist, optimize for accuracy
            return "acc"
        
        # Calculate the 50th percentile threshold
        valid_accuracies.sort()
        n = len(valid_accuracies)
        percentile_50_index = n // 2  # This gives us the median (50th percentile)
        threshold = valid_accuracies[percentile_50_index]
        
        # Get this node's accuracy (if it exists in the frontier)
        node_accuracy = self.pareto_frontier.plans_accuracy.get(node, float("-inf"))
        
        # If node accuracy is above the 50th percentile, optimize for cost
        # Otherwise, optimize for accuracy
        if node_accuracy > threshold:
            return "cost"
        else:
            return "acc"

    def _record_jev_call(self, node, goal, record):
        record = dict(record, node_id=node.get_id(), optimize_goal=goal)
        with self.tree_lock:
            self._add_cost("agent", record["estimated_cost_usd"])
            self.selector_calls.append(record)
            if self.output_dir:
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
                with open(Path(self.output_dir) / "jev_selection_calls.jsonl", "a") as stream:
                    stream.write(json.dumps(record) + "\n")

    def expand(self, node: Node, optimize_goal: str) -> List[Node]:
        """
        Expand a node with a specific optimization goal.
        
        Args:
            node: Node to expand
            optimize_goal: The optimization goal to use
            
        Returns:
            List of newly created child nodes
        """
        max_retries = 3
        retry_count = 0

        # Build action options and initial prompt once
        op_list = list(node.op_dict.keys())
        banned_directives = set()
        last_step = None
        last_op = None
        if len(node.memo) > 0:
            last_step = node.memo[-1]
            if node.value < 0:
                last_directive = last_step[0]
                last_op = last_step[1]
                for group in DIRECTIVE_GROUPS:
                    directive = self.directive_name_to_obj.get(last_directive)
                    if directive in DIRECTIVE_GROUPS[group]:
                        banned_directives = set(DIRECTIVE_GROUPS[group])

        explore = None
        if optimize_goal == "acc":
            action_options = []  # a list of tuple
            for op_name in op_list:
                if op_name in node.used_actions:
                    used_actions = node.used_actions[op_name]
                else:
                    used_actions = set()
                
                # Get compression directives to exclude for code_map and extract operations
                compression_exclusions = (
                    get_excluded_directives_for_operation(node, op_name) | self._gleaning_exclusions(node, op_name))
                
                if last_op is None or last_op != op_name:
                    banned_directives = set()
                
                # Accuracy goal: offer only models that scored higher than the operator's current model
                current_model = self._op_model(node, op_name)
                dynamic_actions = []
                
                for directive in self.available_actions:
                    if isinstance(directive, ChangeModelCostDirective):
                        if directive.target_model and self._is_more_accurate_model(directive.target_model, current_model):
                            dynamic_actions.append(directive)
                    else:
                        # Keep other directives as-is
                        dynamic_actions.append(directive)
                
                action_space = (
                    set(dynamic_actions) - banned_directives - used_actions - compression_exclusions
                )
                ops_now = node.parsed_yaml["operations"]
                action_space = {a for a in action_space if fits_plan_structure(ops_now, op_name, a.name)}  # The actions that are not used on this operator and not excluded by group
                for action in action_space:
                    action_options.append((op_name, action.name))
            if len(action_options) < 1:
                print("NO ACTION FOUND")
                raise RuntimeError(
                    "No applicable action found for expansion. Action space may be exhausted or all actions are inapplicable."
                )
            print("OPTIMIZING ACC:")
            all_options = action_options
            explore = self._exploration_options(action_options)
            if explore is not None:
                action_options = explore
            user_message, condensed_user_message = self.expansion_prompt_acc(
                node, action_options=action_options, input_query=node.parsed_yaml
            )
            tried = (self._tried_elsewhere_text(node) + self._evidence_text()
                     + (EXPLORATION_NOTE if explore is not None else ""))
            user_message, condensed_user_message = user_message + tried, condensed_user_message + tried

        elif optimize_goal == "cost":
            action_options = []  # a list of tuple
            for op_name in op_list:
                if op_name in node.used_actions:
                    used_actions = node.used_actions[op_name]
                else:
                    used_actions = set()
                
                # Get compression directives to exclude for code_map and extract operations
                compression_exclusions = (
                    get_excluded_directives_for_operation(node, op_name) | self._gleaning_exclusions(node, op_name))
                
                if last_op is None or last_op != op_name:
                    banned_directives = set()

                # Cost goal: offer only models cheaper than the operator's current model
                current_model = self._op_model(node, op_name)
                dynamic_cost_directives = []
                
                for directive in self.available_actions:
                    if isinstance(directive, ChangeModelCostDirective):
                        if directive.target_model and self._is_cheaper_model(directive.target_model, current_model):
                            dynamic_cost_directives.append(directive)
                    else:
                        dynamic_cost_directives.append(directive)
                
                action_space = (
                    set(dynamic_cost_directives) - banned_directives - used_actions - compression_exclusions
                )
                ops_now = node.parsed_yaml["operations"]
                action_space = {a for a in action_space if fits_plan_structure(ops_now, op_name, a.name)}  # The cost actions that are not used on this operator and not excluded by group
                for action in action_space:
                    action_options.append((op_name, action.name))
            print("OPTIMIZING COST:")
            if len(action_options) < 1:
                print("NO ACTION FOUND")
                raise RuntimeError(
                    "No applicable action found for expansion. Action space may be exhausted or all actions are inapplicable."
                )
            all_options = action_options
            explore = self._exploration_options(action_options)
            if explore is not None:
                action_options = explore
            user_message, condensed_user_message = self.expansion_prompt_cost(
                node, action_options=action_options, input_query=node.parsed_yaml
            )
            tried = (self._tried_elsewhere_text(node) + self._evidence_text()
                     + (EXPLORATION_NOTE if explore is not None else ""))
            user_message, condensed_user_message = user_message + tried, condensed_user_message + tried

        # Initialize messages with accumulated message history from the path to this node
        messages = node.message_history.copy()
        message_condensed = node.message_history.copy()

        # Add the current system message and user message for this expansion
        if not messages or messages[0]["role"] != "system":
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You are an expert query optimization agent for document processing pipelines. Your role is to analyze user queries and apply rewrite directives to create more accurate and cost effective execution plans. Your output must follow the structured output format.",
                },
            )

        messages.append({"role": "user", "content": user_message})
        if len(message_condensed) == 0 or message_condensed[0]["role"] != "system":
            message_condensed.append(
                {
                    "role": "system",
                    "content": "You are an expert query optimization agent for document processing pipelines. Your role is to analyze user queries and apply rewrite directives to create more accurate and cost effective execution plans. Your output must follow the structured output format.",
                }
            )
        message_condensed.append({"role": "user", "content": condensed_user_message})

        # Trim the history to prevent context window overflow before sending to the model
        messages = trim_history(messages)
        message_condensed = trim_history(message_condensed)

        while retry_count < max_retries:

            if self.jev_selector is not None:
                parsed_selection = self.jev_selector.select(
                    messages=messages,
                    action_options=action_options,
                    directives=self.directive_name_to_obj,
                    pipeline=node.parsed_yaml,
                    goal=optimize_goal,
                    on_call=lambda record: self._record_jev_call(node, optimize_goal, record),
                )
                reply = json.dumps(parsed_selection)
            else:
                response, call_cost = agent_completion(
                    model=self.model,
                    messages=messages,
                    response_format=ExpandResponseFormat,
                )
                with self.tree_lock:
                    print(f"💰 Adding LLM call cost: ${call_cost:.4f} (total before: ${self.total_search_cost:.4f})")
                    self._add_cost("agent", call_cost)
                reply = response.choices[0].message.content

            try:
                # Handle both direct JSON and text with embedded JSON (for reasoning models)
                if self.jev_selector is None and is_together_model(self.model):
                    parsed = _extract_json_from_text(reply)
                else:
                    parsed = json.loads(reply)
                directive_name = parsed.get("directive")
                target_op_list = parsed.get("operators")
                print(f"Directive: {directive_name}, Target ops: {target_op_list}")
                messages.append({"role": "assistant", "content": reply})
                message_condensed.append({"role": "assistant", "content": reply})
            except Exception as e:
                print(f"Failed to parse agent response: {e}")
                retry_count += 1
                continue

            if explore is not None and str(directive_name).strip().lower() == "none":
                # The agent says no under-tried rewrite fits this plan: log why, offer the full list.
                self._log_exploration(node, optimize_goal, explore, None, opt_out=parsed.get("reason") or "")
                with self.tree_lock:      # a decline counts as one use of each rewrite offered
                    self.declined.update({rewrite_family(n) for _, n in explore})
                explore, action_options = None, all_options
                feedback_message = ("Understood. This is now a normal turn; all valid choices for this plan are:\n"
                                    + format_action_options(action_options, self._choice_counts(),
                                                            getattr(self, "prior_counts", {}))
                                    + "Pick one directive and its operator(s) from this list.")
                messages.append({"role": "user", "content": feedback_message})
                message_condensed.append({"role": "user", "content": feedback_message})
                continue


            # Check if directive is already used for this plan + target ops
            directive = self.directive_name_to_obj.get(directive_name)
            if directive is None:
                print(f"Unknown directive name: {directive_name}")
                retry_count += 1
                continue

            offered = {(op, name) for op, name in action_options}
            target_error = self._target_error(node, directive_name, target_op_list, offered)
            if target_error:
                retry_count += 1
                print(f"Agent chose an action that was not offered: {directive_name} on {target_op_list} ({target_error})")
                feedback_message = (
                    f"'{directive_name}' on {target_op_list} is not a valid choice: {target_error}. Pick a "
                    "directive and operator(s) from the options listed in the previous message.")
                messages.append({"role": "user", "content": feedback_message})
                message_condensed.append({"role": "user", "content": feedback_message})
                continue

            # Check if already used
            already_used = False
            for target_op in target_op_list:
                if directive in node.used_actions[target_op]:
                    already_used = True
                    break

            if already_used:
                retry_count += 1
                print(
                    f"Directive '{directive_name}' already used for these ops. Retry {retry_count}/{max_retries}"
                )
                # Append feedback message as new user message
                feedback_message = f"We have already tried the directive '{directive_name}' on these operators: {target_op_list}. Please pick another directive from the available options we listed in the previous message."
                messages.append({"role": "user", "content": feedback_message})
                message_condensed.append({"role": "user", "content": feedback_message})
                continue

            # Valid directive found - break out of directive selection loop
            break

        # If we've exhausted directive selection retries
        if retry_count >= max_retries:
            raise RuntimeError(
                f"Failed to find unused directive after {max_retries} retries"
            )

        orig_default_model = node.parsed_yaml.get("default_model")

        datasets = node.parsed_yaml.get("datasets", {})
        input_file_path = None
        if isinstance(datasets, dict) and datasets:
            first_dataset = next(iter(datasets.values()))
            if isinstance(first_dataset, dict):
                input_file_path = sanitized_copy(first_dataset.get("path"), self.dataset_name, self.output_dir)

        # Mark action as used and increment use count immediately
        for target_op in target_op_list:
            node.mark_action_used(target_op, directive)
        self._register_attempt(node, directive.name, target_op_list, optimize_goal)
        if explore is not None:
            self._log_exploration(node, optimize_goal, explore, directive.name)
        

        message_length = len(messages)

        if directive_name == "jev_classify":
            fits, _, why = fits_jev(node.parsed_yaml["operations"], target_op_list[0])
            if not fits:
                return self._expand_jev_coupled(
                    node, target_op_list[0], optimize_goal, messages, message_condensed, message_length,
                    why, input_file_path, orig_default_model)
        
        # Check if this directive supports multiple instantiations
        is_multi_instance = directive in MULTI_INSTANCE_DIRECTIVES
        grid_spec = GRIDS.get(directive_name) if self.use_param_grid and len(target_op_list) == 1 else None
        if grid_spec is not None:
            is_multi_instance = False      # the grid replaces free-form variations
        num_instantiations = 2 if is_multi_instance else 1
        
        print(f"Creating {num_instantiations} instantiation(s) for directive '{directive_name}'")
        
        children = []
        instantiation_messages = messages.copy()
        if grid_spec is not None and grid_spec.sized:
            instantiation_messages.append({"role": "user", "content": grid_prompt(grid_spec)})
        elif directive_name == "jev_classify":     # to judge whether the input fits Jev's context
            instantiation_messages.append({"role": "user", "content": _param_grid.corpus_stats_text()})
        for i in range(num_instantiations):
            try:
                print(f"Creating instantiation {i+1}/{num_instantiations}")
                
                # For multi-instance directives, add variation to each instantiation
                if is_multi_instance:
                    # Use accumulated message history (includes previous instantiations)
                    # Add instantiation-specific context to encourage variation
                    if i == 0:
                        variation_prompt = f"This is instantiation {i+1} of {num_instantiations} for the '{directive_name}' directive. Focus on creating a distinct approach by exploring different parameter combinations or implementation strategies. For example, you can try different models, different parameter settings (chunk size, top k, etc.), or different implementation strategies."
                    else:
                        variation_prompt = f"This is instantiation {i+1} of {num_instantiations} for the '{directive_name}' directive. Based on the previous {i} instantiation(s) shown above, create a DIFFERENT approach that explores alternative parameters, strategies, or implementations. AVOID repeating the exact same configs asthe previous instantiations."
                    
                    
                    # Insert variation context before the last user message
                    variation_msg = {
                        "role": "system",
                        "content": variation_prompt
                    }
                    instantiation_messages.append(variation_msg)

                cost = 0.0
                with track_agent_cost() as spent, capture_applies() as apply_calls:
                    try:
                        new_ops_list, updated_message_history, cost = directive.instantiate(
                            operators=node.parsed_yaml["operations"],
                            target_ops=target_op_list,
                            agent_llm=self.model,
                            optimize_goal=optimize_goal,
                            global_default_model=orig_default_model,
                            message_history=instantiation_messages,
                            input_file_path=input_file_path,
                            pipeline_code=node.parsed_yaml,
                            dataset=self.dataset_name,
                        )
                    finally:
                        # every agent call counts, also those of an instantiation that failed
                        cost = max(spent.total, float(cost or 0.0))
                        with self.tree_lock:
                            print(f"💰 Adding agent instantiation cost: ${cost:.4f} (total before: ${self.total_search_cost:.4f})")
                            self._add_cost("agent", cost)
                if new_ops_list is None:
                    print(f"Instantiation {i+1} failed: no ops list returned")
                    continue
                bad_models = disallowed_models(new_ops_list)
                if bad_models:
                    raise ValueError(f"the rewrite uses models outside the allowed list: {bad_models}")

                instantiation_messages = updated_message_history

                # Create child node for this instantiation
                # For multi-instance directives, use parent_id-instantiation_num format
                custom_id = f"{node.get_id()}-{i+1}-{optimize_goal}" if is_multi_instance else None
                child = self.instantiate_node(
                    node, new_ops_list, directive_name, target_op_list, optimize_goal, 
                    message_condensed + updated_message_history[message_length:], custom_id
                )
                
                rewrite = find_rewrite(apply_calls[-1]) if apply_calls else None
                child.decomposition = getattr(rewrite, "decomposition", None)
                if grid_spec is not None and apply_calls:
                    children.extend(self._grid_children(
                        node, directive, grid_spec, apply_calls[-1], child, target_op_list, optimize_goal,
                        message_condensed + updated_message_history[message_length:]))
                else:
                    children.append(child)
                print(f"Instantiation {i+1} created successfully")

            except Exception as e:
                print(f"Instantiation {i+1} failed with error: {str(e)}")
                continue

        if not children:
            raise RuntimeError(f"All {num_instantiations} instantiation(s) failed for directive '{directive_name}'")
        
        return children

    def simulate(self, node: Node):
        """
        Simulate a node (plan). Execute the plan and evaluate it separately.

        Args:
            node: Node to start simulation from

        Returns:
            The accuracy and cost of the node
        """

        accuracy = float("-inf")
        cost = -1

        try:
            # Step 1: Execute the plan (this will set node.cost)
            node.execute_plan()
        except Exception as e:
            print(f"Failed to execute plan for node {node.get_id()}: {str(e)}")
            # Log -inf occurrence for debugging
            node._log_inf_occurrence("simulation_execution_failure", str(e), node.yaml_file_path)
            # Set cost to -1 to indicate failure (this is already done in Node.execute_plan)
            # Do not add failed plans to the frontier
            return cost, accuracy

        # Step 2: Evaluate the plan (this will call the evaluation function)
        try:
            accuracy = self.evaluate_node(node)
            cost = node.cost
            print(f"Node {node.get_id()} evaluation - cost: ${cost:.2f}, accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Failed to evaluate plan for node {node.get_id()}: {str(e)}")
            # Log -inf occurrence for debugging
            node._log_inf_occurrence("simulation_evaluation_failure", str(e), node.yaml_file_path)
            return cost, accuracy
        
        return cost, accuracy

    def add_to_frontier(self, node: Node, accuracy: float):
        # Step 3: Add to frontier (this only manages frontier state, no evaluation)
        affected_nodes, is_frontier_updated = self.pareto_frontier.add_plan_f1(node, accuracy)
        self.action_rewards = self.pareto_frontier.action_rewards
        return affected_nodes, is_frontier_updated

    def _block_if_unaffordable(self, leaf: Node, children: List[Node]) -> None:
        """Every child was skipped by the cost guard. If that holds even without other iterations'
        reservations, it holds for the rest of the search (the budget only shrinks): stop expanding
        this node, as for a node with no rewrite left. Its existing children stay selectable."""
        if self.max_budget_usd is None:
            return
        estimates = [getattr(c, "guard_estimate", None) for c in children]
        estimates = [e for e in estimates if e is not None]
        if not estimates:
            return
        remaining = self.max_budget_usd - self.total_search_cost
        allowance = min(remaining, self.cost_guard_fraction * remaining)
        if min(estimates) > allowance:
            leaf.no_actions = True
            print(f"node {leaf.get_id()}: children cost about ${min(estimates):.2f}, over the ${allowance:.2f} "
                  f"the cost guard allows now; it will not be expanded again")
            if all(getattr(n, "no_actions", False) for n in self._all_nodes()
                   if n is not self.root and not n.children and getattr(n, "cost", -1) != -1):
                self.action_space_exhausted = True

    def _undo_visits(self, node: Node, times: int) -> None:
        """Take back the visits counted at selection when the iteration evaluated no plan."""
        for _ in range(times):
            current = node
            while current is not None:
                current.visits = max(0, current.visits - 1)
                current = getattr(current, "parent", None)

    def increment_visits_up_tree(self, node: Node):
        """
        Increment visit count up the tree from the given node to root.
        Called immediately after expansion.
        """
        current = node
        while current is not None:
            current.update_visit()
            current = current.parent

    def backpropagate(self, affected_nodes: Dict[Node, int], visit_node):
        """
        Backpropagate only the simulation value changes up the tree.
        Visit counts are updated separately in increment_visits_up_tree.
        """
        for node, val in affected_nodes.items():
            current = node
            while current is not None:
                current.update_value(val)
                current = current.parent
        
        visit_node.update_visit()

    def _log_failure(self, error: str) -> None:
        """Append a failed iteration's error to failures.log in the output directory."""
        if not self.output_dir:
            return
        with open(os.path.join(self.output_dir, "failures.log"), "a") as f:
            f.write(f"--- failure {self.failed_iteration_count} "
                    f"(after {self.iteration_count} successful iterations) ---\n{error}\n")

    def should_continue(self) -> bool:
        """Check if MCTS should continue running."""
        if self.iteration_count >= self.max_iterations:
            return False

        if self.max_budget_usd is not None and self.total_search_cost >= self.max_budget_usd:
            print(f"Stopping: spent ${self.total_search_cost:.2f} of the ${self.max_budget_usd:.2f} budget")
            return False

        if self.consecutive_all_skipped >= self.max_consecutive_all_skipped:
            print(f"Stopping: the last {self.consecutive_all_skipped} iterations only produced children the cost "
                  f"guard skipped; ${self.max_budget_usd - self.total_search_cost:.2f} of the budget is left")
            return False

        # Early stopping: return False if action space is exhausted
        if self.action_space_exhausted:
            print("Early stopping: Action space exhausted, no more actions available")
            return False

        return True

    def get_frontier_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all plans in the Pareto frontier."""
        return self.pareto_frontier.get_all_plans_summary()
    
    def _is_cheaper_model(self, target_model: str, current_model: str) -> bool:
        """Check if target_model is cheaper than current_model for this dataset."""
        if self.dataset_name not in MODEL_COSTS:
            return False
        
        dataset_costs = MODEL_COSTS[self.dataset_name]
        # Remove the 'gemini/' prefix if present
        if current_model.startswith("gemini/"):
            current_model = current_model[len("gemini/") :]
        if target_model.startswith("gemini/"):
            target_model = target_model[len("gemini/") :]
            
        current_cost = dataset_costs.get(current_model)
        target_cost = dataset_costs.get(target_model)

        if current_cost is None or target_cost is None:
            return False
            
        return target_cost < current_cost

    def log_tree_to_file(self, iteration_num):
        """Append the current tree, one readable line per plan, to mcts_tree_log.txt."""
        events = read_jsonl(self.nodes_log_path) if os.path.exists(self.nodes_log_path) else []
        acc = self.pareto_frontier.plans_accuracy
        nodes = []
        for n in self._all_nodes():
            a = acc.get(n)
            nodes.append({
                "id": n.get_id(), "parent": n.parent.get_id() if n.parent is not None else None,
                "action": getattr(n.latest_action, "name", None),
                "targets": self._rewrite_targets(n) if n.parent is not None else [],
                "goal": getattr(n, "optimization_goal", None), "grid": getattr(n, "grid", None),
                "cost": n.cost if n.cost is not None and n.cost >= 0 else None,
                "acc": None if a in (None, float("-inf")) else a, "visits": n.visits,
            })
        new_ids = {e["id"] for e in events if e.get("iteration") == iteration_num - 1 and e.get("status") == "ok"}
        last, attempt = getattr(self, "_tree_log_last", (None, 0))
        attempt = attempt + 1 if last == iteration_num else 1
        self._tree_log_last = (iteration_num, attempt)
        again = f" (attempt {attempt}; earlier attempts added no plan)" if attempt > 1 else ""
        header = (f"\n=== After iteration {iteration_num}{again} · spent ${self.total_search_cost:.2f} "
                  f"(agent ${self.agent_cost:.2f}, pipelines ${self.pipeline_cost:.2f}) ===")
        stats = []
        for action in self.available_actions:
            runs = self.action_outcomes.get(action, [])
            if runs:
                stats.append(f"  {action.name}: {len(runs)} run(s), avg cost change "
                             f"{sum(c for c, _ in runs) / len(runs):+.2f}, avg score change "
                             f"{sum(d for _, d in runs) / len(runs):+.3f}")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(render_tree(nodes, events, header, new_ids))
            if stats:
                f.write("Rewrites that ran successfully:\n" + "\n".join(stats) + "\n")

    # ---- budget, grid, logs, checkpoint ------------------------------------------------------

    def _add_cost(self, kind: str, usd: float) -> None:
        usd = float(usd or 0.0)
        if usd <= 0:
            return
        with self.tree_lock:
            self.total_search_cost += usd
            if kind == "agent":
                self.agent_cost += usd
            else:
                self.pipeline_cost += usd
            append_jsonl(self.costs_log_path, {"kind": kind, "usd": usd, "iteration": self.iteration_count,
                                               "t": time.time()})

    def _op_model(self, node: Node, op_name: str) -> str:
        op = node.op_dict.get(op_name, {})
        model = op.get("model") or node.parsed_yaml.get("default_model") or DEFAULT_OPERATOR_MODEL
        return str(model).replace("gemini/", "")

    def _is_more_accurate_model(self, target_model: str, current_model: str) -> bool:
        stats = MODEL_STATS.get(self.dataset_name, {})
        target, current = target_model.replace("gemini/", ""), current_model.replace("gemini/", "")
        if target == current or target not in stats or current not in stats:
            return False
        return stats[target]["acc"] > stats[current]["acc"]

    def _is_expensive_model(self, model: str) -> bool:
        """At least 3x the cheapest first-layer model: run only proven grid values there."""
        costs = MODEL_COSTS.get(self.dataset_name, {})
        frontier = [costs[m] for m in FRONTIER_MODELS.get(self.dataset_name, ()) if m in costs]
        return bool(frontier) and costs.get(model, 0.0) >= 3 * min(frontier)

    # Operator types each rewrite may target (design doc, "Rewrites and grids"); unlisted: any.
    TARGET_TYPES = {
        "take_head_tail": {"filter", "map"},
        "doc_chunking_topk": {"filter", "map"},
        "jev_retrieval": {"filter", "map"},
        "doc_summarization": {"filter", "map"},
        "doc_chunking": {"map"},
        "isolating_subtasks": {"map"},
        "gleaning": {"filter", "map"},
        "presence_gate": {"map"},
        "cascade_filtering": {"filter"},
        "reduce_gleaning": {"reduce"},
        "reduce_chaining": {"reduce"},
        "map_reduce_fusion": {"reduce"},
        "swap_with_code": {"reduce"},
        "jev_classify": {"filter", "map"},
        "operator_fusion": {"filter", "map"},
        "jev_fusion": {"filter", "map"},
    }

    # Rewrites a Jev operator can still take: ones that change its input, its prompt, or the
    # whole pipeline. The rest assume an LLM (gleaning, model changes, generated outputs).
    JEV_OP_ALLOWED = {"take_head_tail", "doc_chunking_topk", "jev_retrieval", "doc_summarization",
                      "deterministic_doc_compression", "clarify_instructions", "arbitrary_rewrite",
                      "jev_fusion"}

    # Rewrites that take a pair of operators, and those that accept several; the rest take one.
    PAIR_TARGETS = {"operator_fusion", "map_reduce_fusion", "jev_fusion"}
    MULTI_TARGETS = {"doc_summarization", "deterministic_doc_compression", "change model"}

    def _target_error(self, node: Node, name: str, ops, offered: set) -> Optional[str]:
        """Why the agent's (directive, operators) choice cannot be instantiated, or None."""
        if not isinstance(ops, list) or not ops or any(not isinstance(o, str) for o in ops):
            return "operators must be a non-empty list of operator names"
        if len(set(ops)) != len(ops):
            return "an operator is listed twice"
        order = [o["name"] for o in node.parsed_yaml["operations"]]
        unknown = [o for o in ops if o not in order]
        if unknown:
            return f"no operator named {unknown}"
        if name in self.PAIR_TARGETS:
            if len(ops) != 2:
                return "this directive takes exactly two operators"
            i, j = order.index(ops[0]), order.index(ops[1])
            if j != i + 1:
                return "the two operators must be consecutive in the pipeline, in pipeline order"
            t1, t2 = (node.op_dict[o].get("type") for o in ops)
            if name == "operator_fusion" and not ({t1, t2} <= {"map", "filter", "parallel_map"}):
                return "operator_fusion fuses two filter/map operators (not a reduce or code operator)"
            if name == "map_reduce_fusion" and not (t1 == "map" and t2 == "reduce"):
                return "map_reduce_fusion takes a map followed by the reduce it feeds"
            if name == "jev_fusion":
                err = JevFusionDirective.check(node.parsed_yaml["operations"], ops)
                if err:
                    return err
            if not any((o, name) in offered for o in ops):
                return "not offered on these operators"
            return None
        if len(ops) > 1 and not any(name.startswith(m) for m in self.MULTI_TARGETS):
            return "this directive takes exactly one operator"
        missing = [o for o in ops if (o, name) not in offered]
        if missing:
            return f"not offered on {missing}"
        return None

    def _gleaning_exclusions(self, node: Node, op_name: str) -> set:
        """Rewrites this operator cannot take: ones meant for other operator types, and a
        second gleaning (at most one gleaning per operator, including from an ancestor)."""
        op = node.op_dict.get(op_name, {})
        excluded = {d for d in self.available_actions
                    if d.name in self.TARGET_TYPES and op.get("type") not in self.TARGET_TYPES[d.name]}
        if op.get("gleaning"):
            excluded |= {d for d in self.available_actions if d.name in ("gleaning", "reduce_gleaning")}
        if not op.get("prompt"):        # code and retrieval operators have no model to change
            excluded |= {d for d in self.available_actions if isinstance(d, ChangeModelCostDirective)}
        if is_jev_op(op):
            excluded |= {d for d in self.available_actions if d.name not in self.JEV_OP_ALLOWED}
        else:                           # jev_fusion merges two operations already on Jev
            excluded |= {d for d in self.available_actions if d.name == "jev_fusion"}
            if not jev_compatible(op):  # Jev answers only boolean and enum fields
                excluded |= {d for d in self.available_actions if d.name == "jev_classify"}
        return excluded

    def _grid_children(self, node, directive, spec, call, agent_child, target_op_list, optimize_goal, history):
        """One child per selected value of the rewrite's main parameter."""
        rewrite = find_rewrite(call)
        try:
            agent_value = spec.get(rewrite)
        except Exception as e:
            print(f"Grid: could not read {spec.param} from the rewrite ({e}); keeping the agent's child only")
            return [agent_child]
        op = target_op_list[0]
        parent_model = self._op_model(node, op)
        with self.tree_lock:
            values = self.param_table.choose(
                spec, op, parent_model, agent_value,
                model_cost_rank=MODEL_COSTS.get(self.dataset_name, {}),
                expensive_parent=self._is_expensive_model(parent_model))
        print(f"Grid {spec.directive}.{spec.param} on {op} under {parent_model}: agent chose {agent_value}, running {values}")
        children, reused = [], False
        for value in values:
            if not reused and value == agent_value:
                child, reused = agent_child, True
            else:
                try:
                    new_ops = reapply(directive, call, spec.set(rewrite, value))
                    child = self.instantiate_node(node, new_ops, directive.name, target_op_list, optimize_goal, history)
                except Exception as e:
                    print(f"Grid value {value} could not be applied: {e}")
                    continue
                child.decomposition = agent_child.decomposition
            child.grid = {"directive": spec.directive, "param": spec.param, "value": value, "op": op,
                          "parent_model": parent_model, "agent_value": agent_value}
            children.append(child)
        if not reused:
            with self.tree_lock:
                agent_child.delete()
        return children

    def _estimate_child_cost(self, child: Node) -> float:
        """Rough upper estimate of a child's pipeline cost from its parent's cost."""
        parent = child.parent
        base = parent.cost if parent is not None and parent.cost and parent.cost > 0 else 0.0
        action = getattr(child.latest_action, "name", "") or ""
        if isinstance(child.latest_action, ChangeModelCostDirective) and child.latest_action.target_model:
            costs = MODEL_COSTS.get(self.dataset_name, {})
            target = child.latest_action.target_model.replace("gemini/", "")
            ratios = [costs[target] / costs[self._op_model(parent, op)]
                      for op in parent.op_dict if costs.get(self._op_model(parent, op)) and target in costs]
            return base * max([1.0] + ratios)
        if action == "take_head_tail":
            cut = self._head_cut_estimate(child, base)
            if cut is not None:
                return cut
        adds_llm_work = {"chaining", "isolating_subtasks", "arbitrary_rewrite", "gleaning", "reduce_gleaning",
                         "doc_summarization", "doc_chunking", "operator_fusion", "map_reduce_fusion"}
        return base * (2.0 if action in adds_llm_work else 1.0)

    @staticmethod
    def _head_cut_estimate(child: Node, base: float) -> Optional[float]:
        """When no LLM operation of the child reads a full document field any more (every reader has
        its own head cut), its cost scales with the words kept: 2x the token share, as a margin.
        None when some operation still reads the full field."""
        ops = child.parsed_yaml.get("operations", [])
        cuts = [op for op in ops if op.get("type") == "code_map" and re.search(r"words\[:(\d+)\]", op.get("code") or "")]
        if not cuts:
            return None
        raw = set()
        for op in cuts:
            m = re.search(r"input_doc\.get\('([^']+)'", op.get("code") or "")
            if m:
                raw.add(m.group(1))
        for op in ops:
            if str(op.get("type", "")).startswith("code_"):
                continue
            text = json.dumps({k: v for k, v in op.items() if k not in ("name", "type", "model")})
            if any(re.search(r"\." + re.escape(k) + r"\b|\[['\"]" + re.escape(k) + r"['\"]\]", text) for k in raw):
                return None
        words = max(int(re.search(r"words\[:(\d+)\]", op["code"]).group(1)) for op in cuts)
        return 2.0 * estimate_child_cost(GRIDS["take_head_tail"], words, base, 1.0)

    def _cost_guard(self, child: Node) -> Optional[str]:
        """Skip a child whose estimated cost is too large for the budget left. A child that
        passes reserves its estimate until it finishes, so concurrent iterations cannot
        together spend past the cap."""
        if self.max_budget_usd is None:
            return None
        estimate = self._estimate_child_cost(child)
        with self.tree_lock:
            remaining = self.max_budget_usd - self.total_search_cost - self.reserved_cost
            if estimate > remaining or estimate > self.cost_guard_fraction * remaining:
                child.guard_estimate = estimate
            if estimate > remaining:
                return f"estimated ${estimate:.2f} is more than the remaining ${remaining:.2f}"
            if estimate > self.cost_guard_fraction * remaining:
                return (f"estimated ${estimate:.2f} is more than {self.cost_guard_fraction:.0%} of the "
                        f"remaining ${remaining:.2f}")
            self.reserved_cost += estimate
            child.reserved_cost = estimate
        return None

    def _release_reservation(self, child: Node) -> None:
        with self.tree_lock:
            self.reserved_cost = max(0.0, self.reserved_cost - getattr(child, "reserved_cost", 0.0))
            child.reserved_cost = 0.0

    def _record_result(self, child: Node, cost: float, accuracy: float) -> None:
        ok = cost is not None and cost != -1 and accuracy is not None and accuracy != float("-inf")
        grid = getattr(child, "grid", None)
        if grid:
            self.param_table.record(grid["directive"], grid["op"], grid["parent_model"], grid["value"],
                                    cost if ok else None, accuracy if ok else None, str(child.get_id()))
        if ok:
            with self.tree_lock:
                self._add_outcome(child)
        self._log_node(child, cost, accuracy, status="ok" if ok else "failed")

    def _add_outcome(self, child: Node) -> None:
        """Record a successful child's change vs its parent and refresh the per-rewrite stats."""
        parent, action = child.parent, child.latest_action
        acc = self.pareto_frontier.plans_accuracy
        if (action is None or parent is None or child not in acc or parent not in acc
                or child.cost is None or child.cost < 0 or parent.cost is None or parent.cost < 0):
            return
        self.action_outcomes.setdefault(action, []).append((child.cost - parent.cost, acc[child] - acc[parent]))
        self.action_counts = {a: float(len(self.action_outcomes.get(a, []))) for a in self.available_actions}
        self.action_cost_changes = {a: sum(c for c, _ in self.action_outcomes.get(a, []))
                                    for a in self.available_actions}
        self.action_accuracy_changes = {a: sum(d for _, d in self.action_outcomes.get(a, []))
                                        for a in self.available_actions}

    def _log_node(self, child: Node, cost, accuracy, status: str, reason: Optional[str] = None) -> None:
        parent = child.parent
        self._attempt_result(child, cost, accuracy, status, reason)
        append_jsonl(self.nodes_log_path, {
            "t": time.time(), "iteration": self.iteration_count, "id": child.get_id(),
            "parent": parent.get_id() if parent is not None else None,
            "directive": getattr(child.latest_action, "name", None),
            "target_ops": self._rewrite_targets(child),
            "goal": child.optimization_goal, "grid": getattr(child, "grid", None),
            "decomposition": getattr(child, "decomposition", None),
            "yaml": str(child.yaml_file_path), "status": status, "reason": reason,
            "cost": cost, "accuracy": None if accuracy in (None, float("-inf")) else accuracy,
            "spent_total": self.total_search_cost, "spent_agent": self.agent_cost,
            "spent_pipeline": self.pipeline_cost,
        })

    # ---- seeds from an earlier sweep -----------------------------------------------------------

    def _seed_first_layer(self) -> List[Node]:
        """Give each first-layer plan one child with the sweep's best data view (a head cut on every
        operation that reads the document field). The children are evaluated like any other child
        (cost guard, frontier, logs); the first-layer plans stay expandable."""
        from docetl.reasoning_optimizer.directives.take_head_tail import TakeHeadTailDirective
        from docetl.reasoning_optimizer.instantiate_schemas import TakeHeadTailInstantiateSchema
        seed = self.prior["seed"]
        if seed.get("directive", "take_head_tail") != "take_head_tail":
            raise ValueError(f"unsupported seed directive {seed.get('directive')!r}")
        words, key = int(seed["head_words"]), seed.get("document_key", "tot_text_raw")
        directive = self.directive_name_to_obj.get("take_head_tail") or TakeHeadTailDirective()
        seeds: List[Node] = []
        with self.tree_lock:
            for parent in [c for c in self.root.children if self.is_first_layer_node(c)]:
                ops, targets = deepcopy(parent.parsed_yaml["operations"]), []
                for name in [o["name"] for o in parent.parsed_yaml["operations"]]:
                    op = next(o for o in ops if o["name"] == name)
                    if str(op.get("type", "")).startswith("code_"):
                        continue
                    try:
                        ops = directive.apply(ops, name, TakeHeadTailInstantiateSchema(
                            name=f"head_{words}_{name}", document_key=key, head_words=words))
                        targets.append(name)
                    except ValueError:
                        continue        # does not read the field
                if not targets:
                    continue
                if "take_head_tail" not in self.directive_name_to_obj:
                    self.directive_name_to_obj["take_head_tail"] = directive
                child = self.instantiate_node(parent, ops, "take_head_tail", targets, "cost", [])
                child.seeded = True
                seeds.append(child)
        print(f"Seeding {len(seeds)} first-layer plans with a {words}-word head cut on {key}")

        def run(child):
            self._iteration_stats.executed = 0
            self._iteration_stats.skipped = 0
            self._simulate_children([child], "cost")

        with ThreadPoolExecutor(max_workers=max(1, self.max_concurrent_agents)) as pool:
            list(pool.map(run, seeds))
        with self.tree_lock:
            self.log_tree_to_file(0)
            self._checkpoint()
        return seeds

    # ---- exploration turns: offer only the least-chosen rewrites --------------------------------

    def _choice_counts(self) -> Dict[str, int]:
        """Times the agent chose each rewrite family in this search (model switches count as one).
        A choice whose children were all skipped by the cost guard never ran, so it does not count."""
        with self.tree_lock:
            return dict(Counter(
                rewrite_family(a["directive"]) for a in getattr(self, "attempts", [])
                if not (a.get("results") and all(r.get("status") == "skipped" for r in a["results"]))))

    def _exploration_options(self, action_options: List[Tuple[str, str]]) -> Optional[List[Tuple[str, str]]]:
        """On every explore_every-th agent decision, the options restricted to the explore_k
        least-chosen rewrite families (all families tied with the k-th are kept). None on a normal
        turn, or when the restriction would not remove anything."""
        every = getattr(self, "explore_every", 0)
        with self.tree_lock:
            decisions = len(getattr(self, "attempts", []))
        if not every or (decisions + 1) % every:
            return None
        counts = Counter(self._choice_counts())
        counts.update(getattr(self, "prior_counts", {}))
        counts.update(getattr(self, "declined", {}))
        families = sorted({rewrite_family(n) for _, n in action_options}, key=lambda f: (counts.get(f, 0), f))
        k = max(1, getattr(self, "explore_k", 3))
        if len(families) <= k:
            return None
        cutoff = counts.get(families[k - 1], 0)
        keep = {f for f in families if counts.get(f, 0) <= cutoff}
        if len(keep) == len(families):
            return None
        return [(o, n) for o, n in action_options if rewrite_family(n) in keep]

    def _load_declines(self) -> Counter:
        """Rebuild the decline counts from exploration.jsonl (rows with opted_out set)."""
        declined: Counter = Counter()
        path = Path(self.output_dir) / "exploration.jsonl" if self.output_dir else None
        if path and path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line) if line.strip() else {}
                if row.get("opted_out") is not None and not row.get("chosen"):
                    declined.update(set(row.get("offered") or {}))
        return declined

    def _evidence_text(self) -> str:
        text = (getattr(self, "prior", None) or {}).get("evidence_text")
        return f"\n\n{text.strip()}\n" if text else ""

    def _log_exploration(self, node: Node, goal: str, offered, chosen: Optional[str], opt_out: Optional[str] = None):
        counts = Counter(self._choice_counts())
        if chosen is not None:          # already registered; report counts as they were when offered
            counts[rewrite_family(chosen)] -= 1
        row = {"iteration": self.iteration_count, "node": node.get_id(), "goal": goal,
               "offered": {f: counts.get(f, 0) for f in sorted({rewrite_family(n) for _, n in offered})},
               "chosen": chosen, "opted_out": opt_out,
               "earlier_sweep": {f: self.prior_counts[f] for f in sorted({rewrite_family(n) for _, n in offered})
                                 if f in getattr(self, "prior_counts", {})}}
        print(f"Exploration turn on #{node.get_id()}: {row}")
        if self.output_dir:
            with self.tree_lock:
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
                with open(Path(self.output_dir) / "exploration.jsonl", "a") as stream:
                    stream.write(json.dumps(row) + "\n")

    # ---- rewrites tried on the same operations in other plans ---------------------------------

    def _restore_attempts(self, nodes: Dict[Any, Node], records: Dict[Any, Dict]) -> None:
        """Rebuild the attempt list from nodes.jsonl (one attempt per parent, rewrite and iteration)."""
        groups: Dict[Tuple, Dict[str, Any]] = {}
        for row in read_jsonl(self.nodes_log_path):
            if not row.get("directive"):
                continue
            key = (row.get("parent"), row["directive"], row.get("iteration"))
            a = groups.setdefault(key, {"parent": row.get("parent"), "directive": row["directive"],
                                        "targets": list(row.get("target_ops") or []), "goal": row.get("goal"),
                                        "iteration": row.get("iteration"), "results": [], "closed": None})
            r: Dict[str, Any] = {"node": row.get("id"), "status": row.get("status"),
                                 "grid": self._grid_label(row.get("grid"))}
            parent = records.get(row.get("parent"))
            if row.get("status") in ("ok", "pruned") and parent and parent.get("accuracy") is not None \
                    and (parent.get("cost") or -1) >= 0 and row.get("accuracy") is not None:
                r["dacc"], r["dcost"] = row["accuracy"] - parent["accuracy"], row["cost"] - parent["cost"]
            elif row.get("status") in ("ok", "pruned"):
                r["acc"], r["cost"] = row.get("accuracy"), row.get("cost")
            if row.get("status") == "skipped":
                reason = row.get("reason") or ""
                r["why"] = "cost guard" if reason.startswith("estimated $") else reason[:60]
            a["results"].append(r)
        self.attempts = list(groups.values())

    @staticmethod
    def _grid_label(grid: Optional[Dict]) -> Optional[str]:
        if not grid or grid.get("value") is None:
            return None
        return f"{grid.get('param')}={grid['value']}" if grid.get("param") else str(grid["value"])

    def _expand_tracked(self, leaf: Node, goal: str) -> List[Node]:
        if getattr(self, "_attempt_local", None) is None:
            self._attempt_local = threading.local()
        self._attempt_local.made = []
        try:
            return self.expand(leaf, optimize_goal=goal)
        except Exception:
            for a in self._attempt_local.made:
                if not a["results"]:
                    a["closed"] = "produced no plan (instantiation failed)"
            raise

    def _register_attempt(self, node: Node, directive: str, targets: List[str], goal: str) -> None:
        a = {"parent": node.get_id(), "directive": directive, "targets": list(targets), "goal": goal,
             "iteration": self.iteration_count, "results": [], "closed": None}
        with self.tree_lock:
            self.__dict__.setdefault("attempts", []).append(a)
        made = getattr(getattr(self, "_attempt_local", None), "made", None)
        if made is not None:
            made.append(a)

    def _attempt_result(self, child: Node, cost, accuracy, status: str, reason: Optional[str]) -> None:
        name = getattr(child.latest_action, "name", None)
        parents = {p.get_id() for p in (child.parent, getattr(child.parent, "parent", None)) if p is not None}
        targets = set(self._rewrite_targets(child))
        with self.tree_lock:
            for a in reversed(getattr(self, "attempts", [])):
                if a["directive"] == name and a["parent"] in parents and (targets & set(a["targets"]) or not targets):
                    break
            else:
                return
            r: Dict[str, Any] = {"node": child.get_id(), "status": status,
                                 "grid": self._grid_label(getattr(child, "grid", None))}
            acc = self.pareto_frontier.plans_accuracy
            p = child.parent
            if status in ("ok", "pruned") and p is not None and p in acc and accuracy is not None \
                    and p.cost is not None and p.cost >= 0 and cost is not None:
                r["dacc"], r["dcost"] = accuracy - acc[p], cost - p.cost
            elif status in ("ok", "pruned"):
                r["acc"], r["cost"] = accuracy, cost
            if status == "skipped":
                r["why"] = "cost guard" if (reason or "").startswith("estimated $") else (reason or "")[:60]
            a["results"].append(r)

    def _tried_elsewhere_text(self, node: Node, limit: int = 25) -> str:
        ops = set(node.op_dict.keys()) if getattr(node, "op_dict", None) else set()
        lines = []
        with self.tree_lock:
            attempts = [a for a in getattr(self, "attempts", []) if a["parent"] != node.get_id() and ops & set(a["targets"])]
        for a in reversed(attempts):
            if a["results"]:
                parts = []
                for r in a["results"]:
                    g = f"{r['grid']}: " if r.get("grid") is not None else ""
                    if r["status"] == "pruned":
                        g += "(dropped: a sibling value was as good and cheaper) "
                    if "dacc" in r:
                        sign = "-" if r["dcost"] < -0.005 else "+"
                        parts.append(f"{g}score {r['dacc']:+.3f}, cost {sign}${abs(r['dcost']):.2f}")
                    elif r["status"] in ("ok", "pruned"):
                        parts.append(f"{g}score {r.get('acc', 0):.3f}, cost ${r.get('cost') or 0:.2f}")
                    elif r["status"] == "skipped":
                        parts.append(f"{g}not run ({r.get('why')})")
                    else:
                        parts.append(f"{g}failed when run")
                outcome = "; ".join(parts)
            else:
                outcome = a["closed"] or "in progress, no result yet"
            lines.append(f"- {a['directive']} on {', '.join(a['targets'])} (plan #{a['parent']}, {a['goal']} goal): "
                         f"{outcome}")
            if len(lines) >= limit:
                break
        if not lines:
            return ""
        return ("\n\nRewrites already chosen for these operations in other plans of this search (newest first; "
                "score and cost changes are relative to the plan they were applied to):\n" + "\n".join(lines)
                + "\nOther plans may use different models or earlier rewrites, so results can differ here. Prefer "
                "a rewrite that has not been tried on this operation, or one that worked elsewhere; repeat one that "
                "did not help, or that is still in progress, only if you expect a different outcome on this plan.\n")

    @staticmethod
    def _rewrite_targets(child: Node) -> List[str]:
        """Operations the child's own rewrite targeted (its memo entries beyond the parent's)."""
        memo = list(getattr(child, "memo", None) or [])
        start = len(getattr(child.parent, "memo", None) or []) if child.parent is not None else 0
        ops = [op for _, op in memo[start:]] or [op for _, op in memo[-1:]]
        return list(dict.fromkeys(ops))

    def _all_nodes(self) -> List[Node]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children)
        return out

    def _checkpoint(self) -> None:
        """Write the full search state; called under tree_lock after every iteration."""
        with self.tree_lock:
            order = [n for n in self.pareto_frontier.plans_accuracy.keys()]
            evaluated = set(order)
            nodes = [n for n in self._all_nodes() if n is self.root or n in evaluated]
            state = {
                "version": 1,
                "root_id": self.root.id,
                "frontier_order": [n.id for n in order],
                "nodes": [node_record(n, self.pareto_frontier.plans_accuracy.get(n)) for n in nodes],
                "iteration_count": self.iteration_count,
                "failed_iteration_count": self.failed_iteration_count,
                "action_counts": {a.name: c for a, c in self.action_counts.items()},
                "action_space_exhausted": self.action_space_exhausted,
                "next_node_id": Node._id_counter,
            }
            write_checkpoint(self.checkpoint_path, state)
        if self.checkpoint_hook is not None:
            try:
                self.checkpoint_hook()
            except Exception as e:
                print(f"⚠️ checkpoint hook failed: {e}")

    def _restore_checkpoint(self) -> None:
        state = read_checkpoint(self.checkpoint_path)
        by_name = self.directive_name_to_obj
        nodes: Dict[Any, Node] = {}
        records = {r["id"]: r for r in state["nodes"]}

        def apply_record(node: Node, r: dict) -> None:
            node.id = r["id"]
            node.cost = r["cost"] if r["cost"] is not None else -1
            node.value = r["value"] or 0
            node.visits = r["visits"]
            node.latest_action = by_name.get(r["latest_action"]) if r["latest_action"] else None
            node.optimization_goal = r["goal"]
            node.memo = [tuple(m) for m in r["memo"]]
            node.sample_result = r["sample_result"]
            for op, names in r["used_actions"].items():
                node.used_actions.setdefault(op, set()).update(by_name[n] for n in names if n in by_name)
            if r.get("grid"):
                node.grid = r["grid"]

        apply_record(self.root, records[state["root_id"]])
        nodes[state["root_id"]] = self.root
        pending = [r for r in state["nodes"] if r["id"] != state["root_id"]]
        while pending:
            progressed = False
            for r in list(pending):
                parent = nodes.get(r["parent"])
                if parent is None:
                    continue
                node = Node(r["yaml"], parent=parent, message_history=r["message_history"], id=r["id"] or None)
                sanitize_pipeline(node.parsed_yaml, self.dataset_name, self.output_dir)
                apply_record(node, r)
                parent.add_child(node)
                nodes[r["id"]] = node
                pending.remove(r)
                progressed = True
            if not progressed:
                raise RuntimeError(f"checkpoint has nodes whose parent is missing: {[r['id'] for r in pending]}")

        # Replay the frontier in the original order, then restore the search values it changed.
        for node_id in state["frontier_order"]:
            node = nodes[node_id]
            acc = records[node_id]["accuracy"]
            if node.cost == -1 or acc is None:
                self.pareto_frontier.plans_accuracy[node] = float("-inf")
            else:
                self.pareto_frontier.add_plan_f1(node, acc)
        self.action_rewards = self.pareto_frontier.action_rewards
        for node_id, node in nodes.items():
            node.value = records[node_id]["value"] or 0
            node.visits = records[node_id]["visits"]
            node.no_actions = bool(records[node_id].get("no_actions", False))
            node.exhausted_goals = set(records[node_id].get("exhausted_goals") or [])
        for node in nodes.values():
            self._add_outcome(node)
        self._restore_attempts(nodes, records)
        self.iteration_count = state["iteration_count"]
        self.failed_iteration_count = state["failed_iteration_count"]
        self.action_space_exhausted = state["action_space_exhausted"]
        # Nodes created after the last checkpoint (in flight at the crash) already used ids and
        # wrote files; start above every id seen so nothing is overwritten.
        seen = [state["next_node_id"]]
        seen += [r["id"] + 1 for r in read_jsonl(self.nodes_log_path) if isinstance(r.get("id"), int)]
        for f in os.listdir(self.output_dir or "."):
            stem = f.rsplit(".", 1)[0]
            tail = stem.rsplit("_", 1)[-1]
            if f.endswith(".yaml") and tail.isdigit():
                seen.append(int(tail) + 1)
        Node._id_counter = max([Node._id_counter] + seen)
        totals = ledger_totals(self.costs_log_path)     # includes work lost in the crash
        self.agent_cost, self.pipeline_cost = totals.get("agent", 0.0), totals.get("pipeline", 0.0)
        self.total_search_cost = self.agent_cost + self.pipeline_cost
        self.declined = self._load_declines()
        self.resumed = True
        print(f"🔁 Resumed from {self.checkpoint_path}: {len(nodes)} nodes, iteration {self.iteration_count}, "
              f"spent ${self.total_search_cost:.2f}")

    def _instantiate_child(self, node, directive, target_op_list, optimize_goal, messages, message_condensed,
                           message_length, input_file_path, orig_default_model, extra=None) -> Node:
        """One instantiation of directive on node (no grid fan-out), with every agent call charged."""
        inst_messages = messages.copy() + ([{"role": "user", "content": extra}] if extra else [])
        cost = 0.0
        new_ops_list = None
        with track_agent_cost() as spent, capture_applies() as apply_calls:
            try:
                new_ops_list, history, cost = directive.instantiate(
                    operators=node.parsed_yaml["operations"], target_ops=target_op_list, agent_llm=self.model,
                    optimize_goal=optimize_goal, global_default_model=orig_default_model,
                    message_history=inst_messages, input_file_path=input_file_path,
                    pipeline_code=node.parsed_yaml, dataset=self.dataset_name)
            finally:
                cost = max(spent.total, float(cost or 0.0))
                with self.tree_lock:
                    self._add_cost("agent", cost)
        if new_ops_list is None:
            raise ValueError("no ops list returned")
        bad_models = disallowed_models(new_ops_list)
        if bad_models:
            raise ValueError(f"the rewrite uses models outside the allowed list: {bad_models}")
        child = self.instantiate_node(node, new_ops_list, directive.name, target_op_list, optimize_goal,
                                      message_condensed + history[message_length:])
        rewrite = find_rewrite(apply_calls[-1]) if apply_calls else None
        child.decomposition = getattr(rewrite, "decomposition", None)
        return child

    def _expand_jev_coupled(self, node, op, optimize_goal, messages, message_condensed, message_length,
                            why, input_file_path, orig_default_model) -> List[Node]:
        """jev_classify chosen for an operation whose input can exceed Jev's context: the agent first
        picks a rewrite that shortens the input; one iteration creates that child S under node and the
        Jev child J under S. Both are simulated (S first); J's reward backpropagates through S."""
        jev = self.directive_name_to_obj["jev_classify"]
        excluded = get_excluded_directives_for_operation(node, op) | self._gleaning_exclusions(node, op)
        used = node.used_actions.get(op, set())
        options = [d for d in self.available_actions
                   if d.name in SHORTENING_FOR_JEV and d not in excluded and d not in used]
        if not options:
            raise RuntimeError(f"jev_classify on {op}: the input is too long for Jev ({why}) and no shortening "
                               f"rewrite is left for {op}")
        names = [d.name for d in options]
        ask = (
            f"Jev cannot read the input of {op} as the pipeline stands: {why}. Jev's limit is 32K tokens per call "
            f"and it does not truncate. Before {op} is switched to Jev, its input must be shortened by another "
            f"rewrite; this iteration applies that rewrite first and then the Jev switch on top of it.\n\n"
            "Choose the shortening rewrite (it must leave the evidence the decision needs):\n"
            + "\n".join(f"- {d.to_string_for_plan()}" for d in options)
            + "\n\nNotes: take_head_tail keeps the start (and end) of the document; jev_retrieval keeps passages "
            "relevant to the task; doc_summarization replaces the document with a summary; doc_chunking (maps "
            "only) runs the map per chunk, and Jev is then applied to the per-chunk map, so use no 'middle' "
            "peripheral section (it would add every earlier chunk to each call).\n"
            f"Answer with the directive name, one of {names}, and a one-sentence reason."
        )
        msgs = messages + [{"role": "user", "content": ask}]
        choice = None
        for _ in range(3):
            response, call_cost = agent_completion(model=self.model, messages=msgs,
                                                   response_format=ShorteningChoice)
            with self.tree_lock:
                self._add_cost("agent", call_cost)
            content = response.choices[0].message.content
            try:
                picked = json.loads(content).get("directive")
            except Exception:
                picked = None
            msgs.append({"role": "assistant", "content": content})
            if picked in names:
                choice = options[names.index(picked)]
                break
            msgs.append({"role": "user", "content": f"{picked!r} is not one of {names}. Pick one of them."})
        if choice is None:
            raise RuntimeError(f"jev_classify on {op}: no valid shortening rewrite chosen")
        node.mark_action_used(op, choice)
        self._register_attempt(node, choice.name, [op], optimize_goal)
        print(f"Jev coupling on {op}: shortening with {choice.name} first ({why})")

        extra = None
        spec = GRIDS.get(choice.name)
        if spec is not None and spec.sized:
            extra = _param_grid.corpus_stats_text() + "\n\nThe shortened input must fit in Jev's 32K-token context (prompt included)."
        shortened = self._instantiate_child(node, choice, [op], optimize_goal, msgs, message_condensed,
                                            message_length, input_file_path, orig_default_model, extra)
        shortened.coupled = True
        children = [shortened]

        ops = shortened.parsed_yaml["operations"]
        candidates = [o["name"] for o in ops if jev_compatible(o) and (o["name"] == op or op in o["name"])]
        if not candidates:
            print(f"Jev coupling: after {choice.name} no filter/map with boolean/enum outputs is left for {op}")
            return children
        jev_op = candidates[0]
        fits, bound, why2 = fits_jev(ops, jev_op)
        if not fits:
            print(f"Jev coupling: after {choice.name} the input of {jev_op} still does not fit ({why2}); "
                  f"keeping only the shortened child")
            return children
        try:
            note = (f"The input of {jev_op} has been shortened by {choice.name}: {why2}. It fits Jev's context.\n\n"
                    + _param_grid.corpus_stats_text())
            jev_child = self._instantiate_child(shortened, jev, [jev_op], optimize_goal, msgs, message_condensed,
                                                message_length, input_file_path, orig_default_model, note)
            shortened.mark_action_used(jev_op, jev)
            jev_child.coupled = True
            children.append(jev_child)
        except Exception as e:
            print(f"Jev coupling: the Jev switch on {jev_op} failed ({e}); keeping the shortened child")
        return children

    def instantiate_node(
        self, node, new_ops_list, directive_name, target_op_list, optimize_goal, message_condensed, custom_id=None
    ):
        """
        Instantiate a new child node by applying the directive to the given node and target operations.
        Args:
            node: The parent node
            directive: The directive object to apply
            directive_name: The name of the directive
            target_op_list: List of target operations
            optimize_goal: The optimization goal (e.g., 'acc' or 'cost')
        Returns:
            The newly created child node
        """

        new_parsed_yaml = deepcopy(node.parsed_yaml)
        sanitize_pipeline(new_parsed_yaml, self.dataset_name, self.output_dir)
        new_parsed_yaml["operations"] = new_ops_list
        new_parsed_yaml["bypass_cache"] = True
        new_parsed_yaml = update_pipeline(
            new_parsed_yaml, new_ops_list, target_op_list
        )

        fix_models(new_parsed_yaml)

        # Determine the node ID to use for filename
        if custom_id is not None:
            node_id_for_file = custom_id
        else:
            node_id_for_file = Node.get_next_id()
        
        # Determine where to save the new pipeline file
        if self.output_dir:
            # Use output directory with original filename as base (strip existing node IDs)
            original_filename = os.path.basename(str(node.yaml_file_path)).removesuffix(".yaml")
            # Remove any existing node ID suffix 
            if "_" in original_filename:
                # Split and take only the first part before any underscores with numbers
                parts = original_filename.split("_")
                base_name = parts[0]
                # Check if subsequent parts are node IDs (numbers or multi-instance format)
                for part in parts[1:]:
                    if not (part.isdigit() or "-" in part):
                        base_name += "_" + part
            else:
                base_name = original_filename
            base_path = os.path.join(self.output_dir, base_name)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            # Use same directory as original pipeline
            base_path = str(node.yaml_file_path).removesuffix(".yaml")


        new_yaml_path = f"{base_path}_{node_id_for_file}.yaml"
        new_parsed_yaml["pipeline"]["output"][
            "path"
        ] = f"{base_path}_{node_id_for_file}.json"

        with open(new_yaml_path, "w") as file:
            yaml.dump(
                new_parsed_yaml,
                file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        print("NEW YAML FILE: ", new_yaml_path)

        # generate the child node
        child = Node(yaml_file_path=new_yaml_path, parent=node, message_history=message_condensed, id=custom_id)
        action = self.directive_name_to_obj.get(directive_name)
        child.latest_action = action
        child.optimization_goal = optimize_goal

        # Copy parent's memo and add new entries for this action
        child.memo = node.memo.copy()
        for target_op in target_op_list:
            child.add_memo_entry(directive_name, target_op)
            
        print("last action: ", child.latest_action.name, "child.id: ", child.id)
        
        if directive_name == "gleaning":
            for name in ("chaining", "doc_chunking", "gleaning"):
                blocked = self.directive_name_to_obj.get(name)
                if blocked is None:
                    continue        # not in this search's action set
                for op in target_op_list:
                    child.mark_action_used(op, blocked)

        elif directive_name == "change model":
            change_model = self.directive_name_to_obj.get("change model")
            assert change_model
            for op in target_op_list:
                child.mark_action_used(op, change_model)

        elif directive_name == "doc_chunking":
            doc_chunking = self.directive_name_to_obj.get("doc_chunking")
            assert doc_chunking
            for op in child.parsed_yaml["operations"]:
                op_name = op["name"]
                child.mark_action_used(op_name, doc_chunking)

        elif directive_name == "deterministic_doc_compression":
            d_comp = self.directive_name_to_obj.get("deterministic_doc_compression")
            assert d_comp
            for op in child.parsed_yaml["operations"]:
                op_name = op["name"]
                child.mark_action_used(op_name, d_comp)

        elif directive_name == "reduce_chaining":
            reduce_chain = self.directive_name_to_obj.get("reduce_chaining")
            assert reduce_chain
            for op in child.parsed_yaml["operations"]:
                op_name = op["name"]
                child.mark_action_used(op_name, reduce_chain)

        # Check if this was a compression directive and mark newly generated operators
        # to exclude all compression directives from their action space
        compression_directive_names = [d.name for d in DIRECTIVE_GROUPS.get("compression", [])]
        if directive_name in compression_directive_names:
            # Find newly created operators by comparing old and new operation lists
            old_op_names = {op["name"] for op in node.parsed_yaml["operations"]}
            new_op_names = {op["name"] for op in new_ops_list}
            newly_created_ops = new_op_names - old_op_names
            print("newly_created_ops: ", newly_created_ops)
            
            # Mark all compression directives as "used" for newly created operators
            # This prevents compression directives from being applied to operators
            # that were themselves generated by compression directives
            compression_directives = DIRECTIVE_GROUPS.get("compression", [])
            for new_op_name in newly_created_ops:
                for compression_directive in compression_directives:
                    child.mark_action_used(new_op_name, compression_directive)
                print(f"Marked compression directives as excluded for newly created operator: {new_op_name}")

        node.add_child(child)
        return child
