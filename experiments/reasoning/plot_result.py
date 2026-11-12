import json
import numpy as np
import matplotlib.pyplot as plt
import os

dataset_metrics = {
    "cuad": "f1",
    "blackvault": "avg_distinct_locations",
    "game_reviews": "weighted_score",
    "medec": "combined_score",
    "sustainability": "combined_score",
    "biodex": "avg_rp_at_5",
}


def find_pareto_frontier(file, exp_name):
    """
    Find Pareto frontier points from a JSON file.
    
    Args:
        file: Path to the JSON file containing evaluation results
        exp_name: Name of the experiment (e.g., "medec", "cuad", etc.)
    
    Returns:
        List of tuples: (iteration_number, accuracy, cost)
    """
    # Get the metric for this experiment
    if exp_name not in dataset_metrics:
        print(f"Unknown experiment: {exp_name}")
        return []
    
    accuracy_metric = dataset_metrics[exp_name]
    print(f"Using metric: {accuracy_metric}")
    
    # Load data from the file
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} data points from {file}")
    except FileNotFoundError:
        print(f"File not found: {file}")
        return []
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file}")
        return []
    
    # Filter out entries that don't have the required metrics
    valid_data = [item for item in data if accuracy_metric in item and "cost" in item]
    
    if not valid_data:
        return []
    
    print("valid_data length: ", len(valid_data))
    
    # Sort by cost (ascending) and accuracy (descending)
    sorted_data = sorted(valid_data, key=lambda x: (x["cost"], -x[accuracy_metric]))
    
    pareto_points = []
    best_accuracy = float('-inf')

    # if "on_frontier" in sorted_data[0]:
    #     for item in sorted_data:
    #         if item["on_frontier"]:
    #             pareto_points.append((item["node_id"], item[accuracy_metric], item["cost"]))
    # else:
    for item in sorted_data:
        current_accuracy = item[accuracy_metric]
        current_cost = item["cost"]
        
        # Check if this point dominates previous points
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            
            # Extract iteration number from filename
            iteration = item.get("node_id", "")
            # if iteration.endswith("_results"):
            #     iteration = iteration[:-8]
            # if iteration.startswith("iteration_"):
            #     iteration = iteration[10:]
            
            pareto_points.append((iteration, current_accuracy, current_cost))
        
    return pareto_points


def plot_pareto_frontier(file, exp_name, pareto_points, output_path=None):
    """
    Plot Pareto frontier points from evaluation results.
    
    Args:
        file: Path to the JSON file containing evaluation results
        exp_name: Name of the experiment
        pareto_points: List of Pareto frontier points (iteration, accuracy, cost)
        output_path: Directory to save the plot (optional)
    
    Returns:
        None
    """
    # Get the metric for this experiment
    if exp_name not in dataset_metrics:
        print(f"Unknown experiment: {exp_name}")
        return
    
    accuracy_metric = dataset_metrics[exp_name]
    
    # Load data from the file
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} data points from {file}")
    except FileNotFoundError:
        print(f"File not found: {file}")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file}")
        return
    
    # Filter out entries that don't have the required metrics
    valid_data = [item for item in data if accuracy_metric in item and "cost" in item]
    
    if not valid_data:
        return
    
    if not pareto_points:
        print("No Pareto frontier points provided.")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot only Pareto frontier points with black markers
    pareto_costs = [point[2] for point in pareto_points]
    pareto_accuracies = [point[1] for point in pareto_points]
    plt.scatter(pareto_costs, pareto_accuracies, c="black", s=100, alpha=0.8)
    
    # Add annotations for Pareto frontier points with full iteration labels
    for iteration, accuracy, cost in pareto_points:
        if iteration == "original":
            label = "original"
        else:
            label = f"iteration_{iteration}"
        plt.annotate(label, (cost, accuracy), 
                    textcoords="offset points", xytext=(5, 5), 
                    fontsize=10, fontweight="bold", color="black")
    
    # Add labels and title
    plt.xlabel("Cost ($)", fontsize=12)
    plt.ylabel(f"{accuracy_metric.replace('_', ' ').title()}", fontsize=12)
    plt.title(f"Cost vs {accuracy_metric.replace('_', ' ').title()} - {exp_name.upper()} Dataset\nPareto Frontier Points", fontsize=14)
    
    # # Set log scale for cost axis
    # plt.xscale('log')
    
    # Add grid
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Save the plot if output path is provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plot_filename = f"pareto_frontier_{exp_name}.png"
        plot_path = os.path.join(output_path, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"📈 Pareto frontier plot saved to: {plot_path}")
    
    plt.show()


def plot_pareto_frontier_comparison(file_baseline, file_mcts, file_simple, file_abacus, exp_name, pareto_points_baseline, pareto_points_mcts, pareto_points_simple, pareto_points_abacus=None, output_path=None):
    """
    Plot Pareto frontier points from baseline, MCTS, simple baseline, and optionally Abacus on a single graph.
    
    Args:
        file_baseline: Path to the baseline evaluation file
        file_mcts: Path to the MCTS evaluation file
        file_simple: Path to the simple baseline evaluation file
        file_abacus: Path to the Abacus evaluation file (optional, can be None)
        exp_name: Name of the experiment
        pareto_points_baseline: List of baseline Pareto frontier points
        pareto_points_mcts: List of MCTS Pareto frontier points
        pareto_points_simple: List of simple baseline Pareto frontier points
        pareto_points_abacus: List of Abacus Pareto frontier points (optional, can be None)
        output_path: Directory to save the plot (optional)
    
    Returns:
        None
    """
    # Get the metric for this experiment
    if exp_name not in dataset_metrics:
        print(f"Unknown experiment: {exp_name}")
        return
    
    accuracy_metric = dataset_metrics[exp_name]
    
    # Load baseline data
    try:
        with open(file_baseline, 'r') as f:
            data_baseline = json.load(f)
        print(f"Loaded {len(data_baseline)} baseline data points")
    except FileNotFoundError:
        print(f"Baseline file not found: {file_baseline}")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in baseline file: {file_baseline}")
        return
    
    # Load MCTS data
    try:
        with open(file_mcts, 'r') as f:
            data_mcts = json.load(f)
        print(f"Loaded {len(data_mcts)} MCTS data points")
    except FileNotFoundError:
        print(f"MCTS file not found: {file_mcts}")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in MCTS file: {file_mcts}")
        return
    
    # Load simple baseline data
    try:
        with open(file_simple, 'r') as f:
            data_simple = json.load(f)
        print(f"Loaded {len(data_simple)} simple baseline data points")
    except FileNotFoundError:
        print(f"Simple baseline file not found: {file_simple}")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in simple baseline file: {file_simple}")
        return
    
    # Load Abacus data (optional)
    data_abacus = []
    if file_abacus:
        try:
            with open(file_abacus, 'r') as f:
                data_abacus = json.load(f)
            print(f"Loaded {len(data_abacus)} Abacus data points")
        except FileNotFoundError:
            print(f"Abacus file not found: {file_abacus}")
            data_abacus = []
        except json.JSONDecodeError:
            print(f"Invalid JSON in Abacus file: {file_abacus}")
            data_abacus = []
    
    # Filter data for required metrics
    valid_baseline = [item for item in data_baseline if accuracy_metric in item and "cost" in item]
    valid_mcts = [item for item in data_mcts if accuracy_metric in item and "cost" in item]
    valid_simple = [item for item in data_simple if accuracy_metric in item and "cost" in item]
    valid_abacus = [item for item in data_abacus if accuracy_metric in item and "cost" in item]
    
    if not valid_baseline or not valid_mcts or not valid_simple:
        print("No valid data found in baseline, MCTS, or simple baseline files.")
        return
    
    # Find original point in baseline data
    original_point = None
    for item in data_baseline:
        if item.get("file") == "original_output.json":
            original_point = item
            break
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Log what's being plotted for each approach
    print(f"Plotting Baseline: {len(pareto_points_baseline) if pareto_points_baseline else 0} Pareto frontier points")
    print(f"Plotting MCTS: {len(pareto_points_mcts) if pareto_points_mcts else 0} Pareto frontier points")
    print(f"Plotting Simple Baseline: {len(pareto_points_simple) if pareto_points_simple else 0} Pareto frontier points")
    print(f"Plotting Abacus: {len(pareto_points_abacus) if pareto_points_abacus else 0} Pareto frontier points")
    
    # Plot simple baseline Pareto frontier points (green) if available
    if pareto_points_simple:
        simple_costs = [point[2] for point in pareto_points_simple]
        simple_accuracies = [point[1] for point in pareto_points_simple]
        plt.scatter(simple_costs, simple_accuracies, c="green", s=100, alpha=0.8, label="Simple Baseline (with iteration #)")
    else:
        print("Note: No Simple Baseline Pareto frontier points to plot")
    
    # Plot baseline Pareto frontier points (black) if available
    if pareto_points_baseline:
        baseline_costs = [point[2] for point in pareto_points_baseline]
        baseline_accuracies = [point[1] for point in pareto_points_baseline]
        plt.scatter(baseline_costs, baseline_accuracies, c="black", s=100, alpha=0.8, label="Baseline (with iteration #)")
    else:
        print("Note: No Baseline Pareto frontier points to plot")
    
    # Plot MCTS Pareto frontier points (blue) if available
    if pareto_points_mcts:
        mcts_costs = [point[2] for point in pareto_points_mcts]
        mcts_accuracies = [point[1] for point in pareto_points_mcts]
        plt.scatter(mcts_costs, mcts_accuracies, c="blue", s=100, alpha=0.8, label="MCTS (with iteration #)")
    else:
        print("Note: No MCTS Pareto frontier points to plot")
    
    # Plot Abacus pareto frontier points (purple rectangles) if available
    if pareto_points_abacus:
        abacus_costs = [point[2] for point in pareto_points_abacus]
        abacus_accuracies = [point[1] for point in pareto_points_abacus]
        plt.scatter(abacus_costs, abacus_accuracies, c="purple", s=100, alpha=0.8, label="Abacus", marker="s")
    else:
        print("Note: No Abacus Pareto frontier points to plot")
    
    # Plot all non-frontier nodes with lighter colors
    # Simple baseline non-frontier nodes
    if valid_simple:
        simple_frontier_costs = set(point[2] for point in pareto_points_simple) if pareto_points_simple else set()
        simple_frontier_accuracies = set(point[1] for point in pareto_points_simple) if pareto_points_simple else set()
        
        non_frontier_simple = [item for item in valid_simple 
                             if not (item["cost"] in simple_frontier_costs and item[accuracy_metric] in simple_frontier_accuracies)]
        
        if non_frontier_simple:
            non_frontier_costs = [item["cost"] for item in non_frontier_simple]
            non_frontier_accuracies = [item[accuracy_metric] for item in non_frontier_simple]
            plt.scatter(non_frontier_costs, non_frontier_accuracies, c="green", s=50, alpha=0.3, label="Simple Baseline (Non-Frontier)")
    
    # Baseline non-frontier nodes
    if valid_baseline:
        baseline_frontier_costs = set(point[2] for point in pareto_points_baseline) if pareto_points_baseline else set()
        baseline_frontier_accuracies = set(point[1] for point in pareto_points_baseline) if pareto_points_baseline else set()
        
        non_frontier_baseline = [item for item in valid_baseline 
                               if not (item["cost"] in baseline_frontier_costs and item[accuracy_metric] in baseline_frontier_accuracies)]
        
        if non_frontier_baseline:
            non_frontier_costs = [item["cost"] for item in non_frontier_baseline]
            non_frontier_accuracies = [item[accuracy_metric] for item in non_frontier_baseline]
            plt.scatter(non_frontier_costs, non_frontier_accuracies, c="black", s=50, alpha=0.3, label="Baseline (Non-Frontier)")
    
    # MCTS non-frontier nodes
    if valid_mcts:
        mcts_frontier_costs = set(point[2] for point in pareto_points_mcts) if pareto_points_mcts else set()
        mcts_frontier_accuracies = set(point[1] for point in pareto_points_mcts) if pareto_points_mcts else set()
        
        non_frontier_mcts = [item for item in valid_mcts 
                           if not (item["cost"] in mcts_frontier_costs and item[accuracy_metric] in mcts_frontier_accuracies)]
        
        if non_frontier_mcts:
            non_frontier_costs = [item["cost"] for item in non_frontier_mcts]
            non_frontier_accuracies = [item[accuracy_metric] for item in non_frontier_mcts]
            plt.scatter(non_frontier_costs, non_frontier_accuracies, c="blue", s=50, alpha=0.3, label="MCTS (Non-Frontier)")
    
    # Abacus non-frontier nodes
    if valid_abacus:
        abacus_frontier_costs = set(point[2] for point in pareto_points_abacus) if pareto_points_abacus else set()
        abacus_frontier_accuracies = set(point[1] for point in pareto_points_abacus) if pareto_points_abacus else set()
        
        non_frontier_abacus = [item for item in valid_abacus 
                             if not (item["cost"] in abacus_frontier_costs and item[accuracy_metric] in abacus_frontier_accuracies)]
        
        if non_frontier_abacus:
            non_frontier_costs = [item["cost"] for item in non_frontier_abacus]
            non_frontier_accuracies = [item[accuracy_metric] for item in non_frontier_abacus]
            plt.scatter(non_frontier_costs, non_frontier_accuracies, c="purple", s=50, alpha=0.3, label="Abacus (Non-Frontier)")
    
    # Plot original point in red
    if original_point:
        plt.scatter(original_point["cost"], original_point[accuracy_metric], 
                   c="red", s=100, marker="o",
                   label="Original", zorder=5)
        # Add annotation for original point
        plt.annotate("original", (original_point["cost"], original_point[accuracy_metric]), 
                    textcoords="offset points", xytext=(5, 5), 
                    fontsize=10, fontweight="bold", color="red")
    
    # Add annotations for simple baseline Pareto frontier points
    if pareto_points_simple:
        for i, (iteration, accuracy, cost) in enumerate(pareto_points_simple):
            if iteration == "0":
                label = "original"
            else:
                label = str(iteration)  # Use the actual iteration value
            plt.annotate(label, (cost, accuracy), 
                        textcoords="offset points", xytext=(5, 5), 
                        fontsize=10, fontweight="bold", color="green")
    
    # Add annotations for baseline Pareto frontier points
    if pareto_points_baseline:
        counter = 1
        for i, (iteration, accuracy, cost) in enumerate(pareto_points_baseline):
            if iteration == "original_output":
                continue
            else:
                label = str(counter)  # Use sequential numbers 1, 2, 3...
                counter += 1
            plt.annotate(label, (cost, accuracy), 
                        textcoords="offset points", xytext=(5, 5), 
                        fontsize=10, fontweight="bold", color="black")
    
    # Add annotations for MCTS Pareto frontier points
    if pareto_points_mcts:
        for i, (iteration, accuracy, cost) in enumerate(pareto_points_mcts):
            if iteration == "0":
                continue
            else:
                label = str(iteration)  # Use the actual iteration value
            plt.annotate(label, (cost, accuracy), 
                        textcoords="offset points", xytext=(5, 5), 
                        fontsize=10, fontweight="bold", color="blue")
    
    # Add annotations for Abacus Pareto frontier points
    if pareto_points_abacus:
        for i, (iteration, accuracy, cost) in enumerate(pareto_points_abacus):
            label = str(iteration)  # Use the actual iteration value
            plt.annotate(label, (cost, accuracy), 
                        textcoords="offset points", xytext=(5, 5), 
                        fontsize=10, fontweight="bold", color="purple")
    
    # Add labels and title
    plt.xlabel("Cost ($)", fontsize=12)
    plt.ylabel(f"{accuracy_metric.replace('_', ' ').title()}", fontsize=12)
    
    # Always show all three approaches in the title
    title = f"Cost vs {accuracy_metric.replace('_', ' ').title()} - {exp_name.upper()} Dataset"
    plt.title(title, fontsize=14)
    
    # Set log scale for cost axis
    plt.xscale('log')
    
    # Add grid and legend
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.3))
    
    # Save the plot if output path is provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plot_filename = f"pareto_frontier_comparison_{exp_name}.png"
        plot_path = os.path.join(output_path, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"📈 Pareto frontier comparison plot saved to: {plot_path}")
    
    plt.show()


def main():
    
    # evaluation_file_baseline = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/biodex/baseline/evaluation_metrics.json"
    # evaluation_file_mcts = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/biodex/mcts/evaluation_metrics.json"
    # evaluation_file_simple = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/biodex/simple_baseline/evaluation_metrics.json"
    # exp_name = "biodex"


    # evaluation_file_baseline = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/cuad/baseline/evaluation_metrics.json"
    # evaluation_file_mcts = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/cuad/mcts/evaluation_metrics.json"
    # evaluation_file_simple = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/cuad/simple_baseline/evaluation_metrics.json"
    # evaluation_file_abacus = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/cuad/abacus/evaluation_metrics.json"
    # exp_name = "cuad"

    evaluation_file_baseline = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/game_reviews/baseline/evaluation_metrics.json"
    evaluation_file_mcts = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/game_reviews/mcts/evaluation_metrics.json"
    evaluation_file_simple = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/game_reviews/simple_baseline/evaluation_metrics.json"
    evaluation_file_abacus = None  # Set to None when not using Abacus
    exp_name = "game_reviews"

    
    # Find Pareto frontier for all approaches
    pareto_points_baseline = find_pareto_frontier(evaluation_file_baseline, exp_name)
    pareto_points_mcts = find_pareto_frontier(evaluation_file_mcts, exp_name)
    pareto_points_simple = find_pareto_frontier(evaluation_file_simple, exp_name)
    
    # Load Abacus data if file is provided
    pareto_points_abacus = None
    if evaluation_file_abacus:
        pareto_points_abacus = find_pareto_frontier(evaluation_file_abacus, exp_name)
    
    # Check if we have at least baseline and MCTS (required for comparison)
    if pareto_points_baseline and pareto_points_mcts:
        # Plot all approaches together
        print("\n" + "=" * 60)
        if pareto_points_abacus:
            print("PLOTTING ALL FOUR APPROACHES TOGETHER")
        else:
            print("PLOTTING ALL THREE APPROACHES TOGETHER")
        print("=" * 60)
        output_dir = os.path.dirname(evaluation_file_baseline)
        plot_pareto_frontier_comparison(evaluation_file_baseline, evaluation_file_mcts, evaluation_file_simple, evaluation_file_abacus, exp_name, pareto_points_baseline, pareto_points_mcts, pareto_points_simple, pareto_points_abacus, output_dir)
    else:
        print("No Pareto frontier points found for baseline or MCTS approaches.")
    
    return 0


if __name__ == "__main__":
    exit(main())