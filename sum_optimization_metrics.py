"""
Extract and sum optimization_time and optimization_cost values from JSON files.
"""

import json
import sys
import statistics
from pathlib import Path


biodex = [
        {
          "file": "gpt_nano_config_18",
          "cost": 0.12539669999999997,
          "accuracy": 0.0,
          "accuracy_metric": "avg_rp_at_5",
          "latency": 68.39466547966003
        },
        {
          "file": "gpt_nano_config_11-2-acc",
          "cost": 0.8936735499999998,
          "accuracy": 0.25866666666666666,
          "accuracy_metric": "avg_rp_at_5",
          "latency": 117.00188946723938
        },
        {
          "file": "gpt_nano_config_17",
          "cost": 0.8959857899999999,
          "accuracy": 0.24199999999999997,
          "accuracy_metric": "avg_rp_at_5",
          "latency": 94.12122845649719
        },
        {
          "file": "gpt_nano_config_5",
          "cost": 0.9008399899999997,
          "accuracy": 0.29449999999999993,
          "accuracy_metric": "avg_rp_at_5",
          "latency": 60.99222469329834
        },
        {
          "file": "gpt_mini_config_22",
          "cost": 4.39479321,
          "accuracy": 0.33533333333333326,
          "accuracy_metric": "avg_rp_at_5",
          "latency": 303.88859724998474
        },
        {
          "file": "gpt_config_20",
          "cost": 22.186787025000015,
          "accuracy": 0.37016666666666664,
          "accuracy_metric": "avg_rp_at_5",
          "latency": 335.43571758270264
        }
      ]
BV = [
        {
          "file": "gpt_nano_config_35",
          "cost": 0.06407919999999999,
          "accuracy": 6.818181818181818,
          "accuracy_metric": "avg_distinct_locations",
          "latency": 62.19431662559509
        },
        {
          "file": "gpt_nano_config_33",
          "cost": 0.19436684999999992,
          "accuracy": 8.727272727272727,
          "accuracy_metric": "avg_distinct_locations",
          "latency": 76.73519563674927
        },
        {
          "file": "gpt_nano_config_28",
          "cost": 0.16786108,
          "accuracy": 10.875,
          "accuracy_metric": "avg_distinct_locations",
          "latency": 99.00326776504517
        },
        {
          "file": "gpt_nano_config_36",
          "cost": 0.26667702000000004,
          "accuracy": 37.333333333333336,
          "accuracy_metric": "avg_distinct_locations",
          "latency": 144.63721680641174
        },
        {
          "file": "gemini_2.5_pro_config_30",
          "cost": 3.6545000000000005,
          "accuracy": 8.292682926829269,
          "accuracy_metric": "avg_distinct_locations",
          "latency": 80.83966112136841
        }
      ]
def sum_optimization_metrics(json_file_path):
    """
    Extract all optimization_time and optimization_cost values from JSON file and sum them.
    
    Args:
        json_file_path: Path to JSON file
        
    Returns:
        tuple: (total_optimization_time, total_optimization_cost)
    """
    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_optimization_time = 0.0
    total_optimization_cost = 0.0
    
    # Iterate through all keys, excluding metadata
    for key, value in data.items():
        if key == "metadata":
            continue
        
        # Check if value is a dict and contains required fields
        if isinstance(value, dict):
            if "optimization_time" in value:
                total_optimization_time += value["optimization_time"]
            if "optimization_cost" in value:
                total_optimization_cost += value["optimization_cost"]
    
    return total_optimization_time, total_optimization_cost


def calc_plan_execution_time_stats(json_file_path):
    """
    Extract all plan_execution_time values from JSON file and calculate mean and standard deviation.
    
    Args:
        json_file_path: Path to JSON file
        
    Returns:
        tuple: (mean, std, values_list)
    """
    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    plan_execution_times = []
    
    # Iterate through all keys, excluding metadata
    for key, value in data.items():
        if key == "metadata":
            continue
        
        # Check if value is a dict and contains required fields
        if isinstance(value, dict):
            if "plan_execution_time" in value:
                plan_execution_times.append(value["plan_execution_time"])
    
    if not plan_execution_times:
        return None, None, []
    
    # Calculate mean and standard deviation
    mean = statistics.mean(plan_execution_times)
    std = statistics.stdev(plan_execution_times) 
    
    return mean, std, plan_execution_times


def calc_cuad_latency_stats():
    """
    Calculate mean and standard deviation of latency values from cuad list.
    
    Returns:
        tuple: (mean, std, values_list)
    """
    latencies = [item["latency"] for item in BV if "latency" in item]
    
    if not latencies:
        return None, None, []
    
    mean = statistics.mean(latencies)
    std = statistics.stdev(latencies)
    print(latencies)
    
    return mean, std, latencies


def main():
    if len(sys.argv) < 2:
        print("python sum_optimization_metrics.py <json_file_path>")
        print("python sum_optimization_metrics.py experiments/reasoning/othersystems/biodex/pz_direct_evaluation.json")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    if not Path(json_file_path).exists():
        sys.exit(1)
    
    try:
        total_time, total_cost = sum_optimization_metrics(json_file_path)
        mean_time, std_time, values = calc_plan_execution_time_stats(json_file_path)
        
        print(f"File: {json_file_path}")
        print("-" * 60)
        print(f"optimization_time sum: {total_time:.2f}")
        print(f"optimization_cost sum: {total_cost:.2f}")
        if mean_time is not None:
            print(f"plan_execution_time mean: {mean_time:.2f}")
            print(f"plan_execution_time std: {std_time:.2f}")
            print(f"plan_execution_time: {mean_time:.2f} ± {std_time:.2f}")
        print("-" * 60)
        
    except json.JSONDecodeError as e:
        print(f"Error: JSON file parsing failed - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Calculate and print CUAD latency statistics
    mean_latency, std_latency, latency_values = calc_cuad_latency_stats()
    if mean_latency is not None:
        print("BV Latency Statistics:")
        print("-" * 60)
        print(f"Mean latency: {mean_latency:.2f}")
        print(f"Std latency: {std_latency:.2f}")
        print(f"Latency: {mean_latency:.2f} ± {std_latency:.2f}")
        print(f"Values: {latency_values}")
        print("-" * 60)
        print()
    
    main()

