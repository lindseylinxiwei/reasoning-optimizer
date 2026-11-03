#!/usr/bin/env python3
"""
Run frontier models on test datasets.

This script:
1. For each dataset, loads the original pipeline YAML
2. For each frontier model in FRONTIER_MODELS, modifies the pipeline to use that model
3. Runs the pipeline on test data
4. Evaluates the results
5. Saves results to outputs/{dataset}_frontier_model/
"""

import json
import yaml
import traceback
from pathlib import Path
from typing import Any, Dict, List
import modal
from datetime import datetime
import time

from docetl.runner import DSLRunner
from experiments.reasoning.utils import app, volume, VOLUME_MOUNT_PATH, image
from experiments.reasoning.evaluation.utils import dataset_accuracy_metrics, get_evaluate_func
from docetl.reasoning_optimizer.directives.change_model_cost import FRONTIER_MODELS


# Dataset configurations
DATASETS = ["cuad", "blackvault", "game_reviews", "sustainability", "biodex", "medec"]


def get_full_model_name(model_name: str) -> str:
    """
    Get the full model name with the appropriate prefix.
    
    Args:
        model_name: Short model name (e.g., 'gpt-5-nano', 'gemini-2.5-flash')
        
    Returns:
        Full model name with prefix (e.g., 'azure/gpt-5-nano', 'gemini/gemini-2.5-flash')
    """
    if model_name.startswith('gemini'):
        return f"gemini/{model_name}"
    elif model_name.startswith('gpt'):
        return f"azure/{model_name}"
    else:
        # Return as-is if unknown prefix
        return model_name


def change_model_in_config(config: Dict, model_name: str) -> Dict:
    """
    Change the model in all operations of the pipeline config.
    
    Args:
        config: Pipeline configuration dictionary
        model_name: The model name to set (e.g., 'gpt-5-nano')
        
    Returns:
        Modified configuration dictionary
    """
    # Get full model name with prefix
    full_model_name = get_full_model_name(model_name)
    
    # Change the default model if it exists
    if 'default_model' in config:
        config['default_model'] = full_model_name
    
    # Change model in all operations
    if 'operations' in config:
        for op in config['operations']:
            if 'model' in op or 'default_model' in config:
                op['model'] = full_model_name
    
    return config


@app.function(
    image=image, 
    secrets=[modal.Secret.from_dotenv()], 
    volumes={VOLUME_MOUNT_PATH: volume}, 
    timeout=60 * 60 * 2  # 2 hours timeout
)
def run_frontier_model_test(dataset: str, model_name: str) -> Dict[str, Any]:
    """
    Run a specific frontier model on the test dataset.
    
    Args:
        dataset: Dataset name (e.g., 'cuad')
        model_name: Model name (e.g., 'gpt-5-nano')
    
    Returns:
        Dictionary with test results
    """
    try:
        print(f"\n{'='*60}")
        print(f"Running frontier model test: {dataset} - {model_name}")
        print(f"{'='*60}\n")
        
        # Check if output already exists FIRST (before any other work)
        base_output_dir = Path(VOLUME_MOUNT_PATH) / "outputs"
        safe_model_name = model_name.replace('.', '_').replace('-', '_')
        output_dir = base_output_dir / f"{dataset}_frontier_model"
        test_output = str(output_dir / f"{safe_model_name}_test.json")
        
        if Path(test_output).exists():
            print(f"⏭️  Skipping {model_name} - output file already exists: {test_output}")
            print("   Evaluating existing results...")
            
            # Evaluate existing results
            eval_func = get_evaluate_func(dataset, mode="test")
            if eval_func:
                accuracy_results = eval_func(safe_model_name, test_output)
                accuracy_metric = dataset_accuracy_metrics.get(dataset, "accuracy")
                accuracy = accuracy_results.get(accuracy_metric, 0.0)
                print(f"📈 Accuracy ({accuracy_metric}): {accuracy:.4f}")
            else:
                accuracy = None
                accuracy_metric = None
            
            return {
                "success": True,
                "dataset": dataset,
                "model": model_name,
                "cost": None,  # Cost not available for cached results
                "accuracy": accuracy,
                "accuracy_metric": accuracy_metric,
                "output_path": test_output,
                "cached": True
            }
        
        # Load original pipeline YAML
        pipeline_path = Path(f"experiments/reasoning/pipelines/{dataset}.yaml")
        if not pipeline_path.exists():
            return {
                "success": False,
                "dataset": dataset,
                "model": model_name,
                "error": f"Pipeline file not found: {pipeline_path}"
            }
        
        with open(pipeline_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Change the model in the config
        full_model_name = get_full_model_name(model_name)
        config = change_model_in_config(config, model_name)
        print(f"🔧 Changed model to: {full_model_name}")
        
        # Change dataset path from train to test
        if 'datasets' in config:
            for dataset_key, dataset_config in config['datasets'].items():
                original_path = dataset_config.get('path', '')
                # Replace 'train' with 'test' in the path
                test_path = original_path.replace('/train/', '/test/')
                config['datasets'][dataset_key]['path'] = test_path
                print(f"📂 Changed dataset path to: {test_path}")
        
        # Set output path in config
        config['pipeline']['output']['path'] = test_output
        print(f"📤 Output path: {test_output}")
        
        # Create output directory
        Path(test_output).parent.mkdir(parents=True, exist_ok=True)
        
        # Save modified YAML to /tmp
        test_yaml_path = Path("/tmp") / f"{dataset}_{safe_model_name}_test.yaml"
        with open(test_yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        
        # Run the pipeline
        print("🚀 Running pipeline...")
        start_time = time.time()
        runner = DSLRunner.from_yaml(str(test_yaml_path))
        runner.load()
        
        if runner.last_op_container:
            data, _, _ = runner.last_op_container.next()
            runner.save(data)
        
        total_cost = runner.total_cost
        runner.reset_env()
        end_time = time.time()
        latency = end_time - start_time
        
        print(f"✅ Pipeline completed. Cost: ${total_cost:.6f}, Latency: {latency:.3f}s")
        
        # Evaluate accuracy
        eval_func = get_evaluate_func(dataset, mode="test")
        if eval_func:
            accuracy_results = eval_func(safe_model_name, test_output)
            accuracy_metric = dataset_accuracy_metrics.get(dataset, "accuracy")
            accuracy = accuracy_results.get(accuracy_metric, 0.0)
            print(f"📈 Accuracy ({accuracy_metric}): {accuracy:.4f}")
        else:
            print(f"⚠️  No evaluation function found for {dataset}")
            accuracy = None
            accuracy_metric = None
        
        # Commit changes to Modal volume
        volume.commit()
        
        return {
            "success": True,
            "dataset": dataset,
            "model": model_name,
            "cost": total_cost,
            "accuracy": accuracy,
            "accuracy_metric": accuracy_metric,
            "latency": latency,
            "output_path": test_output
        }
        
    except Exception as e:
        print(f"❌ Error running frontier model test: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "dataset": dataset,
            "model": model_name,
            "error": str(e)
        }


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60 * 4  # 4 hours timeout for all models
)
def run_all_frontier_models_for_dataset(dataset: str) -> Dict[str, Any]:
    """
    Run all frontier models for a specific dataset on test data.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Dictionary with all test results
    """
    print(f"\n{'='*70}")
    print(f"RUNNING ALL FRONTIER MODELS FOR {dataset.upper()}")
    print(f"{'='*70}\n")
    
    # Get frontier models for this dataset
    if dataset not in FRONTIER_MODELS:
        return {
            "success": False,
            "dataset": dataset,
            "error": f"No frontier models defined for dataset: {dataset}"
        }
    
    frontier_models = FRONTIER_MODELS[dataset]
    print(f"📊 Frontier models for {dataset}: {frontier_models}")
    
    all_results = {}
    
    for model_name in frontier_models:
        print(f"\n📊 Processing {model_name}...")
        result = run_frontier_model_test.local(dataset, model_name)
        all_results[model_name] = result
        
        if result["success"]:
            print(f"✅ {model_name} completed successfully")
            if result.get("accuracy") is not None:
                print(f"   Accuracy: {result['accuracy']:.4f}")
            if result.get("cost") is not None:
                print(f"   Cost: ${result['cost']:.6f}")
        else:
            print(f"❌ {model_name} failed: {result.get('error', 'Unknown error')}")
    
    # Generate summary
    summary = {
        "dataset": dataset,
        "timestamp": datetime.now().isoformat(),
        "frontier_models": list(frontier_models),
        "results": all_results,
        "successful_models": [m for m, r in all_results.items() if r["success"]],
        "failed_models": [m for m, r in all_results.items() if not r["success"]]
    }
    
    # Save summary to file
    base_output_dir = Path(VOLUME_MOUNT_PATH) / "outputs"
    summary_path = base_output_dir / f"{dataset}_frontier_model" / "frontier_models_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Summary saved to: {summary_path}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY FOR {dataset.upper()}")
    print(f"{'='*70}")
    print(f"{'Model':<25} | {'Accuracy':<12} | {'Cost':<12} | {'Status':<10}")
    print("-" * 70)
    
    for model_name, result in all_results.items():
        if result["success"]:
            acc_str = f"{result['accuracy']:.4f}" if result.get('accuracy') is not None else "N/A"
            cost_str = f"${result['cost']:.4f}" if result.get('cost') is not None else "cached"
            status = "✅"
        else:
            acc_str = "N/A"
            cost_str = "N/A"
            status = "❌"
        print(f"{model_name:<25} | {acc_str:<12} | {cost_str:<12} | {status:<10}")
    
    print(f"{'='*70}\n")
    
    # Commit changes to Modal volume
    volume.commit()
    
    return summary


@app.local_entrypoint()
def main(dataset: str = "cuad", model: str = "all"):
    """
    Main entrypoint for running frontier models on test data.
    
    Args:
        dataset: Dataset to process ('cuad', 'blackvault', etc., or 'all' for all datasets)
        model: Model to run ('all' for all frontier models, or specific model name)
    """
    if dataset not in DATASETS + ["all"]:
        print(f"❌ Invalid dataset: {dataset}")
        print(f"   Valid options: {', '.join(DATASETS + ['all'])}")
        return
    
    datasets_to_process = DATASETS if dataset == "all" else [dataset]
    
    for dataset_name in datasets_to_process:
        if model == "all":
            # Run all frontier models for this dataset
            result = run_all_frontier_models_for_dataset.remote(dataset_name)
            print(f"\n{'='*70}")
            print(f"COMPLETED: {dataset_name.upper()}")
            print(f"{'='*70}")
            print(f"✅ Successful models: {', '.join(result['successful_models'])}")
            if result['failed_models']:
                print(f"❌ Failed models: {', '.join(result['failed_models'])}")
        else:
            # Run specific model
            if dataset_name not in FRONTIER_MODELS:
                print(f"❌ No frontier models defined for {dataset_name}")
                continue
            
            if model not in FRONTIER_MODELS[dataset_name]:
                print(f"❌ Model {model} is not a frontier model for {dataset_name}")
                print(f"   Valid frontier models: {', '.join(FRONTIER_MODELS[dataset_name])}")
                continue
            
            result = run_frontier_model_test.remote(dataset_name, model)
            if result["success"]:
                print(f"\n✅ Successfully processed {dataset_name} - {model}")
                if result.get("accuracy") is not None:
                    print(f"   Accuracy: {result['accuracy']:.4f}")
                if result.get("cost") is not None:
                    print(f"   Cost: ${result['cost']:.6f}")
            else:
                print(f"\n❌ Failed to process {dataset_name} - {model}")
                print(f"   Error: {result.get('error', 'Unknown error')}")
