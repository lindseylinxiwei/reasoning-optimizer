"""
Calculate the sum of model costs for each dataset.
"""

MODEL_COSTS = {
    "cuad": {
        "gpt-5": 2.282243,
        "gpt-5-mini": 0.605154,
        "gpt-5-nano": 0.231796,
        "gpt-4.1": 2.026396,
        "gpt-4.1-mini": 0.380290,
        "gpt-4.1-nano": 0.080214,
        "gpt-4o": 2.210165,
        "gpt-4o-mini": 0.137882,
        "gemini-2.5-pro": 2.820552,
        "gemini-2.5-flash": 1.032156,
        "gemini-2.5-flash-lite": 0.097669,
    },
    "medec": {
        "gpt-5": 1.008255,
        "gpt-5-mini": 0.120811,
        "gpt-5-nano": 0.062694,
        "gpt-4.1": 0.099804,
        "gpt-4.1-mini": 0.019399,
        "gpt-4.1-nano": 0.004886,
        "gpt-4o": 0.135778,
        "gpt-4o-mini": 0.008168,
        "gemini-2.5-pro": 0.689124,
        "gemini-2.5-flash-lite": 0.005106,
    },
    "biodex": {
        "gpt-5": 10.06,
        "gpt-5-mini": 2.24,
        "gpt-5-nano": 0.45,
        "gpt-4.1": 15.41,
        "gpt-4.1-mini": 3.08,
        "gpt-4.1-nano": 0.77,
        "gpt-4o": 17.57,
        "gpt-4o-mini": 1.05,
        "gemini-2.5-flash-lite": 0.71,
    },
    "sustainability": {
        "gpt-5": 8.100397,
        "gpt-5-mini": 1.608668,
        "gpt-5-nano": 0.36,
        "gpt-4.1": 14.88,
        "gpt-4.1-mini": 2.849562,
        "gpt-4.1-nano": 0.74,
        "gpt-4o": 13.23,
        "gpt-4o-mini": 0.82,
        "gemini-2.5-pro": 16.68,
        "gemini-2.5-flash": 2.852281,
        "gemini-2.5-flash-lite": 0.795008,
    },
    "blackvault": {
        "gpt-5": 2.607966,
        "gpt-5-mini": 0.412403,
        "gpt-5-nano": 0.111214,
        "gpt-4.1": 2.449074,
        "gpt-4.1-nano": 0.125311,
        "gpt-4o": 2.934132,
        "gpt-4o-mini": 0.174887,
        "gemini-2.5-pro": 3.129470,
    },
    "game_reviews": {
        "gpt-5": 6.560770,
        "gpt-5-mini": 1.370671,
        "gpt-5-nano": 0.464833,
        "gpt-4.1": 8.729372,
        "gpt-4.1-mini": 1.735382,
        "gpt-4.1-nano": 0.445612,
        "gpt-4o": 10.044458,
        "gpt-4o-mini": 0.602058,
        "gemini-2.5-pro": 10.528458,
        "gemini-2.5-flash-lite": 0.497641,
    }
}


def sum_model_costs_by_dataset():
    """
    Calculate the sum of model costs for each dataset.
    
    Returns:
        dict: A dictionary mapping dataset names to their total model cost sums.
    """
    dataset_sums = {}
    
    for dataset, model_costs in MODEL_COSTS.items():
        total_sum = sum(model_costs.values())
        dataset_sums[dataset] = total_sum
    
    return dataset_sums


if __name__ == "__main__":
    results = sum_model_costs_by_dataset()
    
    print("Sum of model costs by dataset:")
    print("-" * 40)
    for dataset, total_sum in results.items():
        print(f"{dataset:20s}: {total_sum:.6f}")
    
    print("\n" + "-" * 40)
    print(f"Total across all datasets: {sum(results.values()):.6f}")

