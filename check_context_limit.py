"""
Script to check context limits for Together AI models.
"""
import os
from dotenv import load_dotenv
from litellm import model_cost

# Load environment variables from .env file
load_dotenv()

# Models to check
models_to_check = [
    "Kimi-K2-Thinking",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
]

print("=" * 60)
print("Checking context limits from litellm model_cost dictionary")
print("=" * 60)

for model in models_to_check:
    info = model_cost.get(model, {})
    if info:
        print(f"\n{model}:")
        print(f"  max_input_tokens: {info.get('max_input_tokens', 'N/A')}")
        print(f"  max_output_tokens: {info.get('max_output_tokens', 'N/A')}")
        print(f"  max_tokens: {info.get('max_tokens', 'N/A')}")
    else:
        print(f"\n{model}: Not found in litellm model_cost")

# Also try to get info via Together AI API directly
print("\n" + "=" * 60)
print("Checking via Together AI API (if TOGETHER_API_KEY is set)")
print("=" * 60)

try:
    import requests
    
    api_key = os.environ.get("TOGETHER_API_KEY")
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.together.xyz/v1/models", headers=headers)
        
        if response.status_code == 200:
            models = response.json()
            
            # Find Qwen model
            for model_info in models.get("data", models) if isinstance(models, dict) else models:
                model_id = model_info.get("id", "")
                if "Llama-4-Maverick-17B-128E-Instruct-FP8" in model_id or "Kimi-K2-Thinking" in model_id:
                    print(f"\n{model_id}:")
                    print(f"  context_length: {model_info.get('context_length', 'N/A')}")
                    print(f"  Full info: {model_info}")
        else:
            print(f"API request failed: {response.status_code}")
    else:
        print("TOGETHER_API_KEY not set, skipping API check")
        
except Exception as e:
    print(f"Error checking Together AI API: {e}")
