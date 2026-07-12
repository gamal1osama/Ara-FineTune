"""
Token analyzer for calculating total input/output tokens from benchmark output logs.
Extracted from notebook: Load Testing section.
"""
import json
import os
from transformers import AutoTokenizer


def analyze_tokens(tokens_file_path: str, base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Decodes the prompt and response tokens and prints statistics.
    """
    if not os.path.exists(tokens_file_path):
        print(f"Tokens file not found: {tokens_file_path}")
        return None

    vllm_tokens = []
    with open(tokens_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            vllm_tokens.append(json.loads(line_str))

    if not vllm_tokens:
        print("No tokens found in file to analyze.")
        return None

    print(f"Initializing tokenizer {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    total_input_tokens = sum([len(tokenizer.encode(rec["prompt"])) for rec in vllm_tokens])
    total_output_tokens = sum([len(tokenizer.encode(rec["response"])) for rec in vllm_tokens])

    results = {
        "num_requests": len(vllm_tokens),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }

    print(f"--- Analysis Results for {tokens_file_path} ---")
    print(f"Total Requests:      {results['num_requests']}")
    print(f"Total Input Tokens:  {results['total_input_tokens']}")
    print(f"Total Output Tokens: {results['total_output_tokens']}")
    print(f"Total Tokens:        {results['total_tokens']}")
    print("---------------------------------------------")
    
    return results


if __name__ == "__main__":
    import sys
    tokens_file = sys.argv[1] if len(sys.argv) > 1 else "./vllm_tokens.txt"
    model_id = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-1.5B-Instruct"
    analyze_tokens(tokens_file, model_id)
