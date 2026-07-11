"""
Script to run domain-specific task evaluation (details extraction, translation) using the model factory.
Extracted from notebook: Evaluation section.
"""
import argparse
import json
import os
import sys
import torch
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.model_factory import get_model
from src.tasks.news_extraction_task import build_details_extraction_messages
from src.tasks.translation_task import build_translation_messages
from src.inference.utils import parse_json

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ArabicLLM tasks on base, finetuned, openai, or vllm backends.")
    parser.add_argument(
        "--model-type",
        choices=["base", "finetuned", "openai", "vllm"],
        default=os.getenv("DEFAULT_MODEL_TYPE", "vllm"),
        help="Backend model wrapper type to use.",
    )
    args = parser.parse_args()

    # Load configuration parameters
    base_model_id = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path = os.getenv("LORA_PATH", "/gdrive/MyDrive/ara-finetune/models")
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    cloud_model = os.getenv("CLOUD_MODEL_ID", "openai/gpt-oss-120b:free")
    vllm_url = os.getenv("VLLM_ENDPOINT", "http://localhost:8000")
    vllm_model_id = os.getenv("VLLM_MODEL_ID", "news-lora")

    print(f"Initializing {args.model_type.upper()} model for evaluation...")

    # Build kwargs for factory
    model_kwargs = {
        "model_id": base_model_id,
        "base_model_id": base_model_id,
        "adapter_path": adapter_path,
        "api_key": openai_key,
        "base_url": openai_base,
        "default_model": cloud_model,
        "vllm_endpoint": vllm_url,
        "vllm_model_id": vllm_model_id,
        "lora_name": "news-lora" if args.model_type == "vllm" else None,
        "lora_path": adapter_path if args.model_type == "vllm" else None,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else None,
    }

    model = get_model(args.model_type, **model_kwargs)

    # Standard evaluation sample story from notebook
    sample_story = (
        "ذكرت مجلة فوربس أن العائلة تلعب دورا محوريا في تشكيل علاقة الأفراد بالمال، "
        "حيث تتأثر هذه العلاقة بأنماط السلوك المالي المتوارثة عبر الأجيال. "
        "التقرير يستند إلى أبحاث الأستاذ الجامعي شاين إنيت حول الرفاه المالي."
    )

    print("\n-----")
    print("Evaluating Task 1: Details Extraction")
    print("-----")
    
    details_messages = build_details_extraction_messages(sample_story)
    details_raw = model.generate(details_messages, max_new_tokens=1024)
    print("--- Raw output ---")
    print(details_raw)
    
    details_parsed = parse_json(details_raw)
    print("--- Parsed Output ---")
    if details_parsed:
        print(json.dumps(details_parsed, ensure_ascii=False, indent=2))
    else:
        print("Failed to repair/parse JSON output.")

    print("\n-----")
    print("Evaluating Task 2: Translation (English)")
    print("-----")
    
    trans_messages = build_translation_messages(sample_story, targeted_lang="English")
    trans_raw = model.generate(trans_messages, max_new_tokens=1024)
    print("--- Raw output ---")
    print(trans_raw)
    
    trans_parsed = parse_json(trans_raw)
    print("--- Parsed Output ---")
    if trans_parsed:
        print(json.dumps(trans_parsed, ensure_ascii=False, indent=2))
    else:
        print("Failed to repair/parse JSON output.")


if __name__ == "__main__":
    main()
