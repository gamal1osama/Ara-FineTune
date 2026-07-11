"""
Script to generate synthetic SFT records from raw news samples using OpenAI/OpenRouter distillation.
"""
import os
import sys
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_loader import load_jsonl
from src.data.sft_builder import generate_details_extraction_sft, generate_translation_sft
from src.models.openai_model import OpenAIModel

# Load environment variables
load_dotenv()


def main():
    # Load configuration from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set. Please set it in a .env file.")
        sys.exit(1)

    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    cloud_model_id = os.getenv("CLOUD_MODEL_ID", "openai/gpt-oss-120b:free")

    raw_data_path = os.getenv("RAW_DATA_PATH")
    save_to = os.getenv("SFT_SAVE_PATH")

    if not raw_data_path:
        print("Error: RAW_DATA_PATH environment variable is not set. Please set it in a .env file.")
        sys.exit(1)

    if not save_to:
        print("Error: SFT_SAVE_PATH environment variable is not set. Please set it in a .env file.")
        sys.exit(1)

    print(f"Loading raw news stories from: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file not found at: {raw_data_path}")
        sys.exit(1)

    raw_data = load_jsonl(raw_data_path, shuffle=True, seed=101)
    
    # Initialize OpenAI Model client
    model = OpenAIModel(api_key=api_key, base_url=base_url, default_model=cloud_model_id)

    # 1. Distill Details Extraction SFT
    generate_details_extraction_sft(
        raw_data=raw_data,
        model=model,
        save_to=save_to,
        cloud_model_id=cloud_model_id
    )

    # 2. Distill Translation SFT
    generate_translation_sft(
        raw_data=raw_data,
        model=model,
        save_to=save_to,
        languages=["English", "French"],
        cloud_model_id=cloud_model_id
    )

    print(f"SFT Generation Complete. Output saved to: {save_to}")


if __name__ == "__main__":
    main()
