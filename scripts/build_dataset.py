"""
Script to format the distilled SFT dataset into alpaca JSON format and create train/val splits.
"""
import os
import sys
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.sft_formatter import format_to_alpaca, split_and_save

load_dotenv()


def main():
    sft_jsonl_path = os.getenv("SFT_SAVE_PATH")
    output_dir = os.getenv("LLAMAFACTORY_DATA_DIR")

    if not sft_jsonl_path:
        print("Error: SFT_SAVE_PATH environment variable is not set. Please set it in a .env file.")
        sys.exit(1)

    if not output_dir:
        print("Error: LLAMAFACTORY_DATA_DIR environment variable is not set. Please set it in a .env file.")
        sys.exit(1)
    
    print(f"Reading SFT records from: {sft_jsonl_path}")
    if not os.path.exists(sft_jsonl_path):
        print(f"Error: SFT JSONL file not found at: {sft_jsonl_path}")
        print("Please run scripts/generate_sft.py first.")
        sys.exit(1)

    print("Formatting SFT data to Alpaca structure...")
    alpaca_data = format_to_alpaca(sft_jsonl_path, shuffle=True, seed=101)

    train_size = int(os.getenv("TRAIN_SIZE", "2700"))
    
    print(f"Splitting dataset (train_size={train_size}) and saving to: {output_dir}")
    split_and_save(alpaca_data, train_size=train_size, output_dir=output_dir)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
