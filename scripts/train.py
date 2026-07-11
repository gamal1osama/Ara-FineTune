"""
Script to launch LLaMA-Factory fine-tuning using the config YAML.
Extracted from notebook: Training section.
"""
import os
import subprocess
import sys
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()


def main():
    # Allow passing custom YAML path, default to configs/news_finetune.yaml
    default_generated_config = os.path.join("LLaMA-Factory", "examples", "train_lora", "news_finetune.yaml")
    config_path = sys.argv[1] if len(sys.argv) > 1 else (
        default_generated_config if os.path.exists(default_generated_config) else "configs/news_finetune.yaml"
    )
    
    if not os.path.exists(config_path):
        print(f"Error: Fine-tuning config not found at: {config_path}")
        sys.exit(1)

    print(f"Launching LLaMA-Factory training with config: {config_path}")
    
    # Execute llamafactory-cli command
    cmd = ["llamafactory-cli", "train", config_path]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        process.wait()
        
        if process.returncode == 0:
            print("Training completed successfully.")
        else:
            print(f"Training failed with exit code: {process.returncode}")
            sys.exit(process.returncode)
            
    except FileNotFoundError:
        print("Error: llamafactory-cli command not found. Please install LLaMA-Factory.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
