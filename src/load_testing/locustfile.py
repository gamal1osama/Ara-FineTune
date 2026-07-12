"""
Locust load testing configuration file.
Extracted from notebook: Load Testing section.
Run with: locust -f src/load_testing/locustfile.py
"""
import json
import os
import random
from faker import Faker
from locust import HttpUser, between, task

fake = Faker('ar')


class CompletionLoadTest(HttpUser):
    wait_time = between(1, 3)

    @task
    def post_completion(self):
        # Allow customization through environment variables, fallback to notebook defaults
        model_id = os.getenv("VLLM_MODEL_ID", "news-lora")
        lora_name = os.getenv("LORA_NAME", "news-lora")
        lora_path = os.getenv("LORA_PATH", "/gdrive/MyDrive/ara-finetune/models")
        tokens_file = os.getenv("VLLM_TOKENS_FILE", "./vllm_tokens.txt")

        prompt = fake.text(max_nb_chars=random.randint(150, 200))

        message = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.3,
            "lora_name": lora_name,
            "lora_path": lora_path
        }

        llm_response = self.client.post("/v1/completions", json=message)

        if llm_response.status_code == 200:
            # Matches notebook: save outputs for token analysis
            with open(tokens_file, "a", encoding="utf-8") as dest:
                dest.write(json.dumps({
                    "prompt": prompt,
                    "response": llm_response.json()["choices"][0]["text"],
                }, ensure_ascii=False) + "\n")
