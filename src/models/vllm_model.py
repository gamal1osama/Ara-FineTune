"""
vLLM HTTP completions client model class.
Extracted from notebook: VLLM > Inference section.
"""
import requests
from transformers import AutoTokenizer


class VLLMModel:
    def __init__(
        self,
        base_model_id: str,
        vllm_endpoint: str = "http://localhost:8000",
        vllm_model_id: str = "news-lora",
        lora_name: str = "news-lora",
        lora_path: str = "/gdrive/MyDrive/ara-finetune/models"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.vllm_endpoint = vllm_endpoint.rstrip("/")
        self.vllm_model_id = vllm_model_id
        self.lora_name = lora_name
        self.lora_path = lora_path

    def generate(self, messages: list, max_new_tokens: int = 1024, temperature: float = 0.3) -> str:
        """
        Formats chat messages into prompt string using local tokenizer,
        then posts to the running vLLM completion server.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        payload = {
            "model": self.vllm_model_id,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature
        }
        
        # If LoRA attributes are requested / present, pass them
        if self.lora_name and self.lora_path:
            payload["lora_name"] = self.lora_name
            payload["lora_path"] = self.lora_path

        response = requests.post(
            f"{self.vllm_endpoint}/v1/completions",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        # Extract response text
        resp_json = response.json()
        return resp_json["choices"][0]["text"]
