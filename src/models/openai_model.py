"""
OpenAI-compatible client wrapper for calling models (e.g. GPT-4o on OpenRouter).
Extracted from notebook: Evaluation > OpenAI section.
"""
from openai import OpenAI


class OpenAIModel:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", default_model: str = "openai/gpt-oss-120b:free"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model

    def generate(self, messages: list, model: str = None, temperature: float = 0.2) -> str:
        """
        Creates a chat completion and returns the generated content string.
        """
        model_name = model or self.default_model
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def generate_with_usage(self, messages: list, model: str = None, temperature: float = 0.2):
        """
        Creates a chat completion and returns both content and usage stats.
        Useful for SFT generation cost tracking.
        """
        model_name = model or self.default_model
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content, response
