"""
Model factory for instantiating various model wrapper types.
"""
from src.models.base_model import HFModel
from src.models.finetuned_model import FinetunedModel
from src.models.openai_model import OpenAIModel
from src.models.vllm_model import VLLMModel


def get_model(model_type: str, **kwargs):
    """
    Factory method to initialize and return a model instance.

    Args:
        model_type: One of 'base', 'finetuned', 'openai', 'vllm'
        **kwargs: Arguments passed directly to the model's constructor.
    """
    model_type = model_type.lower()
    
    if model_type == "base":
        return HFModel(
            model_id=kwargs.get("model_id"),
            device=kwargs.get("device", "cuda"),
            torch_dtype=kwargs.get("torch_dtype", None)
        )
    elif model_type == "finetuned":
        return FinetunedModel(
            model_id=kwargs.get("model_id"),
            adapter_path=kwargs.get("adapter_path"),
            device=kwargs.get("device", "cuda"),
            torch_dtype=kwargs.get("torch_dtype", None)
        )
    elif model_type == "openai" or model_type == "openrouter":
        return OpenAIModel(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url", "https://openrouter.ai/api/v1"),
            default_model=kwargs.get("default_model", "openai/gpt-oss-120b:free")
        )
    elif model_type == "vllm":
        return VLLMModel(
            base_model_id=kwargs.get("base_model_id"),
            vllm_endpoint=kwargs.get("vllm_endpoint", "http://localhost:8000"),
            vllm_model_id=kwargs.get("vllm_model_id", "news-lora"),
            lora_name=kwargs.get("lora_name", "news-lora"),
            lora_path=kwargs.get("lora_path", "/gdrive/MyDrive/ara-finetune/models")
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'base', 'finetuned', 'openai', or 'vllm'.")
