"""
Base HuggingFace local inference model class.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFModel:
    def __init__(self, model_id: str, device: str = "cuda", torch_dtype=None):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load tokenizer and model as done in the notebook
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch_dtype
        )
        if device == "cuda" and not hasattr(self.model, "device"):
            self.model = self.model.to(device)

    def generate(self, messages: list, max_new_tokens: int = 1024, temperature: float = None) -> str:
        """
        Generate response using chat template.
        Uses greedy decoding (do_sample=False) or temperature if provided.
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Matches notebook: do_sample=False, top_k=None, temperature=None, top_p=None
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False if temperature is None else True,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
            
        generation_ids = self.model.generate(
            model_inputs.input_ids,
            **gen_kwargs
        )
        
        generated_ids = [
            output_ids[len(input_ids) :]
            for output_ids, input_ids in zip(generation_ids, model_inputs.input_ids)
        ]
        
        return self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
