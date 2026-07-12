"""
Finetuned model wrapper. Loads PEFT/LoRA adapter weights on top of the base model
and implements the Chinese-character banning logits processor.
Extracted from notebook: Evaluate > to ignore chinese response section.
"""
import torch
from src.models.base_model import HFModel


class FinetunedModel(HFModel):
    def __init__(self, model_id: str, adapter_path: str, device: str = "cuda", torch_dtype=None):
        super().__init__(model_id, device, torch_dtype)
        self.adapter_path = adapter_path
        
        # Load adapter weights
        self.model.load_adapter(adapter_path)
        self.chinese_mask = None

    def _build_chinese_mask(self, logits):
        """
        Creates a mask tensor containing indices of Chinese characters in the vocabulary.
        This process only runs once and cache the mask.
        """
        token_ids = torch.arange(logits.size(-1))
        # Batch decode all token ids to obtain strings
        decoded_tokens = self.tokenizer.batch_decode(token_ids.unsqueeze(1), skip_special_tokens=True)

        # Create a mask tensor to identify positions of Chinese characters
        mask = torch.tensor([
            any(
                0x4E00 <= ord(c) <= 0x9FFF or 
                0x3400 <= ord(c) <= 0x4DBF or 
                0xF900 <= ord(c) <= 0xFAFF 
                for c in token
            )
            for token in decoded_tokens
        ])
        return mask

    def generate(self, messages: list, max_new_tokens: int = 1024, temperature: float = 0.1) -> str:
        """
        Generates text using logits processor that filters Chinese characters.
        Matches Generator.generate from notebook.
        """
        def ban_chinese_logits_processor(token_ids, logits):
            if self.chinese_mask is None:
                self.chinese_mask = self._build_chinese_mask(logits).to(logits.device)
            
            # Mask the score by -inf
            logits[:, self.chinese_mask] = -float("inf")
            return logits

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Uses the custom logits processor
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            logits_processor=[ban_chinese_logits_processor]
        )

        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
