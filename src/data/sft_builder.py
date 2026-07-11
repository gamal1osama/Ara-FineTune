"""
SFT builder to generate synthetic instructions/labels using OpenAI model.
Extracted from notebook: Knowledge Distillation section.
"""
import json
import os
from tqdm.auto import tqdm

from src.models.openai_model import OpenAIModel
from src.schemas.news_schema import NewsDetails
from src.schemas.translation_schema import Translation
from src.tasks.news_extraction_task import build_details_extraction_messages
from src.tasks.translation_task import build_translation_messages
from src.inference.utils import parse_json


def generate_details_extraction_sft(
    raw_data: list,
    model: OpenAIModel,
    save_to: str,
    cloud_model_id: str = "openai/gpt-oss-120b:free",
    price_per_1m_input: float = 0.0,
    price_per_1m_output: float = 0.0
):
    """
    Run distillation for details extraction task.
    """
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    
    prompt_tokens = 0
    completion_tokens = 0
    ix = 0

    print(f"Generating details extraction SFT for {len(raw_data)} items...")
    for item in tqdm(raw_data):
        story_content = item.get("content", "").strip()
        if not story_content:
            continue

        messages = build_details_extraction_messages(story_content)
        
        try:
            content, raw_resp = model.generate_with_usage(
                messages=messages,
                model=cloud_model_id,
                temperature=0.2
            )
        except Exception as e:
            print(f"Error calling model for item: {e}")
            continue

        if raw_resp.choices[0].finish_reason != "stop":
            prompt_tokens += raw_resp.usage.prompt_tokens
            continue

        llm_resp_dict = parse_json(content)
        if not llm_resp_dict:
            continue

        record = {
            "id": ix,
            "story": story_content,
            "task": "Extrat the story details into a JSON.",
            "output_scheme": json.dumps(NewsDetails.model_json_schema(), ensure_ascii=False),
            "response": llm_resp_dict,
        }

        with open(save_to, "a", encoding="utf-8") as dest:
            dest.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        ix += 1
        prompt_tokens += raw_resp.usage.prompt_tokens
        completion_tokens += raw_resp.usage.completion_tokens

        if (ix % 3) == 0:
            cost_input = (prompt_tokens / 1_000_000) * price_per_1m_input
            cost_output = (completion_tokens / 1_000_000) * price_per_1m_output
            total_cost = cost_input + cost_output
            print(f"Iteration {ix}: Total Cost = ${total_cost:.4f}")


def generate_translation_sft(
    raw_data: list,
    model: OpenAIModel,
    save_to: str,
    languages: list = None,
    cloud_model_id: str = "openai/gpt-oss-120b:free",
    price_per_1m_input: float = 0.0,
    price_per_1m_output: float = 0.0
):
    """
    Run distillation for translation task.
    """
    if languages is None:
        languages = ["English", "French"]
        
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    
    prompt_tokens = 0
    completion_tokens = 0
    ix = 0

    print(f"Generating translation SFT for {len(raw_data)} items...")
    for item in tqdm(raw_data):
        story_content = item.get("content", "").strip()
        if not story_content:
            continue

        for lang in languages:
            messages = build_translation_messages(story_content, targeted_lang=lang)
            
            try:
                content, raw_resp = model.generate_with_usage(
                    messages=messages,
                    model=cloud_model_id,
                    temperature=0.2
                )
            except Exception as e:
                print(f"Error calling model for item ({lang}): {e}")
                continue

            if raw_resp.choices[0].finish_reason != "stop":
                prompt_tokens += raw_resp.usage.prompt_tokens
                continue

            llm_resp_dict = parse_json(content)
            if not llm_resp_dict:
                continue

            record = {
                "id": ix,
                "story": story_content,
                "output_scheme": json.dumps(Translation.model_json_schema(), ensure_ascii=False),
                "task": f"You have to translate the story content into {lang} associated with a title into a JSON.",
                "response": llm_resp_dict,
            }

            with open(save_to, "a", encoding="utf-8") as dest:
                dest.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            ix += 1
            prompt_tokens += raw_resp.usage.prompt_tokens
            completion_tokens += raw_resp.usage.completion_tokens

            if (ix % 3) == 0:
                cost_input = (prompt_tokens / 1_000_000) * price_per_1m_input
                cost_output = (completion_tokens / 1_000_000) * price_per_1m_output
                total_cost = cost_input + cost_output
                print(f"Iteration {ix}: Total Cost = ${total_cost:.4f}")
