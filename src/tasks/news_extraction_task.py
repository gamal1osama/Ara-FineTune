"""
Prompt builder for the news details extraction task.
Extracted from notebook: Tasks > Details Extraction > details_extraction_messages.
"""
import json

from src.schemas.news_schema import NewsDetails


def build_details_extraction_messages(story: str) -> list:
    """
    Build the chat messages list for the details extraction task.

    Args:
        story: Raw Arabic news article text.

    Returns:
        List of {role, content} dicts ready for apply_chat_template.
    """
    return [
        {
            "role": "system",
            "content": "\n".join([
                "You are an NLP data paraser.",
                "You will be provided by an Arabic text associated with a Pydantic scheme.",
                "Generate the ouptut in the same story language.",
                "You have to extract JSON details from text according the Pydantic details.",
                "Extract details as mentioned in text.",
                "Do not generate any introduction or conclusion.",
            ]),
        },
        {
            "role": "user",
            "content": "\n".join([
                "## Story:",
                story.strip(),
                "",
                "## Pydantic Details:",
                json.dumps(NewsDetails.model_json_schema(), ensure_ascii=False),
                "",
                "## Story Details:",
                "```json",
            ]),
        },
    ]
