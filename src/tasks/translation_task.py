"""
Prompt builder for the translation task.
Extracted from notebook: Tasks > Translation > translation_messages.
"""
import json

from src.schemas.translation_schema import Translation


def build_translation_messages(story: str, targeted_lang: str = "english") -> list:
    """
    Build the chat messages list for the translation task.

    Args:
        story:         Raw Arabic news article text.
        targeted_lang: Target language string, e.g. "english", "french".

    Returns:
        List of {role, content} dicts ready for apply_chat_template.
    """
    return [
        {
            "role": "system",
            "content": "\n".join([
                "You are a professional translator.",
                "You will be provided by an Arabic text.",
                "You have to translate the text into the `Targeted Language`.",
                "Follow the provided Scheme to generate a JSON",
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
                json.dumps(Translation.model_json_schema(), ensure_ascii=False),
                "",
                "## Targeted Language:",
                targeted_lang,
                "",
                "## Translated Story:",
                "```json",
            ]),
        },
    ]
