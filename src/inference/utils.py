"""
Shared inference utilities.
parse_json is used by every inference path (base, finetuned, vllm, openai).
"""
import json_repair


def parse_json(text: str):
    """
    Attempt to parse (and repair) a JSON string.
    Returns the parsed dict/list, or None on failure.
    """
    try:
        return json_repair.loads(text)
    except Exception:
        return None
