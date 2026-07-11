"""
Dataset loader for JSONL raw data files.
Extracted from notebook: Knowledge Distillation section.
"""
import json
import random


def load_jsonl(file_path: str, shuffle: bool = True, seed: int = 101) -> list:
    """
    Loads JSONL file and optionally shuffles it using a fixed random seed.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            data.append(json.loads(line_str))
            
    if shuffle:
        random.Random(seed).shuffle(data)
        
    return data
