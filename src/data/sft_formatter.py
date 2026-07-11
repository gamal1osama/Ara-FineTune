"""
SFT formatter to convert intermediate JSONL records to LLaMA-Factory alpaca format,
and perform train/val split.
Extracted from notebook: Format Finetuning Datasets section.
"""
import json
import os
import random


def format_to_alpaca(sft_jsonl_path: str, shuffle: bool = True, seed: int = 101) -> list:
    """
    Reads intermediate SFT JSONL file and formats each record into LLaMA-Factory
    compatible chat template format (instruction, input, output, system, history).
    """
    alpaca_data = []

    system_message = "\n".join([
        "You are a professional NLP data parser.",
        "Follow the provided `Task` by the user and the `Output Scheme` to generate the `Output JSON`.",
        "Do not generate any introduction or conclusion."
    ])

    with open(sft_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            record = json.loads(line_str)

            alpaca_data.append({
                "system": system_message,
                "instruction": "\n".join([
                    "# Story:",
                    record["story"],
                    "",
                    "# Task:",
                    record["task"],
                    "",
                    "# Output Scheme:",
                    record["output_scheme"],
                    "",
                    "# Output JSON:",
                    "```json"
                ]),
                "input": "",
                "output": "\n".join([
                    "```json",
                    json.dumps(record["response"], ensure_ascii=False, default=str),
                    "```"
                ]),
                "history": []
            })

    if shuffle:
        random.Random(seed).shuffle(alpaca_data)

    return alpaca_data


def split_and_save(alpaca_data: list, train_size: int = 2700, output_dir: str = None):
    """
    Splits the alpaca list into train and val splits and saves them as JSON files.
    """
    if output_dir is None:
        raise ValueError("output_dir must be provided.")

    os.makedirs(output_dir, exist_ok=True)

    train_data = alpaca_data[:train_size]
    eval_data = alpaca_data[train_size:]

    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")

    with open(train_path, "w", encoding="utf-8") as dest:
        json.dump(train_data, dest, ensure_ascii=False, default=str)

    with open(val_path, "w", encoding="utf-8") as dest:
        json.dump(eval_data, dest, ensure_ascii=False, default=str)

    print(f"Saved {len(train_data)} train samples to {train_path}")
    print(f"Saved {len(eval_data)} validation samples to {val_path}")
