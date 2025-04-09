from datasets import load_dataset
from collections import defaultdict
import json
import os

def the_vault_function_to_json(
    split_set="train",
    languages=["java"],
    streaming=True,
    num_proj=None,
    num_method=None,
    write_to_file=False,
    output_dir="data/vault_function_json"
):
    assert not (num_proj and num_method), "Cannot use both num_proj and num_method"

    if write_to_file:
        os.makedirs(output_dir, exist_ok=True)

    data_lang = {}
    for lang in languages:
        dataset = load_dataset("Fsoft-AIC/the-vault-function", split_set=[split_set], languages=[lang], streaming=streaming)
        if split_set == "train/small":
            split_set = "train_small"
        elif split_set == "train/medium":
            split_set = "train_medium"
        dataset = dataset[split_set]
        print(f"Loaded {lang}, split={split_set}, streaming={streaming}")

        data = []
        proj_counts = defaultdict(int)
        seen_projects = set()
        for sample in dataset:
            if num_proj and len(seen_projects) >= num_proj:
                break
            if num_method and len(data) >= num_method:
                break
            item = {
                "code": sample["code"],
                "docstring": sample["docstring"],
                "project": sample["repo"],
                "name": sample["identifier"]
            }
            data.append(item)
            project = item["project"]
            if num_proj:
                seen_projects.add(project)
            proj_counts[project] += 1

        print(f"{lang} - collected {len(data)} samples across {len(seen_projects)} projects")

        if write_to_file:
            output_path = os.path.join(output_dir, f"{lang}_{split_set}.jsonl")
            with open(output_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            print(f"Saved to {output_path}")
        data_lang[lang] = data
    return data_lang
