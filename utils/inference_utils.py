import sys
import os
import json
import random
from models.models import load_model, generate_comments
from prompts.templates import prepare_prompt

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

path = 'data'
file_name = 'java-test-train-small.jsonl'
file_path = os.path.join(parent_dir, path, file_name)

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_object = json.loads(line)
                data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")
    return data

def generate_comments(
        data,
        model_name = "codellama/CodeLlama-7b-Instruct-hf",
        prompt_types = ["zero-shot", "few-shot-project", "few-shot-codeBERT", "cot", "critique", "expert"]):
    # Load model
    model, tokenizer = load_model(model_name)
    
    # Generate comments
    res = []
    for project in data:
        project_res = {}
        project_res["project_name"] = project["project_name"]
        project_res["functions_res"] = []
        for i in range(0, len(project['functions'])):
            # prepare project few-shot examples: take 5 random examples from the same project and let the model predict
            project_fs_indices = []
            while len(project_fs_indices) < 5:
                ind = random.randint(0, len(project['functions']) - 1)
                if ind != i and ind not in project_fs_indices:
                    project_fs_indices.append(ind)
            project_fs_examples = []
            for ind in project_fs_indices:
                example = {}
                example['code'] = project['functions'][ind]['code']
                example['comment'] = project['functions'][ind]['docstring']
                project_fs_examples.append(example)
            #prepare codeBERT few-shot examples
            codeBERT_examples = project['functions'][i]['CodeBERT']
            for example in codeBERT_examples:
                example['comment'] = example['docstring']

            #Run each prompt type
            # Test different prompt strategies
            
            function_res = {}
            fs_examples = []
            test_input = {"code": project['functions'][i]['code']}
            for prompt_type in prompt_types:
                if prompt_type == "few-shot-project":
                    fs_examples = project_fs_examples
                elif prompt_type == "few-shot-codeBERT":
                    fs_examples = codeBERT_examples
                print(f"\nTesting {prompt_type} Prompt Strategy:")
                print('===========================')
                if prompt_type == "few-shot-project" or prompt_type == "few-shot-codeBERT":
                    arg_prompt_type = "few-shot"
                else:
                    arg_prompt_type = prompt_type
                prompt = prepare_prompt(test_input, data_type="function", prompt_type=arg_prompt_type, few_shot_examples=fs_examples)
                for p in prompt:
                    print(f"{p['role']}: {p['content']}")
                    print('=======================')
                comments = generate_comments(model, tokenizer, prompt, prompt_type=prompt_type)
                function_res[prompt_type] = comments
                print(f"Generated Comments:\n{comments}")
                print('=======================')
            project_res["functions_res"].append(function_res)
        res.append(project_res)