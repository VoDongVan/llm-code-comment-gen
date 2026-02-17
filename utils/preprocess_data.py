from datasets import config
print(config.HF_DATASETS_CACHE)

from datasets import load_dataset
from collections import defaultdict
import json
import os
import bm25s
import numpy as np
import torch
import time
from transformers import AutoTokenizer, AutoModel
import faiss

def the_vault_function_to_json(
    split_set="test",
    languages=["java"],
    streaming=True,
    num_proj=None,
    num_method=None,
    min_num_functions=6, #keep only project with at least min_num_functions functions
    write_to_file=False,
    output_dir="../data"
):
    """
    Loads and processes functions from The Vault dataset, grouping them by project.

    This function fetches code samples from the `Fsoft-AIC/the-vault-function` dataset
    using the Hugging Face `datasets` library. It groups functions by project and filters 
    them based on a minimum number of functions per project. The output can optionally 
    be saved to disk as JSONL files.

    Args:
        split_set (str): Dataset split to load (e.g., "train", "train/small", "train/medium").
        languages (List[str]): List of programming languages to load (e.g., ["java"]).
        streaming (bool): Whether to stream the dataset instead of loading fully into memory.
        num_proj (int, optional): Maximum number of unique projects to collect. Cannot be used with `num_method`.
        num_method (int, optional): Maximum number of individual methods/functions to collect. Cannot be used with `num_proj`.
        min_num_functions (int): Minimum number of functions a project must have to be included.
        write_to_file (bool): Whether to write the processed data to JSONL files.
        output_dir (str): Directory where JSONL files will be saved if `write_to_file` is True.

    Returns:
        dict: A dictionary mapping each language to a list of project entries. 
              Each entry is a dict with keys `"project_name"` and `"functions"`, 
              where `"functions"` is a list of function metadata.
              
    Raises:
        AssertionError: If both `num_proj` and `num_method` are provided.
    """
    
    assert not (num_proj and num_method), "Cannot use both num_proj and num_method"

    if write_to_file:
        os.makedirs(output_dir, exist_ok=True)

    data_lang = {}
    for lang in languages:
        dataset = load_dataset("Fsoft-AIC/the-vault-function", 
                               split_set=[split_set], 
                               languages=[lang], 
                               streaming=False,
                               trust_remote_code=True)
        if split_set == "train/small":
            split_set = "train_small"
        elif split_set == "train/medium":
            split_set = "train_medium"
        dataset = dataset[split_set]
        print(f"Loaded {lang}, split={split_set}, streaming={streaming}")

        data = []
        project_function = []
        project = ""
        proj_counts = defaultdict(int)
        seen_projects = set()
        total = 0
        i = 0
        for sample in dataset:
            i += 1
            #print(f"{i}: {len(seen_projects)}")
            if num_proj and len(seen_projects) >= num_proj:
                break
            if num_method and len(data) >= num_method:
                break
            item = {
                "code": sample["code"],
                "code_tokens": sample["code_tokens"],
                "docstring": sample["docstring"],
                "project": sample["repo"],
                "name": sample["identifier"]
            }
            # Encounter new project: append project_function to data, refresh project_function
            if project != item["project"] and project != "":
                if len(project_function) > min_num_functions:
                    data.append({"project_name": project, "functions": project_function})
                    total += len(project_function)
                    if num_proj:
                        seen_projects.add(project)
                project_function = []
            project = item["project"]
            project_function.append(item)
            proj_counts[project] += 1

        print(f"{lang} - collected {total} samples across {len(seen_projects)} projects")

        if write_to_file:
            output_path = os.path.join(output_dir, f"{lang}_{split_set}.jsonl")
            with open(output_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            print(f"Saved to {output_path}")
        data_lang[lang] = data
    return data_lang

def the_vault_class_to_json(
    languages=["java"],
    streaming=False,
    num_proj=None,
    num_class=None,
    min_num_classes=5, #keep only project with at least min_num_classes classes
    write_to_file=False,
    output_dir="../data"
):
    """
    Loads and processes classes from The Vault dataset, grouping them by project.

    This function fetches code samples from the `Fsoft-AIC/the-vault-class` dataset
    using the Hugging Face `datasets` library. It groups classes by project and filters 
    them based on a minimum number of classes per project. The output can optionally 
    be saved to disk as JSONL files.

    Args:
        split_set (str): Dataset split to load (e.g., "train", "validation", "test").
        languages (List[str]): List of programming languages to load (e.g., ["java"]).
        streaming (bool): Whether to stream the dataset instead of loading fully into memory.
        num_proj (int, optional): Maximum number of unique projects to collect. Cannot be used with `num_class`.
        num_class (int, optional): Maximum number of individual classes to collect. Cannot be used with `num_proj`.
        min_num_classes (int): Minimum number of classes a project must have to be included.
        write_to_file (bool): Whether to write the processed data to JSONL files.
        output_dir (str): Directory where JSONL files will be saved if `write_to_file` is True.

    Returns:
        dict: A dictionary mapping each language to a list of project entries. 
              Each entry is a dict with keys `"project_name"` and `"classes"`, 
              where `"classes"` is a list of class metadata.
              
    Raises:
        AssertionError: If both `num_proj` and `num_class` are provided.
    """
    
    assert not (num_proj and num_class), "Cannot use both num_proj and num_class"

    if write_to_file:
        os.makedirs(output_dir, exist_ok=True)

    data_lang = {}
    for lang in languages:
        dataset = load_dataset("/home/quanvo/Documents/van-vo-projects/llm-code-comment-gen/the-vault-class", 
                               languages=[lang], 
                               streaming=streaming,
                               trust_remote_code=True)
        dataset = dataset['train']
        print(f"Loaded {lang}, streaming={streaming}")

        data = []
        project_class = []
        project = ""
        proj_counts = defaultdict(int)
        seen_projects = set()
        i = 0
        total = 0
        for sample in dataset:
            i += 1
            #print(f"{i}: {len(seen_projects)}")
            if num_proj and len(seen_projects) >= num_proj:
                break
            if num_class and len(data) >= num_class:
                break
            item = {
                "code": sample["code"].replace(sample["original_docstring"], ""),
                "code_tokens": sample["code_tokens"],
                "docstring": sample["docstring"],
                "project": sample["repo"],
                "name": sample["identifier"]
            }
            # Encounter new project: append project_class to data, refresh project_class
            if project != item["project"] and project != "":
                if len(project_class) > min_num_classes:
                    data.append({"project_name": project, "classes": project_class})
                    total += len(project_class)
                    if num_proj:
                        seen_projects.add(project)
                project_class = []
            project = item["project"]
            project_class.append(item)
            proj_counts[project] += 1

        print(f"{lang} - collected {total} samples across {len(seen_projects)} projects")

        if write_to_file:
            output_path = os.path.join(output_dir, f"{lang}_{split_set}.jsonl")
            with open(output_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            print(f"Saved to {output_path}")
        data_lang[lang] = data
    return data_lang

import re
import keyword

def split_identifier(token):
    # Split snake_case and then camelCase
    parts = re.split(r'[_]', token)
    split_parts = []
    for part in parts:
        # Split camelCase and PascalCase using lookahead and lookbehind
        camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', part)
        split_parts.extend(camel_parts)
    return split_parts

JAVA_KEYWORDS = {
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
    'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
    'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements',
    'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new',
    'package', 'private', 'protected', 'public', 'return', 'short', 'static',
    'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
    'transient', 'try', 'void', 'volatile', 'while', 'true', 'false', 'null'
}

JAVA_API_COMMON = {
    'System', 'out', 'in', 'err', 'print', 'println', 'printf', 'String', 'Integer',
    'List', 'ArrayList', 'Map', 'HashMap', 'Set', 'HashSet', 'Math', 'Object',
}

PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", ".", "?", "!", ",", ":", "-", "--", "...", ";", "(", ")", 
               "[", "]", "=", ">", "<", "+", "-", "/", "*", ">=", "<=", "==", "+=", "-=", "/=", "*="]

def is_stopword_java(token):
    token_lower = token.lower()
    return (token in JAVA_KEYWORDS or token in JAVA_API_COMMON or token in PUNCTUATIONS)

def process_tokens_java(tokens):
    split_tokens = []
    for token in tokens:
        if re.match(r'^[A-Za-z_]+$', token):
            split_tokens.extend(split_identifier(token))
        else:
            split_tokens.append(token)
    split_tokens = [t for t in split_tokens if not is_stopword_java(t)]
    return split_tokens

#################################PYTHON##############################################
PYTHON_KEYWORDS = set(keyword.kwlist)

PYTHON_BUILTINS = {
    'print', 'input', 'len', 'range', 'open', 'map', 'filter', 'zip', 'int', 'float', 'str',
    'list', 'dict', 'set', 'tuple', 'type', 'isinstance', 'enumerate', 'sorted', 'reversed',
    'sum', 'min', 'max', 'abs', 'any', 'all', 'bool', 'dir', 'divmod', 'id', 'hex', 'oct',
    'ord', 'chr', 'pow', 'round', 'slice', 'vars', 'format', 'next', 'iter'
}

def is_stopword_python(token):
    token_lower = token.lower()
    return (token in PYTHON_KEYWORDS or token in PYTHON_BUILTINS or token in PUNCTUATIONS)

def process_tokens_python(tokens):
    split_tokens = []
    for token in tokens:
        if re.match(r'^[A-Za-z_]+$', token):
            split_tokens.extend(split_identifier(token))
        else:
            split_tokens.append(token)
    split_tokens = [t for t in split_tokens if not is_stopword_python(t)]
    return split_tokens

def process_tokens(tokens, lang):
    match lang:
        case 'java':
            return process_tokens_java(tokens)
        case 'python':
            return process_tokens_python(tokens)
        case _:
            print("Invalid language")
            return None

if __name__ == '__main__':
    try:
        #option 0: inline, 1: functions, 2: classes
        option = 2
        lang = "python"
        
        if option == 0:
            pass
        elif option == 1:
            data_lang = the_vault_function_to_json(split_set="test", 
                                                   languages=[lang], 
                                                   num_proj=200, 
                                                   write_to_file=False)
        elif option == 2:
            data_lang = the_vault_class_to_json(languages=[lang], 
                                            num_proj=200,
                                            streaming=True,
                                            write_to_file=False)
        data = data_lang[lang]
    
        if option == 0:
           data_type = 'snippet'
        elif option == 1:
           data_type = 'functions'
        else:
           data_type = 'classes'
        for i in range(0, len(data)):
           for j in range(0, len(data[i][data_type])):
               tokens = data[i][data_type][j]['code_tokens']
               data[i][data_type][j]['code_tokens_processed'] = process_tokens(tokens, lang)
    
        if option == 0:
            pass
        elif option == 1:
            dataset = load_dataset("Fsoft-AIC/the-vault-function", 
                                   split_set=["train/small"], 
                                   languages=[lang], 
                                   streaming=False,
                                   trust_remote_code=True)
            dataset = dataset["train_small"]
        elif option == 2:
            dataset = load_dataset("/home/quanvo/Documents/van-vo-projects/llm-code-comment-gen/the-vault-class", 
                languages=[lang], 
                streaming=False,
                trust_remote_code=True)
            dataset = dataset["train"]
        
        # UNCOMMENTED TO RUN
        print("create corpus")
        corpus = []
        corpus_tokens = []
        i = 0
        for sample in dataset:
            if i % 10000 == 0:
                print(i)
            i += 1
            corpus.append(sample["code"])
            corpus_tokens.append(process_tokens(sample["code_tokens"], lang))
            if i >= 500000:
                break
        # corpus = [sample["code"] for sample in dataset]
        # corpus_tokens = [process_tokens(sample["code_tokens"], lang) for sample in dataset]
        
        print("indexing...")
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        
        # query_tokens = data[0]['functions'][5]['code_tokens_processed']
        # res, indices = retriever.retrieve([query_tokens], k=5, return_as="tuple_custom")
        # docs = res.documents
        # scores = res.scores
        # print(f"Best result (score: {scores[0, 4]:.2f}): {docs[0, 4]}, index: {indices[0, 4]}")
        
        # UNCOMMENTED TO RUN
        for i in range(len(data)):
            for j in range(len(data[i][data_type])):
                print(f"{i}: {j}")
                query_tokens = data[i][data_type][j]['code_tokens_processed']
                if option == 1:
                    res, indices = retriever.retrieve([query_tokens], k=5, return_as="tuple_custom")
                    indices = indices[0, 0:]
                else:
                    res, indices = retriever.retrieve([query_tokens], k=6, return_as="tuple_custom")
                    indices = indices[0, 1:]
                docs = res.documents
                scores = res.scores
                retrieved_examples = [{'code': dataset[ind.item()]['code'],
                                       'docstring': dataset[ind.item()]['docstring'],
                                       'code_tokens': dataset[ind.item()]['code_tokens'],} for ind in indices]
                data[i][data_type][j]['bm25'] = retrieved_examples
    
        output_dir = '.'
        output_path = None
        if option == 1:
            output_path = os.path.join(output_dir, f"{lang}-test-train-small.jsonl")
            with open(output_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        elif option == 2:
            output_path = os.path.join(output_dir, f"{lang}-class.jsonl")
            with open(output_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        print(f"Saved to {output_path}")
    
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        codes = corpus
        batch_size = 8
        
        # Run through model
        embeddings = []
        i = 0
        print(len(dataset))
        start = time.time()
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]
            print(i)
            inputs = tokenizer(batch_codes, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
                embeddings.append(embedding.cpu().numpy())
        end = time.time()
        print(end - time.time())
        
        embeddings_np = np.array(embeddings, dtype=object)
        embeddings_np = np.vstack(embeddings_np)
        if option == 1:
            pass
            #np.save('embedding_func_{lang}_train_small.npy', embeddings_np, allow_pickle=True)
        elif option == 2:
            pass
            np.save(f'embedding_class_{lang}.npy', embeddings_np, allow_pickle=True)

        import json
        import sys
        import os
        file_name = f'{lang}-class.jsonl'
        data = []
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    json_object = json.loads(line)
                    data.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()} - {e}")

        if option == 1:
            embeddings_np = np.load(f'embedding_func_{lang}_train_small.npy', allow_pickle=True)
        elif option == 2:
            embeddings_np = np.load(f'embedding_class_{lang}.npy', allow_pickle=True)
        embeddings_np = np.vstack(embeddings_np)
        print(embeddings_np.shape)
        print(embeddings_np.dtype)
        print(embeddings_np[0].dtype)
        print(type(embeddings_np[0, 0]))
        faiss.normalize_L2(embeddings_np)
        print("van")
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        print("indexing...")
        index.add(embeddings_np)
        
        query_code = data[0][data_type][5]['code']
        query_input = tokenizer(query_code, padding=True, truncation=True, return_tensors="pt").to(device)
        query_embedding = model(**query_input).last_hidden_state[:, 0, :].detach().cpu().numpy()
        
        k = 5
        distances, indices = index.search(np.array(query_embedding), k)
        
        retrieved_examples = [dataset[i.item()]["code"] for i in indices[0]]
        print(retrieved_examples[0])
        
        for i in range(len(data)):
            for j in range(len(data[i][data_type])):
                print(f"{i}: {j}")
                query_code = data[i][data_type][j]['code']
                query_input = tokenizer(query_code, padding=True, truncation=True, return_tensors="pt").to(device)
                query_embedding = model(**query_input).last_hidden_state[:, 0, :].detach().cpu().numpy()
        
                if option == 1:
                    k = 5
                    distances, indices = index.search(np.array(query_embedding), k)
                    indices = indices[0, 0:]
                else:
                    k = 6
                    distances, indices = index.search(np.array(query_embedding), k)
                    indices = indices[0, 1:]
                retrieved_examples = [{'code': dataset[ind.item()]['code'],
                                       'docstring': dataset[ind.item()]['docstring'],
                                       'code_tokens': dataset[ind.item()]['code_tokens'],} for ind in indices]y
                data[i][data_type][j]['CodeBERT'] = retrieved_examples
        
        output_dir = '.'
        if option == 1:
            output_path = os.path.join(output_dir, f"{lang}-test-(train_small).jsonl")
        elif option == 2:
            output_path = os.path.join(output_dir, f"{lang}-class.jsonl")
        with open(output_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved to {output_path}")
    except Exception as e:
        print(e)
