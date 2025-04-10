def format_inline_prompt(prev_context, next_context):
    return f"""This is the previous context before a blank line:\n{prev_context}\n\n\
            This is the next context after a blank line:\n{next_context}\n\n\
            Generate a helpful inline comment that should be insert in the blank line"""

def format_code_prompt(code):
    return f"""Generate helpful docstrings for the following code snippet:\n {code}"""

def zero_shot_prompt(input: object, type: str):
    if type == "inline":
        prev_context = input["prev_context"]
        next_context = input["next_context"]
        prompt = format_inline_prompt(prev_context, next_context)
    else:
        code = input["code"]
        prompt = format_code_prompt(code)
    prompt = {"role": "user", "content": prompt}
    return [prompt]

def few_shot_prompt(input: object, few_shot_examples: list, type: str):
    prompt = f""""""
    if type == "inline":
        prev_context = input["prev_context"]
        next_context = input["next_context"]
        for example in few_shot_examples:
            prev = example["prev_context"]
            next = example["next_context"]
            comment = example["comment"]
            prompt += f"Example:\nPrevious context:\n{prev}\n\nNext context:\n{next}\n\nComment:\n{comment}\n\n---\n\n"
        prompt = f"""Here are examples of comments for code snippet:\n{prompt}\n\n\
            """ + format_inline_prompt(prev_context, next_context)
    else:
        code = input["code"]
        for example in few_shot_examples:
            code_fs = example["code"]
            comment = example["comment"]
            prompt += f"Example:\nFunction:\n{code_fs}\n\nComment:\n{comment}\n\n---\n\n"
        prompt = f"""Here are examples of comments for code snippet:\n{prompt}\n\n\
            """ + format_code_prompt(code)
    prompt = {"role": "user", "content": prompt}
    return [prompt]

def cot_prompt(input: object, type: str):
    if type == "class":
        code = input["code"]
        prompt1 = f"""Analyze the following code snippet:\n{code}\n\n
            Answer these questions:\n\
            1. What are class variables?
            2. What are the purposes of these variables?
            3. What are public methods?
            4. What are the functionalities of these public methods?
            5. What are private methods?
            6. What are the functionalities of these private methods?
            """
    elif type == "function":
        code = input["code"]
        prompt1 = f"""Analyze the following code snippet:\n{code}\n\n\
            Answer these questions:\n\
            1. What are the inputs?
            2. What are the outputs?
            3. What functions are called by this function?
            4. What are the main functionalities of this function?
            5. Are there any constraints or specific conditions that a user should be aware of?
            """
    else:
        prev_context = input["prev_context"]
        next_context = input["next_context"]
        prompt1 = f"""This is the previous context of a line:\n{prev_context}\n\n\
            This is the next context of a line:\n{next_context}\n\n\
            Answer these questions:\n\
            1. What are the variables and functions used in previous context?
            2. What does the previous context do?
            3. What are the variables and functions used in next context?
            4. What does the next context do?
            5. Should comments be written here, between these 2 contexts?
            """
    prompt2 = f"""From the previous answers, generate helpful comments for the code"""
    prompt1 = {"role": "user", "content": prompt1}
    prompt2 = {"role": "user", "content": prompt2}
    return [prompt1, prompt2]

def critique_prompt(input: object, type: str):
    if type == "inline":
        prev_context= input["prev_context"]
        next_context = input["next_context"]
        prompt1 = format_inline_prompt(prev_context, next_context)
    else:
        code = input["code"]
        prompt1 = format_code_prompt(code)
    prompt2 = f"""Review and find problems with your answer"""
    prompt3 = f"""Based on the problems, improve your answer"""
    prompt1 = {"role": "user", "content": prompt1}
    prompt2 = {"role": "user", "content": prompt2}
    prompt3 = {"role": "user", "content": prompt3}
    return [prompt1, prompt2, prompt3]

def expert_prompt(input: object, type: str):
    prompt1 = f"""Describe an experienced software engineer that can write high-quality code comments in second person perspective:\n"""
    if type == "inline":
        prev_context = input["prev_context"]
        next_context = input["next_context"]
        prompt2 = format_inline_prompt(prev_context, next_context)
    else:
        code = input["code"]
        prompt2 = format_code_prompt(code)
    prompt1 = {"role": "user", "content": prompt1}
    prompt2 = {"role": "user", "content": prompt2}
    return [prompt1, prompt2]

PROMPT_BUILDERS = {
    "zero-shot": lambda input, data_type, few_shot_examples: zero_shot_prompt(input, data_type),
    "few-shot": few_shot_prompt,
    "cot": lambda input, data_type, few_shot_examples: cot_prompt(input, data_type),
    "critique": lambda input, data_type, few_shot_examples: critique_prompt(input, data_type),
    "expert": lambda input, data_type, few_shot_examples: expert_prompt(input, data_type)
}

def prepare_prompt(input: object, data_type: str, few_shot_examples=None, prompt_type="zero-shot"):
    if prompt_type not in PROMPT_BUILDERS:
        raise ValueError(f"Prompt strategy '{prompt_type}' not supported.")
    return PROMPT_BUILDERS[prompt_type](input, data_type, few_shot_examples)