def zero_shot_prompt(code: str, type: str, prev_context=None, next_context=None):
    if type == "inline":
        prompt = f"""This is the previous context of a line:\n{prev_context}\n\n.\
            This is the next context of a line:\n{next_context}\n\n.\
            Generate helpful comments that should be put in the line"""
    else:
        prompt = f"""Generate helpful comments for the following code snippet: {code}"""
    prompt = [{"role": "user", "content": prompt}]
    return [prompt]

def few_shot_prompt(code: str, few_shot_examples: list, type: str, prev_context=None, next_context=None):
    prompt = f""""""
    if type == "inline":
        for example in few_shot_examples:
            prev = example["prev_context"]
            next = example["next_context"]
            comment = example["comment"]
            prompt += f"Previous context:\n {prev}\n\n"
            prompt += f"Next context:\n {next}\n\n"
            prompt += f"Comments:\n {comment}"
        prompt = f"""Here are examples of comments for code snippet:\n{prompt}.\n\n\
            This is the previous context of a line:\n{prev_context}\n\n.\
            This is the next context of a line:\n{next_context}\n\n.\
            Generate helpful comments that should be put in the line"""
    else:
        for example in few_shot_examples:
            code = example["code"]
            comment = example["comment"]
            prompt += f"Function:\n {code}\n\n"
            prompt += f"Comment:\n {comment}\n\n"
        prompt = f"""Here are examples of comments for code snippet:\n{prompt}.\n\n\
            Generate helpful comments for the following code snippet: {code}"""
    prompt = [{"role": "user", "content": prompt}]
    return [prompt]

def cot_prompt(code: str, type: str, prev_context=None, next_context=None):
    if type == "class":
        prompt1 = f"""Analyze the following code snippet:\n{code}\n\n.
            Answer these questions:\n\
            1. What are class variables?
            2. What are the purposes of these variables?
            3. What are public methods?
            4. What are the functionalities of these public methods?
            5. What are private methods?
            6. What are the functionalities of these private methods?
            """
    elif type == "function":
        prompt1 = f"""Analyze the following code snippet:\n{code}\n\n.\
            Answer these questions:\n\
            1. What are the inputs?
            2. What are the outputs?
            3. What functions are called by this function?
            4. What are the main functionalities of this function?
            5. Are there any constraints or specific conditions that a user should be aware of?
            """
    else:
        prompt1 = f"""This is the previous context of a line:\n{prev_context}\n\n.\
            This is the next context of a line:\n{next_context}\n\n.\
            Answer these questions:\n\
            1. What are the variables and functions used in previous context?
            2. What does the previouse context do?
            3. What are the variables and functions used in next context?
            4. What does the next context do?
            5. Should comments be written here, between these 2 contexts?
            """
    prompt2 = f"""From the previous answers, generate helpful comments for the code"""
    prompt1 = [{"role": "user", "content": prompt1}]
    prompt2 = [{"role": "user", "content": prompt2}]
    return [prompt1, prompt2]

def critique(code: str, type: str, prev_context=None, next_context=None):
    if type == "inline":
        prompt1 = f"""This is the previous context of a line:\n{prev_context}\n\n.\
            This is the next context of a line:\n{next_context}\n\n.\
            Generate helpful comments to be put in the line"""
    else:
        prompt1 = f"""Generate helpful comments for the following code snippet: {code}"""
    prompt2 = f"""Review and find problems with your answer"""
    prompt3 = f"""Based on the problems, improve your answer"""
    prompt1 = [{"role": "user", "content": prompt1}]
    prompt2 = [{"role": "user", "content": prompt2}]
    prompt3 = [{"role": "user", "content": prompt3}]
    return [prompt1, prompt2, prompt3]

def expert(code: str, type: str, prev_context=None, next_context=None):
    prompt1 = f"""Describe an experienced software engineer that can write high-quality code comments in second person perspective:\n"""
    if type == "inline":
        prompt2 = f"""This is the previous context of a line:\n{prev_context}\n\n.\
            This is the next context of a line:\n{next_context}\n\n.\
            Generate helpful comments that should be put in the line"""
    else:
        prompt2 = f"""Generate helpful comments for the following code snippet: {code}"""
    prompt1 = [{"role": "user", "content": prompt1}]
    prompt2 = [{"role": "user", "content": prompt2}]
    return [prompt1, prompt2]