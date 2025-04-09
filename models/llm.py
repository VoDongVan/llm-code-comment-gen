from transformers import AutoTokenizer, AutoModel

def load_model(model_name="codellama/CodeLlama-7b-Instruct-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

def generate(model, tokenizer, message):
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    response = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=512)
    response = tokenizer.decode(response[0], skip_special_tokens=True)
    return response

def generate_comments(model, tokenizer, messages, prompt_type="zero-shot"):
    if prompt_type == "expert":
        # Generate expert description
        response = generate(model, tokenizer, messages[0])
        response = {"role": "system", "content": response}
        # Generate comment
        final_message = [response, messages[1]]
        response = generate(model, tokenizer, final_message)
        response = {"role": "assistant", "content": response}
    else:
        cur_message = []
        for message in messages:
            cur_message.append(message)
            response = generate(model, tokenizer, cur_message)
            response = {"role": "assistant", "content": response}
            cur_message.append(response)
    comments = response["content"]
    return comments