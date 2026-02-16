import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

def load_model(model_name="codellama/CodeLlama-7b-Instruct-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, message, max_new_tokens=200):
    device = next(model.parameters()).device
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded_cleaned = re.sub(r"\[INST\].*?\[/INST\]", "", decoded, flags=re.DOTALL).strip()
    return {"role": "assistant", "content": decoded_cleaned}

def generate_comments(model, tokenizer, messages, prompt_type="zero-shot"):
    if prompt_type == "expert":
        # Generate expert description
        expert_response = generate(model, tokenizer, [messages[0]])
        system_message = {"role": "system", "content": expert_response["content"]}
        # Generate comment with system message
        final_messages = [system_message, messages[1]]
        response = generate(model, tokenizer, final_messages)
    else:
        cur_messages = []
        for message in messages:
            cur_messages.append(message)
            response = generate(model, tokenizer, cur_messages)
            cur_messages.append(response)
    return response["content"]

if __name__ == "__main__":
    model, tokenizer = load_model()
    sample_messages = [{"role": "user", "content": "Generate comments for:\ndef add(a, b): return a + b"}]
    comment = generate(model, tokenizer, sample_messages)
    print(comment["content"])