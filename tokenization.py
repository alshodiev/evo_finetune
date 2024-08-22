from transformers import AutoTokenizer
import torch

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = 0
    return tokenizer

def tokenize_sequences(tokenizer, sequences, max_length=91):
    return tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

def convert_to_tensor(data, dtype=torch.float32):
    return torch.tensor(data, dtype=dtype).unsqueeze(1)