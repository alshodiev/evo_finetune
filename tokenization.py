from transformers import AutoTokenizer
import torch
from torch.nn import MultiheadAttention as MHA

def load_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer for model '{model_name}'. Error: {e}")

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = 0

    # Ensure the tokenizer has the necessary tokens
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer does not have a valid pad token.")

    return tokenizer

def tokenize_sequences(tokenizer, sequences, max_length=91):
    # Check for empty sequences list
    if not sequences:
        raise ValueError("The sequences list is empty.")

    # Ensure all elements in sequences are strings
    for seq in sequences:
        if not isinstance(seq, str):
            raise TypeError(f"All sequences must be strings. Found: {type(seq)}")

    # Check for overly long sequences (before tokenization)
    if any(len(seq) > max_length for seq in sequences):
        raise ValueError(f"Some sequences exceed the maximum length of {max_length} characters.")

    try:
        return tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    except Exception as e:
        raise RuntimeError(f"Tokenization failed. Error: {e}")
'''
def convert_to_tensor(data, dtype=torch.float32):
    # Ensure data is convertible to a tensor
    try:
        tensor = torch.tensor(data, dtype=dtype).unsqueeze(1)
    except Exception as e:
        raise ValueError(f"Failed to convert data to tensor. Data type: {type(data)}, Error: {e}")

    return tensor
'''

