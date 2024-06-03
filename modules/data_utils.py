import os
import json
import numpy as np

# トークンとIDの対応付け
tokens = ["(", ")", "[", "]", "{", "}", "input", ",output", ","]
token2id = {token: i for i, token in enumerate(tokens)}

def load_dataset(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def prepare_sequences(encoded_tokens, seq_length):
    input_sequences = []
    target_tokens = []
    for i in range(len(encoded_tokens) - seq_length):
        input_sequences.append(encoded_tokens[i:i+seq_length])
        target_tokens.append(encoded_tokens[i+seq_length])
    return np.array(input_sequences), np.array(target_tokens)
