# modules/data_utils.py
import os
import json
import numpy as np

# トークンとIDの対応付け
# 括弧の種類とキーワード
tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]

# トークンとIDを対応付ける辞書
token2id = {token: i + 1 for i, token in enumerate(tokens)}

# IDとトークンを対応付ける辞書
id2token = {i + 1: token for i, token in enumerate(tokens)}

def load_dataset(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def prepare_sequences(encoded_tokens, seq_length):
    input_sequences = []
    target_tokens = []
    for i in range(len(encoded_tokens) - seq_length):
        input_sequences.append(encoded_tokens[i : i + seq_length])

        # ターゲットトークンとして、入力シーケンスの後に続く全ての括弧を含める
        j = i + seq_length
        while j < len(encoded_tokens) and encoded_tokens[j] in [1, 3, 5]:  # 1, 3, 5 は閉じ括弧のトークンID
            target_tokens.append(encoded_tokens[j])
            j += 1

    return np.array(input_sequences), np.array(target_tokens)

