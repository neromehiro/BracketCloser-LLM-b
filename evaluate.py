import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention
from typing import List

# ディレクトリ設定
dirs = {
    "original": "./dataset/original",
    "tokenize": "./dataset/tokenize",
    "preprocessed": "./dataset/preprocessed",
}

# モデルの保存パス
model_save_path = "./models/best_model.h5"

# テストデータの保存パス
test_data_path = os.path.join(dirs["original"], "test_bracket_dataset.json")

# 評価結果の保存パス
evaluation_result_path = "evaluation_result.txt"

# トークンとIDを対応付ける辞書
tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]
token2id = {token: i for i, token in enumerate(tokens)}
id2token = {i: token for token, i in token2id.items()}

# CustomMultiHeadAttentionの定義
class CustomMultiHeadAttention(MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False, training=None, **kwargs):
        if isinstance(attention_mask, list):
            attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.float32)
        return super().call(query, value, key=key, attention_mask=attention_mask, return_attention_scores=return_attention_scores, training=training)

# モデルのロード
model = load_model(model_save_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})

def load_dataset(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset

def tokenize_string(string: str) -> List[str]:
    tokens = []
    current_token = ""
    for char in string:
        if char in token2id:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            current_token += char
    if current_token:
        tokens.append(current_token)
    return tokens

def preprocess_input(input_seq: str) -> List[int]:
    tokens = tokenize_string(input_seq)
    return [token2id[token] for token in tokens if token in token2id]

def decode_output(output_seq: List[int]) -> str:
    return "".join([id2token[id] for id in output_seq if id in id2token])

def evaluate_model(model, test_data: List[str]):
    correct_predictions = 0
    results = []

    for idx, data in enumerate(test_data):
        input_seq = data.split(",")[0].split(":")[1].strip()
        expected_output = data.split(",")[1].split(":")[1].strip()

        # Preprocessing
        preprocessed_input = preprocess_input(input_seq)
        preprocessed_input = np.array(preprocessed_input).reshape(1, -1)

        expected_output_tokens = preprocess_input(expected_output)
        
        # モデルに input と expected_output の最初の部分を与える
        for token in expected_output_tokens[:len(preprocessed_input[0])]:
            preprocessed_input = np.append(preprocessed_input, [[token]], axis=1)

        predicted_output_ids = []
        for i in range(len(expected_output_tokens) - len(preprocessed_input[0])):
            # Create attention_mask based on the current input length
            attention_mask = np.ones((1, preprocessed_input.shape[1]), dtype=np.float32)

            # Make prediction
            predicted_output = model.predict([preprocessed_input, attention_mask])
            predicted_id = np.argmax(predicted_output, axis=-1).flatten()[0]
            predicted_output_ids.append(predicted_id)
            preprocessed_input = np.append(preprocessed_input, [[predicted_id]], axis=1)

        predicted_output = decode_output(predicted_output_ids)
        expected_output_reconstructed = decode_output(expected_output_tokens[len(preprocessed_input[0]):])
        
        if predicted_output == expected_output_reconstructed:
            results.append(f"問題{idx + 1} 正解\n入力した単語 Input: {input_seq}\n出力の単語: {predicted_output}\n正解の単語: {expected_output_reconstructed}\n")
            correct_predictions += 1
        else:
            results.append(f"問題{idx + 1} 不正解\n入力した単語 Input: {input_seq}\n出力の単語: {predicted_output}\n正解の単語: {expected_output_reconstructed}\n")

    accuracy = correct_predictions / len(test_data)

    # 結果をファイルに保存
    with open(evaluation_result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
        f.write(f"\nAccuracy: {accuracy * 100:.2f}%")

    return accuracy

# テストデータのロード
test_data = load_dataset(test_data_path)

# モデルの評価
accuracy = evaluate_model(model, test_data)
print(f"モデルの精度: {accuracy * 100:.2f}%")
print(f"評価結果は {evaluation_result_path} に保存されました。")
