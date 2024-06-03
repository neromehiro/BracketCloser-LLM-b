import os
import json
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention
from typing import List

# ログ設定
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# ディレクトリ設定
dirs = {
    "original": "./dataset/original",
    "tokenize": "./dataset/tokenize",
    "preprocessed": "./dataset/preprocessed",
}

# モデルの保存パス
model_save_path = "./models/best_model.h5"
# model_save_path = "./models/temp_model.h5"

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

# モデルの種類を指定
model_type = "gru"  # ここでモデルの種類を指定します

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
    logging.debug(f"Tokenized string: {tokens}")  # デバッグ: トークン化された文字列をログに記録
    return [token2id[token] for token in tokens if token in token2id]

def decode_output(output_seq: List[int]) -> str:
    decoded = "".join([id2token[id] for id in output_seq if id in id2token])
    logging.debug(f"Decoded output: {decoded}")  # デバッグ: デコードされた出力をログに記録
    return decoded

def split_input_output(data):
    input_output_pairs = []
    for item in data:
        input_seq = item.split(",output:")[0] + ",output"
        output_seq = item.split(",output:")[1]
        input_output_pairs.append((input_seq, output_seq))
    return input_output_pairs

def evaluate_model(model, test_data: List[str], model_type: str):
    correct_predictions = 0
    results = []

    input_output_pairs = split_input_output(test_data)

    for idx, (input_seq, expected_output) in enumerate(input_output_pairs):
        # Preprocessing
        preprocessed_input = preprocess_input(input_seq)
        preprocessed_input = np.array(preprocessed_input).reshape(1, -1)
        
        # デバッグ: 入力シーケンスの確認
        logging.debug(f"Input with output token: {input_seq}")
        logging.debug(f"Preprocessed input: {preprocessed_input}")

        expected_output_tokens = preprocess_input(expected_output)
        logging.debug(f"Expected output tokens: {expected_output_tokens}")  # デバッグ: 期待される出力のトークンをログに記録
        
        # モデルに input と expected_output の最初の部分を与える
        predicted_output_ids = []
        for i in range(len(expected_output_tokens)):
            if model_type in ["transformer", "bert", "gpt"]:
                # Create attention_mask based on the current input length
                attention_mask = np.ones((1, preprocessed_input.shape[1]), dtype=np.float32)
                # デバッグ: マスクの確認
                logging.debug(f"Attention mask: {attention_mask}")
                predicted_output = model.predict([preprocessed_input, attention_mask])
            else:
                predicted_output = model.predict(preprocessed_input)
                
            logging.debug(f"Predicted output raw (step {i}): {predicted_output}")  # デバッグ: 予測結果の確認
            predicted_id = np.argmax(predicted_output, axis=-1).flatten()[0]
            predicted_output_ids.append(predicted_id)
            preprocessed_input = np.append(preprocessed_input, [[predicted_id]], axis=1)

            # デバッグ: 予測結果の確認
            logging.debug(f"Predicted ID (step {i}): {predicted_id}")

        predicted_output = decode_output(predicted_output_ids)
        expected_output_reconstructed = decode_output(expected_output_tokens)
        
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
accuracy = evaluate_model(model, test_data, model_type)
print(f"モデルの精度: {accuracy * 100:.2f}%")
print(f"評価結果は {evaluation_result_path} に保存されました。")