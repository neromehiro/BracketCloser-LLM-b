# evaluate.py
import sys
import os
import json
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention
from typing import List
from tensorflow.keras.preprocessing.sequence import pad_sequences  # pad_sequencesをインポート

# モジュールのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

import data_generator

# ログ設定
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# ディレクトリ設定
dirs = {
    "original": "./components/dataset/original",
    "tokenize": "./components/dataset/tokenize",
    "preprocessed": "./components/dataset/preprocessed",
}

# モデルの保存パス
model_save_path = "./models/best_model.h5"
model_metadata_path = "./models/training_info.json"

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

# モデルメタデータをロードしてモデルの種類を自動設定
def get_model_type(metadata_path: str) -> str:
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model_architecture = metadata.get("model_architecture", "")
    if "gru" in model_architecture.lower():
        return "gru"
    elif "transformer" in model_architecture.lower():
        return "transformer"
    elif "lstm" in model_architecture.lower():
        return "lstm"
    elif "bert" in model_architecture.lower():
        return "bert"
    elif "gpt" in model_architecture.lower():
        return "gpt"
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture}")

model_type = get_model_type(model_metadata_path)

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



def evaluate_model(model, test_data, model_type):
    correct_predictions = 0
    results = []

    # デフォルトの最大シーケンス長を設定
    default_max_seq_length = 30

    input_output_pairs = split_input_output(test_data)

    for idx, (input_seq, expected_output) in enumerate(input_output_pairs):
        # Preprocessing
        preprocessed_input = preprocess_input(input_seq)

        # 最大シーケンス長を修正: 常にdefault_max_seq_lengthにパディング
        if len(preprocessed_input) > default_max_seq_length:
            preprocessed_input = preprocessed_input[:default_max_seq_length]
        else:
            preprocessed_input = pad_sequences([preprocessed_input], maxlen=default_max_seq_length, padding='post', value=0)[0]

        # デバッグ: 入力シーケンスの確認
        logging.debug(f"Input with output token: {input_seq}")
        logging.debug(f"Preprocessed input: {preprocessed_input}")

        expected_output_tokens = preprocess_input(expected_output)
        logging.debug(f"Expected output tokens: {expected_output_tokens}")

        # モデルに input と expected_output の最初の部分を与える
        predicted_output_ids = []
        for i in range(len(expected_output_tokens)):
            if model_type in ["transformer", "bert", "gpt"]:
                attention_mask = np.ones((1, preprocessed_input.shape[0]), dtype=np.float32)  # attention_maskを修正
                model_input = [np.expand_dims(preprocessed_input, axis=0), attention_mask]  # model_inputを修正
                predicted_output = model.predict(model_input)
            else:
                model_input = np.expand_dims(preprocessed_input, axis=0)  # バッチ次元追加
                predicted_output = model.predict(model_input)

            logging.debug(f"Predicted output raw (step {i}): {predicted_output}")
            predicted_id = np.argmax(predicted_output, axis=-1).flatten()[0]
            predicted_output_ids.append(predicted_id)

            # 入力シーケンスを更新して次の予測に使用 (train.pyのprepare_sequences関数を参考に修正)
            preprocessed_input = np.roll(preprocessed_input, -1)  # 左に1つシフト
            preprocessed_input[-1] = predicted_id  # 最後の要素を更新

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


# テストデータのサンプル数
num_test_samples = 100

# テストデータの生成
test_dataset = data_generator.generate_test_data(num_test_samples)

# テストデータの前処理と保存
data_generator.preprocess_and_save_dataset(test_dataset, "test_bracket_dataset.json")
print("テストデータセットが保存された場所:", data_generator.dirs["original"] + "/test_bracket_dataset.json")

# テストデータのロード
test_data = load_dataset(test_data_path)

# モデルの評価
accuracy = evaluate_model(model, test_data, model_type)
print(f"モデルの精度: {accuracy * 100:.2f}%")
print(f"評価結果は {evaluation_result_path} に保存されました。")
