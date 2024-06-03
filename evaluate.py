
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from modules.custom_layers import CustomMultiHeadAttention  # CustomMultiHeadAttentionをインポート
from modules.data_utils import load_dataset, prepare_sequences  # 必要なモジュールのインポート
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
preprocessed_test_data_path = os.path.join(dirs["preprocessed"], "test_bracket_dataset.json")

# 評価結果の保存パス
evaluation_result_path = "evaluation_result.txt"

# モデルをロード
model = load_model(model_save_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})

def load_preprocessed_dataset(filepath: str) -> List[list]:
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print("Loaded preprocessed dataset:", dataset)  # データの中身を確認
    return dataset

def data_generator(data):
    for encoded_tokens_list in data:
        input_sequences, target_tokens = prepare_sequences(encoded_tokens_list, seq_length=len(encoded_tokens_list))
        for input_seq, target_token in zip(input_sequences, target_tokens):
            yield input_seq, target_token

def evaluate_model(model, data):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(data),
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).padded_batch(1, padded_shapes=([None], ()))

    true_outputs = []
    predicted_classes = []

    for input_seq, true_output in dataset:
        predicted_output = model.predict(input_seq)
        true_outputs.append(true_output.numpy())
        predicted_classes.append(np.argmax(predicted_output, axis=-1))

    true_outputs = np.concatenate(true_outputs)
    predicted_classes = np.concatenate(predicted_classes)

    accuracy = np.mean(predicted_classes == true_outputs)

    return accuracy

# テストデータのロード
test_data = load_preprocessed_dataset(preprocessed_test_data_path)

# モデルの評価
accuracy = evaluate_model(model, test_data)
print(f"モデルの精度: {accuracy * 100:.2f}%")
print(f"評価結果は {evaluation_result_path} に保存されました。")