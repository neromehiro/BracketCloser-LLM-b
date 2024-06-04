import numpy as np
import json
import os
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.activations import softmax
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from modules.model_utils import CustomMultiHeadAttention

# ログ設定
logging.basicConfig(filename='inference_validation_log.txt', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# トークンとIDを対応付ける辞書
tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]
# トークンとIDを対応付ける辞書
token2id = {token: i + 1 for i, token in enumerate(tokens)}

# IDとトークンを対応付ける辞書
id2token = {i + 1: token for i, token in enumerate(tokens)}

# ディレクトリ設定(コメントアウトしているのも削除しないで!!)
# model_save_path = "./models/trash/best_model_lstm.h5"
# model_save_path = "./models/trash/temp_model.h5"
model_save_path = "./models/trash/best_model_gpt.h5"
# model_save_path = "./models/残すモデル/best_model.h5"
# model_save_path = "./models/残すモデル/best_modellstm_model10hour.h5"
model_metadata_path = "./models/training_info.json"

# テスト用データ
test_data = {
    "input": "input:(){({}){【】【】(){}{,output",
    "output": "}}}"
}

# decode_output 関数の定義
def decode_output(output_seq):
    decoded = "".join([id2token[int(id)] for id in output_seq if int(id) in id2token and int(id) != 0])
    return decoded

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

# モデルのロード
model = load_model(model_save_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})
model_type = get_model_type(model_metadata_path)

# モデルのコンパイル
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True))

# トークンをIDにエンコード
encoded_input = [token2id[token] for token in test_data["input"] if token in token2id]
encoded_output = [token2id[token] for token in test_data["output"] if token in token2id]

# シーケンス長の取得
if isinstance(model.input_shape, list):
    seq_length = model.input_shape[0][1]
else:
    seq_length = model.input_shape[1]

# パディング
preprocessed_input = pad_sequences([encoded_input], maxlen=seq_length, padding='post', value=0)[0]

# モデルの入力数を確認
num_model_inputs = len(model.input_shape)

# 推論
predicted_output_ids = []
for i in range(len(encoded_output)):
    if num_model_inputs == 2:
        attention_mask = np.ones((1, seq_length))
        model_inputs = [np.array([preprocessed_input]), attention_mask]
    else:
        model_inputs = np.array([preprocessed_input])
    
    predicted_output = model.predict(model_inputs)
    
    # 出力がタプルの場合、その形状を確認し適切に処理する
    if isinstance(predicted_output, tuple):
        predicted_output = predicted_output[0]
    
    # ソフトマックスを適用して確率分布を得る
    predicted_output_softmax = softmax(tf.convert_to_tensor(predicted_output)).numpy()
    predicted_id = np.argmax(predicted_output_softmax, axis=-1).item()
    predicted_output_ids.append(predicted_id)
    
    # ログに出力結果を記録
    logging.debug(f"Step {i+1}")
    logging.debug(f"Predicted raw output: {predicted_output}")
    logging.debug(f"Predicted softmax output: {predicted_output_softmax}")
    logging.debug(f"Predicted token id: {predicted_id}")

    if len(preprocessed_input) < seq_length:
        preprocessed_input = np.concatenate([preprocessed_input, [predicted_id]])
    else:
        preprocessed_input = np.roll(preprocessed_input, -1)
        preprocessed_input[-1] = predicted_id

predicted_output = decode_output(predicted_output_ids)
expected_output_reconstructed = decode_output(encoded_output)

print("Predicted Output:", predicted_output)
print("Expected Output:", expected_output_reconstructed)

# ログに最終出力を記録
logging.debug(f"Final Predicted Output: {predicted_output}")
logging.debug(f"Expected Output: {expected_output_reconstructed}")
