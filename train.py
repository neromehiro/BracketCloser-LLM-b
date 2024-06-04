# train.py
import os
import sys
import json
import numpy as np
from datetime import datetime
import pytz
from modules.data_utils import load_dataset, prepare_sequences, tokens, token2id
from modules.model_utils import define_gru_model, define_transformer_model, define_lstm_model, define_bert_model, define_gpt_model
from modules.training_utils import train_model, plot_training_history, save_metadata
from modules.custom_layers import CustomMultiHeadAttention

# 日本時間のタイムゾーンを設定
japan_timezone = pytz.timezone("Asia/Tokyo")

# 環境変数設定
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"
# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# データセットの保存先ディレクトリ
encode_dir_path = "./components/dataset/preprocessed/"

# モデル保存先ディレクトリ
model_save_path = "./models/"

MODEL_ARCHITECTURES = {
    "gru": define_gru_model,
    "transformer": define_transformer_model,
    "lstm": define_lstm_model,
    "bert": define_bert_model,
    "gpt": define_gpt_model
}

SHORTCUTS = {
    "gru": "gru",
    "tra": "transformer",
    "lstm": "lstm",
    "ber": "bert",
    "gpt": "gpt"
}

TRAINING_MODES = {
    "1min": {"epochs": 1, "batch_size": 128, "num_files": 5, "learning_rate": 0.01},
    "10min": {"epochs": 3, "batch_size": 256, "num_files": 10, "learning_rate": 0.01},
    "1hour": {"epochs": 7, "batch_size": 512, "num_files": 50, "learning_rate": 0.001},
    "6hours": {"epochs": 20, "batch_size": 1024, "num_files": 300, "learning_rate": 0.001},
    "12hours": {"epochs": 40, "batch_size": 1024, "num_files": 600, "learning_rate": 0.001},
    "24hours": {"epochs": 80, "batch_size": 1024, "num_files": 1200, "learning_rate": 0.0005},
    "2days": {"epochs": 160, "batch_size": 1024, "num_files": 2400, "learning_rate": 0.0005},
    "4days": {"epochs": 320, "batch_size": 1024, "num_files": 4800, "learning_rate": 0.0005},
}

def select_mode():
    mode = input("Select a mode from: " + ", ".join(TRAINING_MODES.keys()) + "\n")
    while mode not in TRAINING_MODES:
        print(f"Invalid mode. Please select a mode from: {', '.join(TRAINING_MODES.keys())}")
        mode = input()
    return TRAINING_MODES[mode]["epochs"], TRAINING_MODES[mode]["batch_size"], TRAINING_MODES[mode]["num_files"], TRAINING_MODES[mode]["learning_rate"]

def select_mode_and_architecture():
    modes = list(TRAINING_MODES.keys())
    architectures = list(SHORTCUTS.keys())
    choices = [f"{arch} {mode}" for arch in architectures for mode in modes]

    print("以下のモードとアーキテクチャから選んでください。選択肢は英語のまま入力してください：\n")
    
    print("1. GRU (Gated Recurrent Unit)")
    for mode in modes:
        print(f"    - {mode}: gru {mode}")

    print("\n2. Transformer")
    for mode in modes:
        print(f"    - {mode}: tra {mode}")

    print("\n3. LSTM")
    for mode in modes:
        print(f"    - {mode}: lstm {mode}")

    print("\n4. BERT")
    for mode in modes:
        print(f"    - {mode}: ber {mode}")

    print("\n5. GPT")
    for mode in modes:
        print(f"    - {mode}: gpt {mode}")

    choice = input("\nあなたの選択: ")
    
    while choice not in choices:
        print(f"\n無効な選択です。以下の選択肢から選んでください：\n")
        
        print("1. GRU (Gated Recurrent Unit)")
        for mode in modes:
            print(f"    - {mode}: gru {mode}")

        print("\n2. Transformer")
        for mode in modes:
            print(f"    - {mode}: tra {mode}")

        print("\n3. LSTM")
        for mode in modes:
            print(f"    - {mode}: lstm {mode}")

        print("\n4. BERT")
        for mode in modes:
            print(f"    - {mode}: ber {mode}")

        print("\n5. GPT")
        for mode in modes:
            print(f"    - {mode}: gpt {mode}")
        
        choice = input("\nあなたの選択: ")
    
    arch, mode = choice.split()
    architecture = SHORTCUTS[arch]
    return MODEL_ARCHITECTURES[architecture], TRAINING_MODES[mode], architecture

def main():
    model_architecture_func, training_mode, architecture = select_mode_and_architecture()
    epochs = training_mode["epochs"]
    batch_size = training_mode["batch_size"]
    num_files = training_mode["num_files"]
    learning_rate = training_mode["learning_rate"]
    seq_length = 1

    vocab_set = set(tokens)

    all_input_sequences = []
    all_target_tokens = []

    num_datasets = 0

    for dirpath, dirnames, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:  # num_filesに基づいてファイル数を制限
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            for encoded_tokens in encoded_tokens_list:
                num_datasets += 1
                if len(encoded_tokens) > seq_length:
                    input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                    all_input_sequences.extend(input_sequences)
                    all_target_tokens.extend(target_tokens)
                else:
                    print(f"Not enough data in: {file_path}")

    vocab_size = len(vocab_set)
    model = model_architecture_func(seq_length, vocab_size + 1, learning_rate)

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    # 日本時間のタイムゾーンを設定
    start_time = datetime.now(japan_timezone)
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # 一時フォルダ名作成
    temp_save_dir = os.path.join(model_save_path, f"{architecture}_{timestamp}_temp")
    os.makedirs(temp_save_dir, exist_ok=True)

    training_info_path = os.path.join(temp_save_dir, "training_info.json")

    with open(training_info_path, "w") as info_file:
        json.dump({"training_start_time": timestamp}, info_file)

    model_path = os.path.join(temp_save_dir, "best_model.h5")
    plot_path = os.path.join(temp_save_dir, "training_history.png")

    # モデルの学習
    history, dataset_size = train_model(model, all_input_sequences, all_target_tokens, epochs=epochs, batch_size=batch_size, model_path=model_path, num_files=num_files, learning_rate=learning_rate, architecture=architecture, model_architecture_func=model_architecture_func)
    
    if history:
        plot_training_history(history, save_path=plot_path, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, num_files=num_files, dataset_size=dataset_size)

    end_time = datetime.now(japan_timezone)
    training_duration = (end_time - start_time).total_seconds() / 60  # 分単位に変換

    # 実際の学習時間をフォルダ名に反映
    final_save_dir = os.path.join(model_save_path, f"{architecture}_{timestamp}_{int(training_duration)}m")
    os.rename(temp_save_dir, final_save_dir)

    # パスを更新
    training_info_path = os.path.join(final_save_dir, "training_info.json")
    model_path = os.path.join(final_save_dir, "best_model.h5")
    plot_path = os.path.join(final_save_dir, "training_history.png")

    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB単位に変換
    model_params = model.count_params()

    metadata = {
        "training_duration_minutes": training_duration,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_files": num_files,
        "learning_rate": learning_rate,
        "dataset_size": dataset_size,
        "model_size_MB": model_size,
        "model_params": model_params,
        "model_architecture": model_architecture_func.__name__,
        "training_end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(training_info_path, "r") as info_file:
        training_info = json.load(info_file)
    training_info.update(metadata)
    
    with open(training_info_path, "w") as info_file:
        json.dump(training_info, info_file, indent=4)

    print(f"Training finished.")
    print(f"Training duration: {training_duration:.2f} minutes")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Model parameters: {model_params}")

if __name__ == "__main__":
    main()
