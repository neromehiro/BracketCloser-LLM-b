import optuna
import sys
import os
import numpy as np
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from modules.data_utils import load_dataset, prepare_sequences, tokens
from modules.model_utils import define_gru_model, define_transformer_model, define_lstm_model, define_bert_model, define_gpt_model
from modules.training_utils import train_model, plot_training_history, save_metadata

# 訓練データのパス
encode_dir_path = "./dataset/preprocessed/"
model_save_path = "./models/"
study_name = "model_optimization_study"  # Study name for Optuna
storage_name = "sqlite:///optuna_study.db"  # SQLite database for Optuna
best_model_path = f"{model_save_path}best_model.h5"  # Best model path

# モデルアーキテクチャの辞書
MODEL_ARCHITECTURES = {
    "gru": define_gru_model,
    "transformer": define_transformer_model,
    "lstm": define_lstm_model,
    "bert": define_bert_model,
    "gpt": define_gpt_model
}

# 初期設定のグローバル変数
model_architecture_func = None
architecture = None
best_loss = float('inf')

def setup(architecture_name):
    global model_architecture_func, architecture
    if architecture_name in MODEL_ARCHITECTURES:
        model_architecture_func = MODEL_ARCHITECTURES[architecture_name]
        architecture = architecture_name
    else:
        raise ValueError(f"Unsupported architecture: {architecture_name}")

def parse_time_limit(time_limit_str):
    """時間制限の文字列をtimedeltaに変換する"""
    if 'min' in time_limit_str:
        minutes = int(time_limit_str.replace('min', '').strip())
        return timedelta(minutes=minutes)
    elif 'hour' in time_limit_str:
        hours = int(time_limit_str.replace('hour', '').strip())
        return timedelta(hours=hours)
    else:
        raise ValueError("Unsupported time limit format. Use 'min' or 'hour'.")

def objective(trial):
    global model_architecture_func, architecture, best_loss
    
    # ハイパーパラメータの範囲を設定
    epochs = trial.suggest_int("epochs", 1, 100)
    batch_size = trial.suggest_int("batch_size", 32, 1024)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    seq_length = trial.suggest_int("seq_length", 1, 50)
    
    vocab_set = set(tokens)
    all_input_sequences = []
    all_target_tokens = []

    num_datasets = 0
    num_files = 5  # 最初は小さな数で固定

    for dirpath, dirnames, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            for encoded_tokens in encoded_tokens_list:
                num_datasets += 1
                input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                all_input_sequences.extend(input_sequences)
                all_target_tokens.extend(target_tokens)

    if not all_input_sequences or not all_target_tokens:
        print("No data for training.")
        return float('inf')

    vocab_size = len(vocab_set)
    model = model_architecture_func(seq_length, vocab_size + 1, learning_rate)

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    # 一時的なモデル保存パス
    temp_model_path = f"{model_save_path}temp_model_{trial.number}.h5"

    # モデルの訓練
    history, dataset_size = train_model(model, all_input_sequences, all_target_tokens, epochs=epochs, batch_size=batch_size, model_path=temp_model_path, num_files=num_files, learning_rate=learning_rate, architecture=architecture)
    
    # ベストモデルの評価
    if isinstance(history, list):
        print("Training failed. Returning inf loss.")
        return float('inf')
    else:
        loss = history.history['loss'][-1]
        if loss < best_loss:
            best_loss = loss
            model.save(best_model_path)
            print(f"New best model saved with loss: {best_loss}")
        return loss

def main():
    architecture_name = input("Enter the model architecture (gru, transformer, lstm, bert, gpt): ").strip()
    setup(architecture_name)
    
    time_limit_str = input("Enter the training time limit (e.g., '3min', '1hour', '5hour'): ").strip()
    time_limit = parse_time_limit(time_limit_str)
    start_time = datetime.now()

    # Optunaのストレージを設定
    study = optuna.create_study(
        study_name=study_name, 
        direction="minimize", 
        storage=storage_name, 
        load_if_exists=True
    )
    
    progress_bar = tqdm(total=time_limit.total_seconds(), desc="Optimization Progress", unit="s")

    def callback(study, trial):
        elapsed_time = (datetime.now() - start_time).total_seconds()
        progress_bar.update(elapsed_time - progress_bar.n)
        if elapsed_time >= time_limit.total_seconds():
            progress_bar.close()
            print("Time limit exceeded, stopping optimization.")
            study.stop()

    # トライアルの最適化
    try:
        study.optimize(objective, timeout=time_limit.total_seconds(), callbacks=[callback])
    except Exception as e:
        print(f"An exception occurred during optimization: {e}")
    finally:
        progress_bar.close()

    print("Best hyperparameters: ", study.best_params)
    print("Best loss: ", study.best_value)

    # ベストパラメータで再トレーニングして保存
    best_params = study.best_params
    epochs = best_params["epochs"]
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    seq_length = best_params["seq_length"]

    global model_architecture_func, architecture
    vocab_set = set(tokens)
    all_input_sequences = []
    all_target_tokens = []

    num_datasets = 0
    num_files = 5

    for dirpath, dirnames, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            for encoded_tokens in encoded_tokens_list:
                num_datasets += 1
                input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                all_input_sequences.extend(input_sequences)
                all_target_tokens.extend(target_tokens)

    if not all_input_sequences or not all_target_tokens:
        print("No data for training.")
        return

    vocab_size = len(vocab_set)
    model = model_architecture_func(seq_length, vocab_size + 1, learning_rate)

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    model_path = best_model_path
    plot_path = f"{model_save_path}training_history.png"

    history, dataset_size = train_model(model, all_input_sequences, all_target_tokens, epochs=epochs, batch_size=batch_size, model_path=model_path, num_files=num_files, learning_rate=learning_rate, architecture=architecture)
    
    if history:
        plot_training_history(history, save_path=plot_path, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, num_files=num_files, dataset_size=dataset_size)

    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB単位に変換
    model_params = model.count_params()

    metadata = {
        "epochs": epochs,
        "batch_size": batch_size,
        "num_files": num_files,
        "learning_rate": learning_rate,
        "dataset_size": dataset_size,
        "model_size_MB": model_size,
        "model_params": model_params,
        "model_architecture": model_architecture_func.__name__
    }
    save_metadata(model_path, metadata)

    print(f"Training finished.")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Model parameters: {model_params}")

if __name__ == "__main__":
    main()