# hyper.py
import optuna
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
from modules.setup import setup, parse_time_limit
from modules.objective import objective
from modules.utils import create_save_folder
from modules.data_utils import load_dataset, prepare_sequences, tokens
from modules.training_utils import train_model, plot_training_history, save_metadata

# 訓練データのパス
encode_dir_path = "./components/dataset/preprocessed/"
model_save_base_path = "./models/"
study_name = "model_optimization_study"  # Study name for Optuna
storage_name = "sqlite:///optuna_study.db"  # SQLite database for Optuna

# 初期設定のグローバル変数
model_architecture_func = None
architecture = None
best_loss = float('inf')

# 環境変数設定
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

def main():
    global model_architecture_func, architecture, best_loss
    
    architecture_name = input("Enter the model architecture (gru, transformer, lstm, bert, gpt): ").strip()
    model_architecture_func, architecture = setup(architecture_name)
    
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
        study.optimize(lambda trial: objective(trial, architecture, best_loss, encode_dir_path, lambda: create_save_folder(model_save_base_path, architecture)), timeout=time_limit.total_seconds(), callbacks=[callback])
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
    seq_length = 1  # 固定値に設定

    # モデル固有のベストパラメータの取得
    if architecture == "gru":
        embedding_dim = best_params["embedding_dim"]
        gru_units = best_params["gru_units"]
        dropout_rate = best_params["dropout_rate"]
        recurrent_dropout_rate = best_params["recurrent_dropout_rate"]
        model = model_architecture_func(seq_length, len(tokens) + 1, embedding_dim, gru_units, dropout_rate, recurrent_dropout_rate, learning_rate)
    
    elif architecture == "transformer":
        embedding_dim = best_params["embedding_dim"]
        num_heads = best_params["num_heads"]
        ffn_units = best_params["ffn_units"]
        dropout_rate = best_params["dropout_rate"]
        model = model_architecture_func(seq_length, len(tokens) + 1, embedding_dim, num_heads, ffn_units, dropout_rate, learning_rate)
    
    elif architecture == "lstm":
        embedding_dim = best_params["embedding_dim"]
        lstm_units = best_params["lstm_units"]
        dropout_rate = best_params["dropout_rate"]
        recurrent_dropout_rate = best_params["recurrent_dropout_rate"]
        num_layers = best_params["num_layers"]
        model = model_architecture_func(seq_length, len(tokens) + 1, embedding_dim, lstm_units, dropout_rate, recurrent_dropout_rate, num_layers, learning_rate)
    
    elif architecture == "bert":
        embedding_dim = best_params["embedding_dim"]
        num_heads = best_params["num_heads"]
        ffn_units = best_params["ffn_units"]
        num_layers = best_params["num_layers"]
        dropout_rate = best_params["dropout_rate"]
        model = model_architecture_func(seq_length, len(tokens) + 1, embedding_dim, num_heads, ffn_units, num_layers, dropout_rate, learning_rate)
    
    elif architecture == "gpt":
        embedding_dim = best_params["embedding_dim"]
        num_heads = best_params["num_heads"]
        ffn_units = best_params["ffn_units"]
        dropout_rate = best_params["dropout_rate"]
        model = model_architecture_func(seq_length, len(tokens) + 1, embedding_dim, num_heads, ffn_units, dropout_rate, learning_rate)

    vocab_set = set(tokens)
    all_input_sequences = []
    all_target_tokens = []

    num_datasets = 0
    num_files = 10  # num_filesを増やす

    for dirpath, dirnames, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            for encoded_tokens in encoded_tokens_list:
                num_datasets += 1
                if len(encoded_tokens) > seq_length:  # データ量が不十分な場合のチェックを追加
                    input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                    all_input_sequences.extend(input_sequences)
                    all_target_tokens.extend(target_tokens)
                else:
                    print(f"Not enough data in: {file_path}")

    if not all_input_sequences or not all_target_tokens:
        print("No data for training.")
        return

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    save_path = create_save_folder(model_save_base_path, architecture)

    mirrored_strategy = tf.distribute.MirroredStrategy()  # 分散学習の設定を追加

    with mirrored_strategy.scope():
        model = model_architecture_func(seq_length, len(tokens) + 1, embedding_dim, num_heads, ffn_units, dropout_rate, learning_rate)

    model_path = os.path.join(save_path, "best_model.h5")
    plot_path = os.path.join(save_path, "training_history.png")

    history, dataset_size = train_model(
        model, 
        all_input_sequences, 
        all_target_tokens, 
        epochs=epochs, 
        batch_size=batch_size, 
        model_path=model_path, 
        num_files=num_files, 
        learning_rate=learning_rate, 
        architecture=architecture, 
        model_architecture_func=model_architecture_func  # 追加された引数
    )
    
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