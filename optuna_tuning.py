import os
import sys
import optuna
import numpy as np
import tensorflow as tf
from modules.data_utils import load_dataset, prepare_sequences, tokens
from modules.model_utils import define_gpt_model
from modules.training_utils import train_model, plot_training_history, save_metadata

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# データセットの保存先ディレクトリ
encode_dir_path = "./dataset/preprocessed/"
model_save_path = "./models/"
study_db_path = "sqlite:///optuna_study.db"  # Optunaの試行結果を保存するSQLiteデータベースのパス


def objective(trial):
    # ハイパーパラメータの定義
    epochs = trial.suggest_int("epochs", 10, 50)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    num_files = trial.suggest_int("num_files", 5, 50)
    
    seq_length = 1
    vocab_set = set(tokens)

    all_input_sequences = []
    all_target_tokens = []

    for dirpath, dirnames, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:  # num_filesに基づいてファイル数を制限
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            for encoded_tokens in encoded_tokens_list:
                if len(encoded_tokens) > seq_length:
                    input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                    all_input_sequences.extend(input_sequences)
                    all_target_tokens.extend(target_tokens)

    vocab_size = len(vocab_set)
    model = define_gpt_model(seq_length, vocab_size + 1, learning_rate)

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    model_path = f"{model_save_path}best_model.h5"

    history, dataset_size = train_model(
        model,
        all_input_sequences,
        all_target_tokens,
        epochs=epochs,
        batch_size=batch_size,
        model_path=f"{model_save_path}trial_{trial.number}_model.h5",  # Save each trial's model
        num_files=num_files,
        learning_rate=learning_rate
    )
    # 目的関数として評価する値を取得
    accuracy = history.accuracies[-1] if history else 0.0  # Handle None return

    return accuracy

def main():
    study = optuna.create_study(direction="maximize", storage=study_db_path, load_if_exists=True)
    study.optimize(objective, timeout=60)  # 10分（600秒）以内で最適化

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 最適なハイパーパラメータで再学習
    best_params = trial.params
    print("最適なハイパーパラメータでモデルを再学習します...")
    
    epochs = best_params["epochs"]
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    num_files = best_params["num_files"]
    
    seq_length = 1
    vocab_set = set(tokens)
    
    all_input_sequences = []
    all_target_tokens = []

    for dirpath, dirnames, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:  # num_filesに基づいてファイル数を制限
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            for encoded_tokens in encoded_tokens_list:
                if len(encoded_tokens) > seq_length:
                    input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                    all_input_sequences.extend(input_sequences)
                    all_target_tokens.extend(target_tokens)

    vocab_size = len(vocab_set)
    model = define_gpt_model(seq_length, vocab_size + 1, learning_rate)

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    model_path = f"{model_save_path}best_model.h5"
    plot_path = f"{model_save_path}training_history.png"

    history, dataset_size = train_model(model, all_input_sequences, all_target_tokens, epochs=epochs, batch_size=batch_size, model_path=model_path, num_files=num_files, learning_rate=learning_rate)
    
    if history:  # Add check to ensure history exists
        plot_training_history(
            history,
            save_path=plot_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_files=num_files,
            dataset_size=dataset_size
        )
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB単位に変換
    model_params = model.count_params()

    metadata = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dataset_size": dataset_size,
        "model_size_MB": model_size,
        "model_params": model_params,
        "model_architecture": "GPT"
    }
    save_metadata(model_path, metadata)

    print(f"Training finished.")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Model parameters: {model_params}")

    # 10回分のTrialの結果を表示
    print("\nTrials results:")
    for i, trial in enumerate(study.trials):
        print(f"Trial {i+1}:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
