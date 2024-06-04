# modules/objective.py
import numpy as np
import os
from modules.model_utils import define_gru_model, define_transformer_model, define_lstm_model, define_bert_model, define_gpt_model
from modules.data_utils import load_dataset, prepare_sequences, tokens
from modules.training_utils import train_model

MODEL_ARCHITECTURES = {
    "gru": define_gru_model,
    "transformer": define_transformer_model,
    "lstm": define_lstm_model,
    "bert": define_bert_model,
    "gpt": define_gpt_model
}

def objective(trial, architecture, best_loss, encode_dir_path, create_save_folder, trial_number):
    model_architecture_func = MODEL_ARCHITECTURES[architecture]
    
    # 最初のトライアルの場合、エポック数を制限
    if trial_number == 0:
        epochs = trial.suggest_int("epochs", 3, 10)
    else:
        epochs = trial.suggest_int("epochs", 1, 100)
    
    batch_size = trial.suggest_int("batch_size", 32, 1024)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    seq_length = 1  # 固定値に設定

    # モデル固有のハイパーパラメータ
    if architecture == "gru":
        embedding_dim = trial.suggest_int("embedding_dim", 32, 256)
        gru_units = trial.suggest_int("gru_units", 32, 256)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        recurrent_dropout_rate = trial.suggest_float("recurrent_dropout_rate", 0.0, 0.5)
        model = model_architecture_func(seq_length, len(tokens) + 1, embedding_dim, gru_units, dropout_rate, recurrent_dropout_rate, learning_rate)
    
    elif architecture == "transformer":
        embedding_dim = trial.suggest_int("embedding_dim", 32, 256)
        num_heads = trial.suggest_int("num_heads", 2, 8)
        ffn_units = trial.suggest_int("ffn_units", 64, 512)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        model = model_architecture_func(seq_length, len(tokens) + 1, embedding_dim, num_heads, ffn_units, dropout_rate, learning_rate)
    
    elif architecture == "lstm":
        embedding_dim = trial.suggest_int("embedding_dim", 32, 512)
        lstm_units = trial.suggest_int("lstm_units", 32, 512)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        recurrent_dropout_rate = trial.suggest_float("recurrent_dropout_rate", 0.0, 0.5)
        num_layers = trial.suggest_int("num_layers", 1, 5)
        model = model_architecture_func(seq_length, len(tokens) + 1, embedding_dim, lstm_units, dropout_rate, recurrent_dropout_rate, num_layers, learning_rate)
    
    elif architecture == "bert":
        embedding_dim = trial.suggest_int("embedding_dim", 32, 256)
        num_heads = trial.suggest_int("num_heads", 2, 8)
        ffn_units = trial.suggest_int("ffn_units", 64, 512)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        model = model_architecture_func(seq_length, len(tokens) + 1, embedding_dim, num_heads, ffn_units, num_layers, dropout_rate, learning_rate)
    
    elif architecture == "gpt":
        embedding_dim = trial.suggest_int("embedding_dim", 32, 256)
        num_heads = trial.suggest_int("num_heads", 2, 8)
        ffn_units = trial.suggest_int("ffn_units", 64, 512)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
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
        return float('inf')

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    save_path = create_save_folder()
    temp_model_path = os.path.join(save_path, f"temp_model_{trial.number}.h5")

    # モデルの訓練
    history, dataset_size = train_model(
        model, 
        all_input_sequences, 
        all_target_tokens, 
        epochs=epochs, 
        batch_size=batch_size, 
        model_path=temp_model_path, 
        num_files=num_files, 
        learning_rate=learning_rate, 
        architecture=architecture, 
        model_architecture_func=model_architecture_func  # 追加された引数
    )
    
    # ベストモデルの評価
    if isinstance(history, list):
        print("Training failed. Returning inf loss.")
        return float('inf')
    else:
        loss = history.history['loss'][-1]
        if loss < best_loss:
            best_loss = loss
            model.save(os.path.join(save_path, "best_model.h5"))
            print(f"New best model saved with loss: {best_loss}")
        return loss
