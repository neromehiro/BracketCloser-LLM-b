# modules/objective.py
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modules.model_utils import define_gru_model, define_transformer_model, define_lstm_model, define_bert_model, define_gpt_model
from modules.data_utils import load_dataset, tokens
from modules.training_utils import train_model
import wandb
from modules.setup import setup, parse_time_limit
import datetime
import json

MODEL_ARCHITECTURES = {
    "gru": define_gru_model,
    "transformer": define_transformer_model,
    "lstm": define_lstm_model,
    "bert": define_bert_model,
    "gpt": define_gpt_model
}

# 環境変数設定
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

def prepare_sequences(encoded_tokens, seq_length):
    input_sequences = []
    target_tokens = []

    # エンコードされたトークンを用いてシーケンスを作成
    for i in range(1, len(encoded_tokens)):
        input_seq = encoded_tokens[:i]
        target_seq = encoded_tokens[i]
        input_sequences.append(input_seq)
        target_tokens.append(target_seq)

    # シーケンスの長さを揃えるためにパディングを追加
    input_sequences = pad_sequences(input_sequences, maxlen=seq_length, padding='post', value=0)  # パディング値を0に設定
    target_tokens = pad_sequences([target_tokens], maxlen=len(input_sequences), padding='post', value=0)[0]

    return input_sequences, target_tokens



import datetime
import json

def objective(trial, architecture, best_loss, encode_dir_path, create_save_folder_func):
    model_architecture_func = MODEL_ARCHITECTURES[architecture]
    
    if trial.number == 0:
        epochs = trial.suggest_int("epochs", 2, 4)
    else:
        epochs = trial.suggest_int("epochs", 1, 10)
    
    batch_size = trial.suggest_int("batch_size", 32, 256)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    seq_length = 30

    if architecture == "gru":
        embedding_dim = trial.suggest_int("embedding_dim", 32, 256)
        gru_units = trial.suggest_int("gru_units", 32, 256)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        recurrent_dropout_rate = trial.suggest_float("recurrent_dropout_rate", 0.0, 0.5)
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, embedding_dim, gru_units, dropout_rate, recurrent_dropout_rate)
    
    elif architecture == "transformer":
        embedding_dim = trial.suggest_int("embedding_dim", 32, 256)
        num_heads = trial.suggest_int("num_heads", 2, 8)
        ffn_units = trial.suggest_int("ffn_units", 64, 512)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, embedding_dim, num_heads, ffn_units, dropout_rate)
    
    elif architecture == "lstm":
        embedding_dim = trial.suggest_int("embedding_dim", 32, 512)
        lstm_units = trial.suggest_int("lstm_units", 32, 512)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        recurrent_dropout_rate = trial.suggest_float("recurrent_dropout_rate", 0.0, 0.5)
        num_layers = trial.suggest_int("num_layers", 1, 5)
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, embedding_dim, lstm_units, dropout_rate, recurrent_dropout_rate, num_layers)
    
    elif architecture == "bert":
        embedding_dim = trial.suggest_int("embedding_dim", 32, 256)
        num_heads = trial.suggest_int("num_heads", 2, 8)
        ffn_units = trial.suggest_int("ffn_units", 64, 512)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, embedding_dim, num_heads, ffn_units, num_layers, dropout_rate)
    
    elif architecture == "gpt":
        embedding_dim = trial.suggest_int("embedding_dim", 32, 256)
        num_heads = trial.suggest_int("num_heads", 2, 8)
        ffn_units = trial.suggest_int("ffn_units", 64, 512)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, embedding_dim, num_heads, ffn_units, dropout_rate)

    vocab_set = set(tokens)
    all_input_sequences = []
    all_target_tokens = []

    num_datasets = 0
    num_files = 10

    for dirpath, dirnames, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            if encoded_tokens_list is None:
                print(f"Skipping file {file} as it contains no data")
                continue
            for encoded_tokens in encoded_tokens_list:
                num_datasets += 1
                input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                all_input_sequences.extend(input_sequences)
                all_target_tokens.extend(target_tokens)

    if not all_input_sequences or not all_target_tokens:
        print("No data for training.")
        return float('inf')

    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    save_path = create_save_folder_func()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    temp_model_path = os.path.join(save_path, f"temp_model_{trial.number}_{timestamp}.h5")
    os.makedirs(os.path.dirname(temp_model_path), exist_ok=True)

    try:
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
            model_architecture_func=model_architecture_func
        )
        
        if history is None or isinstance(history, float):
            print("Training failed with invalid return. Returning inf loss.")
            return float('inf')
        else:
            loss = history.history['loss'][-1]
            if loss < best_loss:
                best_loss = loss
                best_model_path = os.path.join(save_path, f"best_model_{trial.number}_{timestamp}.h5")
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                model.save(best_model_path)
                print(f"New best model saved with loss: {best_loss}")
                
                metadata = {
                    "epoch": len(history.history['loss']),
                    "logs": {
                        "loss": history.history['loss'][-1],
                        "accuracy": history.history.get('accuracy', [None])[-1],
                        "weighted_accuracy": history.history.get('weighted_accuracy', [None])[-1],
                        "val_loss": history.history.get('val_loss', [None])[-1],
                        "val_accuracy": history.history.get('val_accuracy', [None])[-1],
                        "val_weighted_accuracy": history.history.get('val_weighted_accuracy', [None])[-1]
                    },
                    "time": timestamp,
                    "model_architecture": model_architecture_func.__name__,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "embedding_dim": embedding_dim,
                    "gru_units": gru_units if architecture == 'gru' else None,
                    "dropout_rate": dropout_rate,
                    "recurrent_dropout_rate": recurrent_dropout_rate if architecture == 'gru' else None,
                    "num_layers": num_layers if architecture in ['lstm', 'bert'] else None
                }
                
                metadata_path = os.path.join(save_path, f"metadata_{trial.number}_{timestamp}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
            return loss
    except Exception as e:
        print(f"Training failed with exception: {e}. Returning inf loss.")
        return float('inf')
