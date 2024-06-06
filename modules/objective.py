# modules/objective.py
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modules.model_utils import define_gru_model, define_transformer_model, define_lstm_model, define_bert_model, define_gpt_model
from modules.data_utils import load_dataset, tokens
from modules.training_utils import train_model
import wandb
from wandb.integration.keras import WandbCallback
from modules.setup import setup, parse_time_limit
import datetime
import json
import tensorflow as tf

MODEL_ARCHITECTURES = {
    "gru": define_gru_model,
    "transformer": define_transformer_model,
    "lstm": define_lstm_model,
    "bert": define_bert_model,
    "gpt": define_gpt_model
}

os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

def prepare_sequences(encoded_tokens, seq_length):
    input_sequences = []
    target_tokens = []

    for i in range(1, len(encoded_tokens)):
        input_seq = encoded_tokens[:i]
        target_seq = encoded_tokens[i]
        input_sequences.append(input_seq)
        target_tokens.append(target_seq)

    input_sequences = pad_sequences(input_sequences, maxlen=seq_length, padding='post', value=0)
    target_tokens = pad_sequences([target_tokens], maxlen=len(input_sequences), padding='post', value=0)[0]

    return input_sequences, target_tokens

def load_training_data(encode_dir_path, seq_length, num_files=10):
    all_input_sequences = []
    all_target_tokens = []

    for dirpath, _, filenames in os.walk(encode_dir_path):
        for file in filenames[:num_files]:
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            if encoded_tokens_list is None:
                continue
            for encoded_tokens in encoded_tokens_list:
                input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                all_input_sequences.extend(input_sequences)
                all_target_tokens.extend(target_tokens)

    return all_input_sequences, all_target_tokens

def objective(trial, architecture, best_loss, encode_dir_path, create_save_folder_func):
    model_architecture_func = MODEL_ARCHITECTURES[architecture]
    
    epochs = trial.suggest_int("epochs", 1, 10) if trial.number != 0 else trial.suggest_int("epochs", 2, 4)
    batch_size = trial.suggest_int("batch_size", 32, 256)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    seq_length = 30

    if architecture == "gru":
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, trial.suggest_int("embedding_dim", 32, 256), trial.suggest_int("gru_units", 32, 256), trial.suggest_float("dropout_rate", 0.0, 0.5), trial.suggest_float("recurrent_dropout_rate", 0.0, 0.5))
    elif architecture == "transformer":
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, trial.suggest_int("embedding_dim", 32, 256), trial.suggest_int("num_heads", 2, 8), trial.suggest_int("ffn_units", 64, 512), trial.suggest_float("dropout_rate", 0.0, 0.5))
    elif architecture == "lstm":
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, trial.suggest_int("embedding_dim", 32, 512), trial.suggest_int("lstm_units", 32, 512), trial.suggest_float("dropout_rate", 0.0, 0.5), trial.suggest_float("recurrent_dropout_rate", 0.0, 0.5), trial.suggest_int("num_layers", 1, 5))
    elif architecture == "bert":
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, trial.suggest_int("embedding_dim", 32, 256), trial.suggest_int("num_heads", 2, 8), trial.suggest_int("ffn_units", 64, 512), trial.suggest_int("num_layers", 1, 4), trial.suggest_float("dropout_rate", 0.0, 0.5))
    elif architecture == "gpt":
        model = model_architecture_func(seq_length, len(tokens) + 1, learning_rate, trial.suggest_int("embedding_dim", 32, 256), trial.suggest_int("num_heads", 2, 8), trial.suggest_int("ffn_units", 64, 512), trial.suggest_float("dropout_rate", 0.0, 0.5))

    all_input_sequences, all_target_tokens = load_training_data(encode_dir_path, seq_length, 10)

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
            num_files=10, 
            learning_rate=learning_rate, 
            architecture=architecture, 
            model_architecture_func=model_architecture_func,
            callbacks=[WandbCallback(), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
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
                    "embedding_dim": trial.suggest_int("embedding_dim", 32, 256),
                    "gru_units": trial.suggest_int("gru_units", 32, 256) if architecture == 'gru' else None,
                    "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
                    "recurrent_dropout_rate": trial.suggest_float("recurrent_dropout_rate", 0.0, 0.5) if architecture == 'gru' else None,
                    "num_layers": trial.suggest_int("num_layers", 1, 5) if architecture in ['lstm', 'bert'] else None
                }
                
                metadata_path = os.path.join(save_path, f"metadata_{trial.number}_{timestamp}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
            return loss
    except Exception as e:
        print(f"Training failed with exception: {e}. Returning inf loss.")
        return float('inf')

