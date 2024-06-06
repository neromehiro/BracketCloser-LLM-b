# modules/training_utils.py

import os
import time
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import numpy as np
from datetime import datetime  # datetimeモジュールをインポート
import wandb

class WandbCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            wandb.log(logs)


# class TrainingHistory(tf.keras.callbacks.Callback):
#     def __init__(self, model_path, model_architecture_func):
#         super().__init__()
#         self.model_path = model_path
#         self.model_architecture_func = model_architecture_func

#     def on_train_begin(self, logs={}):
#         self.history = []

#     def on_epoch_end(self, epoch, logs={}):
#         self.history.append(logs.copy())
#         self.save_metadata(epoch, logs)

#     def save_metadata(self, epoch, logs):
#         metadata_path = self.model_path.replace('.h5', f'_epoch_{epoch + 1}_metadata.json')
#         metadata = {
#             "epoch": epoch + 1,
#             "logs": logs,
#             "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "model_architecture": self.model_architecture_func.__name__
#         }
#         with open(metadata_path, 'w') as f:
#             json.dump(metadata, f, indent=4)

# hyperのためにクラスの保存方法を変えた
class TrainingHistory(tf.keras.callbacks.Callback):
    def __init__(self, model_path, model_architecture_func, save_interval=5):
        super().__init__()
        self.model_path = model_path
        self.model_architecture_func = model_architecture_func
        self.history = []
        self.save_interval = save_interval

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history.append(logs.copy())
        if (epoch + 1) % self.save_interval == 0:
            self.save_metadata(epoch, logs)
            self.save_model_checkpoint(epoch)

    def save_metadata(self, epoch, logs):
        metadata_path = self.model_path.replace('.h5', f'_epoch_{epoch + 1}_metadata.json')
        metadata = {
            "epoch": epoch + 1,
            "logs": logs,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_architecture": self.model_architecture_func.__name__
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def save_model_checkpoint(self, epoch):
        model_checkpoint_path = self.model_path.replace('.h5', f'_epoch_{epoch + 1}.h5')
        self.model.save(model_checkpoint_path)


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



def train_model_single(model, input_sequences, target_tokens, epochs, batch_size, model_path, num_files, learning_rate, architecture, model_architecture_func,
                embedding_dim=64, gru_units=64, dropout_rate=0.2, recurrent_dropout_rate=0.2):
    if len(input_sequences) > 0 and len(target_tokens) > 0:
        print(f"Shapes: {input_sequences.shape}, {target_tokens.shape}")

        validation_split = 0.2
        num_validation_samples = int(validation_split * len(input_sequences))

        sample_weights = np.where(target_tokens != 0, 1.0, 0.0)

        if 'transformer' in architecture or 'gpt' in architecture:
            attention_mask = (input_sequences != 0).astype(np.float32)

            train_inputs = {
                'input_1': input_sequences[:-num_validation_samples],
                'attention_mask': attention_mask[:-num_validation_samples]
            }
            val_inputs = {
                'input_1': input_sequences[-num_validation_samples:],
                'attention_mask': attention_mask[-num_validation_samples:]
            }

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_inputs,
                 target_tokens[:-num_validation_samples],
                 sample_weights[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (val_inputs,
                 target_tokens[-num_validation_samples:],
                 sample_weights[-num_validation_samples:])
            ).batch(batch_size)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[:-num_validation_samples],
                 target_tokens[:-num_validation_samples],
                 sample_weights[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[-num_validation_samples:],
                 target_tokens[-num_validation_samples:],
                 sample_weights[-num_validation_samples:])
            ).batch(batch_size)

        train_dataset = train_dataset.shuffle(buffer_size=1024)

        for data, labels, weights in train_dataset.take(1):
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"Train data batch shape for {key}: {value.shape}")
            else:
                print("Train data batch shape: ", data.shape)
            print("Train labels batch shape: ", labels.shape)
            print("Train sample weights batch shape: ", weights.shape)

        time_callback = TimeHistory()
        checkpoint_callback = ModelCheckpoint(filepath=model_path, save_weights_only=False, save_best_only=False, save_freq='epoch', verbose=1)
        history_callback = TrainingHistory(model_path, model_architecture_func)

        try:
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=validation_dataset,
                callbacks=[time_callback, checkpoint_callback, history_callback]
            )
            model.save(model_path, include_optimizer=False, save_format='h5')
            return history_callback.history, len(input_sequences)
        except Exception as e:
            print(f"Training failed with exception: {e}")
            print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Epochs: {epochs}")
            print(f"Train data shape: {input_sequences.shape}, Target data shape: {target_tokens.shape}")
            return None, 0
    else:
        print("No data for training.")
        return None, 0

# hyperのためにちょっと変えた
def train_model(model, input_sequences, target_tokens, epochs, batch_size, model_path, num_files, learning_rate, architecture, model_architecture_func,
                embedding_dim=64, gru_units=64, dropout_rate=0.2, recurrent_dropout_rate=0.2):
    if len(input_sequences) > 0 and len(target_tokens) > 0:
        print(f"Shapes: {input_sequences.shape}, {target_tokens.shape}")

        validation_split = 0.2
        num_validation_samples = int(validation_split * len(input_sequences))

        sample_weights = np.where(target_tokens != 0, 1.0, 0.0)

        if 'transformer' in architecture or 'gpt' in architecture:
            attention_mask = (input_sequences != 0).astype(np.float32)

            train_inputs = {
                'input_1': input_sequences[:-num_validation_samples],
                'attention_mask': attention_mask[:-num_validation_samples]
            }
            val_inputs = {
                'input_1': input_sequences[-num_validation_samples:],
                'attention_mask': attention_mask[-num_validation_samples:]
            }

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_inputs,
                 target_tokens[:-num_validation_samples],
                 sample_weights[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (val_inputs,
                 target_tokens[-num_validation_samples:],
                 sample_weights[-num_validation_samples:])
            ).batch(batch_size)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[:-num_validation_samples],
                 target_tokens[:-num_validation_samples],
                 sample_weights[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[-num_validation_samples:],
                 target_tokens[-num_validation_samples:],
                 sample_weights[-num_validation_samples:])
            ).batch(batch_size)

        train_dataset = train_dataset.shuffle(buffer_size=1024)

        for data, labels, weights in train_dataset.take(1):
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"Train data batch shape for {key}: {value.shape}")
            else:
                print("Train data batch shape: ", data.shape)
            print("Train labels batch shape: ", labels.shape)
            print("Train sample weights batch shape: ", weights.shape)

        time_callback = TimeHistory()
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False, save_best_only=False, save_freq='epoch', verbose=1)
        history_callback = TrainingHistory(model_path, model_architecture_func)

        try:
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=validation_dataset,
                callbacks=[time_callback, checkpoint_callback, history_callback]
            )
            model.save(model_path, include_optimizer=False, save_format='h5')
            return history, len(input_sequences)  # 修正: history_callback.historyからhistoryに変更
        except Exception as e:
            print(f"Training failed with exception: {e}")
            print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Epochs: {epochs}")
            print(f"Train data shape: {input_sequences.shape}, Target data shape: {target_tokens.shape}")
            return float('inf'), 0
    else:
        print("No data for training.")
        return None, 0



def plot_training_history(history, save_path, epochs, batch_size, learning_rate, num_files, dataset_size):
    losses = [epoch_logs['loss'] for epoch_logs in history]
    val_losses = [epoch_logs['val_loss'] for epoch_logs in history]
    accuracies = [epoch_logs['accuracy'] for epoch_logs in history]
    val_accuracies = [epoch_logs['val_accuracy'] for epoch_logs in history]

    epochs_range = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.suptitle(f'Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Files: {num_files}, Dataset Size: {dataset_size}', y=1.05)
    plt.savefig(save_path)
    plt.close()

def save_metadata(model_path, metadata):
    metadata_path = model_path.replace('.h5', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)