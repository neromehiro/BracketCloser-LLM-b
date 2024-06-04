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



class TrainingHistory(tf.keras.callbacks.Callback):
    def __init__(self, model_path, model_architecture_func):
        super().__init__()
        self.model_path = model_path
        self.model_architecture_func = model_architecture_func

    def on_train_begin(self, logs={}):
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        self.history.append(logs.copy())
        self.save_metadata(epoch, logs)

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

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



def train_model(model, input_sequences, target_tokens, epochs, batch_size, model_path, num_files, learning_rate, architecture, model_architecture_func):
    if len(input_sequences) > 0 and len(target_tokens) > 0:
        print(f"Shapes: {input_sequences.shape}, {target_tokens.shape}")

        validation_split = 0.2
        num_validation_samples = int(validation_split * len(input_sequences))

        if 'transformer' in architecture or 'gpt' in architecture:
            attention_mask = np.ones_like(input_sequences) 

            train_dataset = tf.data.Dataset.from_tensor_slices(
                ({'input_1': input_sequences[:-num_validation_samples],
                  'input_2': attention_mask[:-num_validation_samples]},
                 target_tokens[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                ({'input_1': input_sequences[-num_validation_samples:],
                  'input_2': attention_mask[-num_validation_samples:]},
                 target_tokens[-num_validation_samples:])
            ).batch(batch_size)

        else:
            # 他のアーキテクチャの場合は、attention_maskは不要なのでそのままのデータセットを使う
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[:-num_validation_samples], target_tokens[:-num_validation_samples])
            ).batch(batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences[-num_validation_samples:], target_tokens[-num_validation_samples:])
            ).batch(batch_size)

        train_dataset = train_dataset.shuffle(buffer_size=1024)

        # データセットの形状を確認
        for data, labels in train_dataset.take(1):
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"Train data batch shape for {key}: {value.shape}")
            else:
                print("Train data batch shape: ", data.shape)
            print("Train labels batch shape: ", labels.shape)

        wandb.init(project="bracket_closer_llm")

        time_callback = TimeHistory()
        checkpoint_callback = ModelCheckpoint(filepath=model_path, save_weights_only=False, save_best_only=False, save_freq='epoch', verbose=1)
        history_callback = TrainingHistory(model_path, model_architecture_func)
        wandb_callback = WandbCallback()

        history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, callbacks=[time_callback, checkpoint_callback, history_callback, wandb_callback])

        model.save(model_path, include_optimizer=False, save_format='h5')
        
        return history_callback.history, len(input_sequences)
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