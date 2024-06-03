import os
import time
import json  # jsonモジュールをインポート
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

class TrainingHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_accuracy'))

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def train_model(model, input_sequences, target_tokens, epochs, batch_size, model_path, num_files, learning_rate):
    if len(input_sequences) > 0 and len(target_tokens) > 0:
        print(f"Shapes: {input_sequences.shape}, {target_tokens.shape}")
        
        validation_split = 0.2
        num_validation_samples = int(validation_split * len(input_sequences))
        
        train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences[:-num_validation_samples], target_tokens[:-num_validation_samples]))
        validation_dataset = tf.data.Dataset.from_tensor_slices((input_sequences[-num_validation_samples:], target_tokens[-num_validation_samples:]))

        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        validation_dataset = validation_dataset.batch(batch_size)
        
        time_callback = TimeHistory()
        checkpoint_callback = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        history_callback = TrainingHistory()

        start_time = time.time()
        history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, callbacks=[time_callback, checkpoint_callback, history_callback])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training finished. Time taken: {elapsed_time} seconds.")
        average_epoch_time = sum(time_callback.times) / len(time_callback.times)
        print(f"Average time per epoch: {average_epoch_time} seconds.")
        
        return history_callback, len(input_sequences)
    else:
        print(f"No data for training.")
        return None, 0

def plot_training_history(history, save_path='training_history.png', epochs=None, batch_size=None, learning_rate=None, num_files=None, dataset_size=None):
    epochs_range = range(1, len(history.losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.losses, label='Training loss')
    plt.plot(epochs_range, history.val_losses, label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.accuracies, label='Training accuracy')
    plt.plot(epochs_range, history.val_accuracies, label='Validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    textstr = f'Dataset size: {dataset_size}\nNum files: {num_files}\nEpochs: {epochs}\nBatch size: {batch_size}\nLearning rate: {learning_rate}'
    plt.gcf().text(0.75, 0.5, textstr, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metadata(model_path, metadata):
    metadata_path = model_path.replace('.h5', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
