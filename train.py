import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt

# トークンとIDの対応付け
tokens = ["(", ")", "[", "]", "{", "}", "input", ",output" ,","]
token2id = {token: i for i, token in enumerate(tokens)}

# データセットの保存先ディレクトリ
encode_dir_path = "./dataset/preprocessed/"

# モデル保存先ディレクトリ
model_save_path = "./models/"


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

def load_dataset(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def select_mode():
    mode = input("Select a mode from: " + ", ".join(TRAINING_MODES.keys()) + "\n")
    while mode not in TRAINING_MODES:
        print(f"Invalid mode. Please select a mode from: {', '.join(TRAINING_MODES.keys())}")
        mode = input()
    return TRAINING_MODES[mode]["epochs"], TRAINING_MODES[mode]["batch_size"], TRAINING_MODES[mode]["num_files"], TRAINING_MODES[mode]["learning_rate"]

def define_gru_model(seq_length, output_dim, learning_rate):
    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=64, mask_zero=True)(inputs)
    x = layers.GRU(64, dropout=0.2, recurrent_dropout=0.2)(x)  # ドロップアウトを追加
    outputs = layers.Dense(output_dim, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # L2正則化を追加

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    return model


def define_transformer_model(seq_length, output_dim, learning_rate):
    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=64, mask_zero=True)(inputs)

    # Transformerブロックを作成
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    attention_output = layers.Dropout(0.1)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)

    ffn = layers.Dense(128, activation='relu')(attention_output)
    ffn_output = layers.Dense(64)(ffn)
    ffn_output = layers.Dropout(0.1)(ffn_output)
    ffn_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

    outputs = layers.Dense(output_dim, activation="softmax")(ffn_output)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    return model

MODEL_ARCHITECTURES = {
    "gru": define_gru_model,
    "transformer": define_transformer_model
}

def select_mode_and_architecture():
    modes = list(TRAINING_MODES.keys())
    architectures = list(MODEL_ARCHITECTURES.keys())
    choices = [f"{arch} {mode}" for arch in architectures for mode in modes]

    print("以下のモードとアーキテクチャから選んでください。選択肢は英語のまま入力してください：\n")
    
    print("1. GRU (Gated Recurrent Unit)")
    for mode in modes:
        print(f"    - {mode}: gru {mode}")

    print("\n2. Transformer")
    for mode in modes:
        print(f"    - {mode}: transformer {mode}")

    choice = input("\nあなたの選択: ")
    
    while choice not in choices:
        print(f"\n無効な選択です。以下の選択肢から選んでください：\n")
        
        print("1. GRU (Gated Recurrent Unit)")
        for mode in modes:
            print(f"    - {mode}: gru {mode}")

        print("\n2. Transformer")
        for mode in modes:
            print(f"    - {mode}: transformer {mode}")
        
        choice = input("\nあなたの選択: ")
    
    arch, mode = choice.split()
    return MODEL_ARCHITECTURES[arch], TRAINING_MODES[mode]

# 例:
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

MODEL_ARCHITECTURES = {
    "gru": define_gru_model,
    "transformer": define_transformer_model
}

def save_model(model, model_path):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    model.save(model_path)

def prepare_sequences(encoded_tokens, seq_length):
    input_sequences = []
    target_tokens = []
    for i in range(len(encoded_tokens) - seq_length):
        input_sequences.append(encoded_tokens[i:i+seq_length])
        target_tokens.append(encoded_tokens[i+seq_length])
    return np.array(input_sequences), np.array(target_tokens)

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

    # 追加情報を図の右側に記載
    textstr = f'Dataset size: {dataset_size}\nNum files: {num_files}\nEpochs: {epochs}\nBatch size: {batch_size}\nLearning rate: {learning_rate}'
    plt.gcf().text(0.75, 0.5, textstr, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metadata(model_path, metadata):
    metadata_path = model_path.replace('.h5', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def main():
    model_architecture_func, training_mode = select_mode_and_architecture()
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

    model_path = f"{model_save_path}best_model.h5"
    plot_path = f"{model_save_path}training_history.png"

    history, dataset_size = train_model(model, all_input_sequences, all_target_tokens, epochs=epochs, batch_size=batch_size, model_path=model_path, num_files=num_files, learning_rate=learning_rate)
    
    if history:
        plot_training_history(history, save_path=plot_path, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, num_files=num_files, dataset_size=dataset_size)

    # モデル容量を取得
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB単位に変換
    # モデルのパラメータ数を取得
    model_params = model.count_params()

    # メタデータの保存
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