import tensorflow as tf
from tensorflow.keras import layers, models

def define_gru_model(seq_length, output_dim, learning_rate):
    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=64, mask_zero=True)(inputs)
    x = layers.GRU(64, dropout=0.2, recurrent_dropout=0.2)(x)
    outputs = layers.Dense(output_dim, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

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

def define_lstm_model(seq_length, output_dim, learning_rate):
    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=64, mask_zero=True)(inputs)
    x = layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)
    outputs = layers.Dense(output_dim, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model

def define_bert_model(seq_length, output_dim, learning_rate):
    # Note: Full BERT implementation requires pre-trained weights and tokenizer
    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=64, mask_zero=True)(inputs)
    
    # Simplified BERT Encoder
    for _ in range(2):  # 2 Encoder layers as an example
        attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        attention_output = layers.Dropout(0.1)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)

        ffn = layers.Dense(128, activation='relu')(attention_output)
        ffn_output = layers.Dense(64)(ffn)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

    outputs = layers.Dense(output_dim, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model


def define_gpt_model(seq_length, output_dim, learning_rate):
    # Note: Full GPT implementation requires pre-trained weights and tokenizer
    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=64, mask_zero=True)(inputs)
    
    # Simplified GPT Decoder
    for _ in range(2):  # 2 Decoder layers as an example
        causal_mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
        attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x, attention_mask=causal_mask)
        attention_output = layers.Dropout(0.1)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)

        ffn = layers.Dense(128, activation='relu')(attention_output)
        ffn_output = layers.Dense(64)(ffn)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

    outputs = layers.Dense(output_dim, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model
