
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
import numpy as np

class CustomMultiHeadAttention(MultiHeadAttention):
    def call(self, query, value, key=None, attention_mask=None, training=None, return_attention_scores=False):
        if isinstance(attention_mask, list):
            attention_mask = np.array(attention_mask)
            attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.float32)
        return super().call(query, value, key=key, attention_mask=attention_mask, 
                            training=training, return_attention_scores=return_attention_scores)

def define_gpt_model(seq_length, output_dim, learning_rate):
    inputs = Input(shape=(seq_length,))
    x = Embedding(input_dim=output_dim, output_dim=64, mask_zero=True)(inputs)
    for _ in range(2):
        causal_mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
        causal_mask = tf.cast(causal_mask, dtype=tf.float32)
        causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]

        attention_output = CustomMultiHeadAttention(num_heads=4, key_dim=64)(x, x, attention_mask=causal_mask)
        attention_output = Dropout(0.1)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output + x)

        ffn = Dense(128, activation='relu')(attention_output)
        ffn_output = Dense(64)(ffn)
        ffn_output = Dropout(0.1)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

    outputs = Dense(output_dim, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# モデルの生成
model = define_gpt_model(seq_length=10, output_dim=1000, learning_rate=0.001)
model_save_path = "./models/best_model.h5"

model.save(model_save_path)

# カスタムオブジェクトを渡して読み込み
loaded_model = tf.keras.models.load_model(model_save_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention})
print("Model loaded successfully.")

# テストデータ
test_data = tf.random.uniform((1, 10), minval=0, maxval=999, dtype=tf.int32)
output = loaded_model.predict(test_data)
print("Output shape:", output.shape)