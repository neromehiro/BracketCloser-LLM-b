# custom_layers.py


import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention

class CustomMultiHeadAttention(MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False, training=None, **kwargs):
        if attention_mask is not None:
            batch_size = tf.shape(query)[0]
            seq_length = tf.shape(query)[1]
            # (batch_size, 1, 1, seq_len)に形状を変更
            attention_mask = tf.reshape(attention_mask, (batch_size, 1, 1, seq_length))

        return super().call(query, value, key=key, attention_mask=attention_mask, return_attention_scores=return_attention_scores, training=training)