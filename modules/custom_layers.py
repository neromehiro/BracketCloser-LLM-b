# custom_layers.py

import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention

class CustomMultiHeadAttention(MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False, training=None, **kwargs):
        if attention_mask is not None:
            # attention_maskの形状を(batch_size, 1, 1, seq_len)に変更
            batch_size = tf.shape(query)[0]
            seq_len = tf.shape(query)[1]
            attention_mask = tf.reshape(attention_mask, (batch_size, 1, 1, seq_len)) 

        return super().call(query, value, key=key, attention_mask=attention_mask, return_attention_scores=return_attention_scores, training=training)
