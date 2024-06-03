# custom_layers.py

import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention

class CustomMultiHeadAttention(MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False, training=None, **kwargs):
        if isinstance(attention_mask, list):
            attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.float32)
        return super().call(query, value, key=key, attention_mask=attention_mask, return_attention_scores=return_attention_scores, training=training)