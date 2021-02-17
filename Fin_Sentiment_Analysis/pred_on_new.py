import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# %%
# for custom models
model_single = tf.keras.models.load_model('models/rcnn_single_layer/')
model_layered = tf.keras.models.load_model('models/rcnn_multi_layered/')


def pred_on_new_custom(s): # Str input; Returns prediction
    input = (s)
    return model_layered.predict(np.array([input]))

# %%
# for finetuned bert model 

bert_model = tf.saved_model.load('models/finetuned_bert/')

def pred_on_new_bert(s): # Str input; Returns prediction
    input = [s]
    return tf.sigmoid(bert_model(tf.constant(input)))