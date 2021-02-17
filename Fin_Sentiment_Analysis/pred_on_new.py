import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model_single = tf.keras.models.load_model('models/rcnn_single_layer/')
# %%
model_layered = tf.keras.models.load_model('models/rcnn_multi_layered/')
# %%
s = ('A deep freeze enveloping large swathes of the U.S. has resulted in rolling blackouts for at least 5M people from the upper Midwest to Houston. More than a million barrels a day of oil and 10B cubic feet of gas production have also gone offline, sending U.S. crude prices above $60 a barrel for the first time in more than year and natural gas prices up 6%. This is a terrible event. Pipelines have also declared force majeure, while massive refineries owned by Exxon Mobil (NYSE:XOM) and Marathon Petroleum (NYSE:MPC) have halted production, threatening to reduce supplies of gasoline and diesel across the country.')
predictions = model_layered.predict(np.array([s]))
print(predictions)
# %%
sample_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = model_layered.predict(np.array([sample_text]))
print(predictions)