import numpy as np

import tensorflow_datasets as tfds

import tensorflow as tf

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric],'')
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.legend([metric,'val_'+metric])
    
dataset, info = tfds.load('imdb_reviews', )