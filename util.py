import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import h5py
import os
from os.path import join
from tensorflow.keras import datasets,layers,models
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from itertools import zip_longest
import openpyxl
import time
import random
import glob
import openslide
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def chunks(l,n):
    for i in range(0,len(l),n):
        yield l[i:i+n]

def load_discimage(patch,label, n_class=2):
    if n_class==2:
        if label==0:           
            label=tf.constant(0,tf.int8)
        else:
            label=tf.constant(1,tf.int8)

    patch=tf.cast(patch,tf.float32)
    patch=tf.keras.applications.resnet50.preprocess_input(patch)


    return (patch, label) #load image for discriminator    

def load_data(bag,label):
    if label==0:           
        label=tf.constant(0,tf.int8)
    else:
        label=tf.constant(1,tf.int8)
    
    return (bag,label) #load image to MIL
       

# def leave_two(dic, start, task):
#     trainnamelist = []
#     testnamelist = []
#     if task == 'HE':
#         for d in dic:
#             testnamelist.append(dic[d][start])
#             trainnamelist.extend(dic[d][:start]+dic[d][start+1:])
#     elif task == 'KI':
#         if start == 24:
#             testnamelist.extend(dic[0][-2:])
#             trainnamelist.extend(dic[0][:-2]+dic[1][:])
#         else:
#             for d in dic:
#                 testnamelist.append(dic[d][start])
#                 trainnamelist.extend(dic[d][:start]+dic[d][start+1:])
#     return trainnamelist, testnamelist

def datasplit(fold, task, ptsdir):
    if task == 'HE':
        posnamelist = glob.glob(join(ptsdir, 'positive', '*.npy'))
        negnamelist = glob.glob(join(ptsdir, 'negative', '*.npy'))

        posnamelist = shuffle(posnamelist, random_state=42)
        negnamelist = shuffle(negnamelist, random_state=42, n_samples=36)
        trainnamelist = posnamelist[:fold] + posnamelist[fold+1:] + negnamelist[:fold] + negnamelist[fold+1:]
        testnamelist = [posnamelist[fold], negnamelist[fold]]


        return trainnamelist, testnamelist
        
    elif task == 'KI':
        trainnamelist = []
        testnamelist = []
        if fold == 24:
            testnamelist.extend(dic[0][-2:])
            trainnamelist.extend(dic[0][:-2]+dic[1][:])
        else:
            for d in dic:
                testnamelist.append(dic[d][fold])
                trainnamelist.extend(dic[d][:fold]+dic[d][fold+1:])

        return trainnamelist, testnamelist

def createbags_oneside(embedding_dir, train, fold, K, encoded_shape):
    scoreset = np.load(join(embedding_dir, train+str(fold)+'foldscores.npy'), allow_pickle=True).item()
    dataset = np.load(join(embedding_dir, train+str(fold)+'foldembeddeddata.npy'), allow_pickle=True).item()
    labelset = np.load(join(embedding_dir, train+str(fold)+'foldlabels.npy'), allow_pickle=True).item()


    bags = []
    labels = []
    for name in dataset:

        scores = scoreset[name]
        scores = np.squeeze(scores, 1)
        sortidxs = np.argsort(-scores)
        length = scores.shape[0]
        

        if length >= K:
            embedding = dataset[name][sortidxs[:K]]

        else:
            embedding = dataset[name][sortidxs[:]]
        
        bags.append(embedding)
        labels.append(int(labelset[name]))
    return bags, labels


class EarlyStopping(Callback):
  """
  Args:
    monitor: Quantity to be monitored.
    min_delta: Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement.
    patience: Number of epochs with no improvement
        after which training will be stopped.
    verbose: verbosity mode.
    mode: One of `{"auto", "min", "max"}`. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `"max"`
        mode it will stop when the quantity
        monitored has stopped increasing; in `"auto"`
        mode, the direction is automatically inferred
        from the name of the monitored quantity.
    baseline: Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline.
    restore_best_weights: Whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used. An epoch will be restored regardless
        of the performance relative to the `baseline`. If no epoch
        improves on `baseline`, training will run for `patience`
        epochs and restore weights from the best epoch in that set.

  """

  def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None,
               restore_best_weights=False,
               min_epoch=10):
    super(EarlyStopping, self).__init__()

    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.baseline = baseline
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0
    self.restore_best_weights = restore_best_weights
    self.min_epoch = min_epoch
    self.best_weights = None

    if mode not in ['auto', 'min', 'max']:
      logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor:
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

  def on_train_begin(self, logs=None):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    self.best_weights = None
    self.best_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.restore_best_weights and self.best_weights is None:
      # Restore the weights after first epoch if no progress is ever made.
      self.best_weights = self.model.get_weights()

    # Count only after min epoch
    if epoch > self.min_epoch:
      self.wait += 1

    if self._is_improvement(current, self.best):
      self.best = current
      self.best_epoch = epoch
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
      # Only restart wait if we beat both the baseline and our previous best.
      if self.baseline is None or self._is_improvement(current, self.baseline):
        self.wait = 0

    # Only check after the first epoch.
    if self.wait >= self.patience and epoch > self.min_epoch:
      self.stopped_epoch = epoch
      self.model.stop_training = True
      if self.restore_best_weights and self.best_weights is not None:
        if self.verbose > 0:
          print('Restoring model weights from the end of the best epoch (%s).' %
                (self.best_epoch + 1))
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    if monitor_value is None:
      logging.warning('Early stopping conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))
    return monitor_value

  def _is_improvement(self, monitor_value, reference_value):
    return self.monitor_op(monitor_value - self.min_delta, reference_value)