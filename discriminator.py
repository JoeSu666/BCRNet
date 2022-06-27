import tensorflow as tf
import numpy as np
import h5py
import os
from os.path import join
from tensorflow.keras import datasets,layers,models,metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import random
from util import *
import argparse


def buildsampleset(datanamelist, args, train='train'):
    
    dataset = np.empty((0, args.psize, args.psize, 3), 'uint8')
    labels = np.empty((0, 1), 'int32')

    for name in datanamelist:
        pid = name.split('/')[-1].split('.')[0]
        label = int(name.split('/')[-2] == 'positive')
        print(pid)


        dataname = join(args.datasrc, pid+'.tif')
        
        pts = np.load(name)
        if pts.shape[0] < args.ssize:
            idx = np.random.choice(pts.shape[0], size=pts.shape[0], replace=False)
        else:
            idx = np.random.choice(pts.shape[0], size=args.ssize, replace=False)

        with openslide.OpenSlide(dataname) as fp:
            for i in idx:
                pt = pts[i]
                image = np.asarray(fp.read_region((pt[1], pt[0]), 0, (args.psize, args.psize)).convert('RGB'))
                dataset.resize((dataset.shape[0]+1, args.psize, args.psize, 3))
                dataset[-1] = image
                labels.resize((labels.shape[0]+1, 1))
                labels[-1] = label

    np.save(join(args.samplesave, train + str(args.fold) + 'foldsample.npy'), dataset)
    np.save(join(args.samplesave, train + str(args.fold) + 'foldsamplelabels.npy'), labels)

    return dataset, labels

def get_callbacks():
    return tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)


def builddiscriminator(args):
    disc_model = models.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(args.psize, args.psize, 3),name='enrescale'),
    layers.Conv2D(32, 3, padding='same', activation='relu',name='enconv1'),
    layers.MaxPooling2D(pool_size=(4, 4),name='enpooling1'),
    layers.Conv2D(16, 1, padding='same', activation='relu',name='enconv2'),
    layers.MaxPooling2D(pool_size=(4, 4), name='enpooling2'),
    layers.Conv2D(16, 3, padding='same', activation='relu',name='enconv3'),
    layers.MaxPooling2D(pool_size=(2, 2), name='enpooling3'),
    layers.Flatten(name='enflatten'),
    layers.Dropout(rate=0.1,name='endrop'),
    layers.Dense(1,activation = 'sigmoid',name='enhead'),

    ],name='discriminator')

    return disc_model

def train(args):
    bsize = 50
    model_dir = join(args.save, str(args.fold)+'fold_disc.h5')
    # get data split names
    trainnamelist, _ = datasplit(args.fold, args.task, args.ptsdir)
    postrainnamelist, postestnamelist = train_test_split(trainnamelist[:35], test_size=2, random_state=42)
    negtrainnamelist, negtestnamelist = train_test_split(trainnamelist[35:], test_size=2, random_state=42)
    trainnamelist = postrainnamelist + negtrainnamelist
    testnamelist = postestnamelist + negtestnamelist

    # Load sample sets if exist, or build sample sets
    if os.path.exists(join(args.samplesave, 'train' + str(args.fold) + 'foldsample.npy')):
        trainset = np.load(join(args.samplesave, 'train' + str(args.fold) + 'foldsample.npy'))
        trainlabels = np.load(join(args.samplesave, 'train' + str(args.fold) + 'foldsamplelabels.npy'))
        testset = np.load(join(args.samplesave, 'test' + str(args.fold) + 'foldsample.npy'))
        testlabels = np.load(join(args.samplesave, 'test' + str(args.fold) + 'foldsamplelabels.npy'))
    else:
        # get train/test sample set for disc
        trainset, trainlabels = buildsampleset(trainnamelist, args, 'train')
        testset, testlabels = buildsampleset(testnamelist, args, 'test')

    print('training size:', trainset.shape, trainlabels.shape)
    print('testing size:', testset.shape, testlabels.shape)
    print('dataset ready')

    ds_train=tf.data.Dataset.from_tensor_slices((trainset,trainlabels))\
            .map(load_discimage, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .shuffle(10000).batch(bsize).shuffle(100).prefetch(tf.data.experimental.AUTOTUNE)
    ds_test=tf.data.Dataset.from_tensor_slices((testset,testlabels))\
            .map(load_discimage, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    print('pipeline built')

    tf.keras.backend.clear_session()
    disc_model = builddiscriminator(args)

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      0.0002,
      decay_steps=132*150,
      decay_rate=1,
      staircase=False)

    METRICS = [
    metrics.BinaryAccuracy(name='accuracy'),
    metrics.Recall(name='recall')]

    disc_model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=METRICS)
    history = disc_model.fit(
      ds_train,
      validation_data=ds_test,
      epochs=150,
      callbacks=get_callbacks()
    )
    
    disc_model.save_weights(model_dir)
    disc_model.trainable = False

    disc_model.evaluate(ds_test)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='HE')
    parser.add_argument('--runningcode', type=str, default='bcrnet')
    parser.add_argument('--start', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--ssize', type=int, default=200)
    parser.add_argument('--psize', type=int, default=128)
    parser.add_argument('--ptsdir', type=str, default='./data/pts')
    parser.add_argument('--datasrc', type=str,
    parser.add_argument('--save', type=str, default='./checkpointsHE')
    parser.add_argument('--samplesave', type=str, default='./data/sampleset')
    parser.add_argument('--code', default='newcases', type=str, help='code') 



    args = parser.parse_args()

    args.ptsdir = join(args.ptsdir, args.code+'l0p' + str(args.psize) + 's' + str(args.psize))
    args.samplesave = join(args.samplesave, args.code+'l0p' + str(args.psize) + 's' + str(args.psize))

    if not os.path.exists(args.samplesave):
        os.mkdir(args.samplesave)

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    print('task: ', args.task)
    print('experiment: ', args.runningcode)
    print('load coords from: ', args.ptsdir)
    train(args)
    # for i in range(args.start, args.start+2):
    #     if i > 35:
    #         print('invalid fold')
    #         break
    #     args.fold = i
    #     print('*********************')
    #     print('fold: ', i)
    #     train(args)

