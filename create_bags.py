import tensorflow as tf
import numpy as np
import datetime
import h5py
import os
from os.path import join
from tensorflow.keras import datasets,layers,models
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import openpyxl
import random
import glob
import argparse
import util
from util import *


class samplegen():
    def __init__(self, dirname, pts, psize):
        self.dirname = dirname
        self.pts = pts
        self.psize = psize
        
    def __call__(self):
        with openslide.OpenSlide(self.dirname) as fp:

            for i in range(self.pts.shape[0]):
                pt = self.pts[i]
                image = np.asarray(fp.read_region((pt[1], pt[0]), 0, (self.psize, self.psize)).convert('RGB'))

                yield image, 0

def intelsampling(datanamelist, encoded_shape, disc_model, encoder, embedding_dir, train, args, sampling=False):
    datadict = {}
    scoredict = {}
    coorddict = {}
    labeldict = {}
    bsize = 200
    for idx, name in enumerate(datanamelist):
        pid = name.split('/')[-1].split('.')[0]
        label = int(name.split('/')[-2] == 'positive')

        dataname = join(args.datasrc, pid+'.tif')
        
        pts = np.load(name)

        # load dataset
        print('*********')
        print(pid)
        start = time.time()
        bag=np.zeros((pts.shape[0],encoded_shape))
        scores=np.zeros((pts.shape[0],1),dtype=np.float64)
        gen = samplegen(dataname, pts, args.psize)

        ds_test=tf.data.Dataset.from_generator(generator=gen, output_types=(tf.uint8, tf.int32),\
                                                    output_shapes=(tf.TensorShape([args.psize, args.psize, 3]),tf.TensorShape([])))\
                .map(load_discimage, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)

        i = 0
        for x,y in ds_test:
            scores[i:i+x.shape[0]]=np.absolute(disc_model.predict_on_batch(x)-0.5)
            bag[i:i+x.shape[0]]=encoder.predict_on_batch(x)

            i = i+x.shape[0]

        # sortidx=np.argsort(-scores,axis=0)
        # sortidx=np.squeeze(sortidx,1)
        # bag = bag[sortidx]
        # scores = scores[sortidx]

        print(bag.shape)

        if bag.shape[-1]!=encoded_shape:
            raise IndexError('wrong shape of bag')

        datadict[name]=bag
        labeldict[name]=label
        
        scoredict[name]=scores
        end = time.time()
        print('Used {} s'.format(end-start))

    np.save(join(embedding_dir, train+str(args.fold)+'fold'+'embeddeddata.npy'),datadict)
    np.save(join(embedding_dir, train+str(args.fold)+'fold'+'labels.npy'),labeldict)
    
    np.save(join(embedding_dir, train+str(args.fold)+'fold'+'scores.npy'),scoredict)
    print('Embeddings saved in ', join(embedding_dir, train+str(args.fold)+'fold'+'embeddeddata.npy'))

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


def run(args):
    model_dir = join('checkpoints'+args.task, str(args.fold) + 'fold_disc.h5')
    embedding_dir = 'data/embedded'+args.task
    trainnamelist, testnamelist = datasplit(args.fold, args.task, args.ptsdir)
    encoded_shape=256 # dim of vectors

    print('TASK: ', args.task)
    print('load pretrained disc from: ', model_dir)
    
    tf.keras.backend.clear_session()

    disc_model = builddiscriminator(args)
    disc_model.load_weights(model_dir, by_name=True, skip_mismatch=True)

    disc_model.trainable=False

    encoder = models.Model(inputs = disc_model.input, outputs = disc_model.get_layer('enflatten').output)
    encoder.trainable=False


    # intelligent sampling
    intelsampling(trainnamelist, encoded_shape, disc_model, encoder, embedding_dir, 'train', args, sampling=True)
    intelsampling(testnamelist, encoded_shape, disc_model, encoder, embedding_dir, 'test', args, sampling=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='HE')
    parser.add_argument('--ptsdir', type=str, default='./data/pts')
    parser.add_argument('--start', type=int)
    parser.add_argument('--fold', type=int, help='fold number')
    parser.add_argument('--datasrc', type=str,
    parser.add_argument('--psize', type=int, default=128)
    parser.add_argument('--code', default='newcases', type=str, help='code') 

    args = parser.parse_args()
    args.ptsdir = join(args.ptsdir, args.code+'l0p' + str(args.psize) + 's' + str(args.psize))

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    print('task: ', args.task)
    print('load coords from: ', args.ptsdir)

    run(args)
    