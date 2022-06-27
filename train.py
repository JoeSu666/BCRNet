import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import datetime
import h5py
import os
from tensorflow.keras import datasets,layers,models, metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import openpyxl
import random
import glob
import argparse
from util import *

INIT = 'glorot_uniform'

class AttMIL2(models.Model):
    def __init__(self):
        super(AttMIL2, self).__init__()
    
    def build(self, input_shape, n_class):
        self.V = tfa.layers.WeightNormalization(layers.Dense(input_shape[-1]//2,use_bias=False, kernel_initializer=INIT))
        self.U = tfa.layers.WeightNormalization(layers.Dense(input_shape[-1]//2,use_bias=False, kernel_initializer=INIT))
        
        self.Wa = layers.Dense(1,use_bias=False, kernel_initializer=INIT)
        self.softmax = layers.Softmax(axis=1)
        self.dot = layers.Dot(axes=1)
        
        self.WC = layers.Dense(1,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.00001), kernel_initializer=INIT)
        
        super(AttMIL2,self).build(input_shape)
        
    def call(self, x):
        x = x[0]
        V = tf.keras.activations.tanh(self.V(x))
        U = tf.keras.activations.sigmoid(self.U(x))
        energy = tf.math.multiply(V,U)
        #hs
        x = tf.expand_dims(x,0)
        att = tf.expand_dims(self.Wa(energy),0)
        att = self.softmax(att)       
        hs = self.dot([att,x]) # 1,vector_size
        
        hs = tf.squeeze(hs,1)
        #slide score for classes
        hs = layers.Dropout(rate=0.1)(hs)
        s = self.WC(hs)
                
        return s

def get_callbacks(log_dir):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return [EarlyStopping(monitor='accuracy', patience=15, restore_best_weights=True), tensorboard_callback]

def run(task, runningcode, fold, args):
    encoded_shape=256 # dim of vectors

    save_dir='./results' + task +'/'+runningcode # results saving dir

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    embedding_dir='./data/embedded' + task # embedding saving dir


    save_dir = save_dir + 'K' + str(args.K) + '/'
    print('sample size: ', args.K)

    # building dataset
    tf.keras.backend.clear_session()
    trainset, trainlabel = createbags_oneside(embedding_dir, 'train', fold, args.K, encoded_shape)
    testset, testlabel = createbags_oneside(embedding_dir, 'test', fold, args.K, encoded_shape)

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.0002,
        decay_steps=70*args.epochs,
        decay_rate=1,
        staircase=False)


    def traingen():
        for xy in zip(trainset,trainlabel):
            yield xy
    def testgen():
        for xy in zip(testset,testlabel):
            yield xy
    print('sets built')

    ds_train=tf.data.Dataset.from_generator(generator=traingen, output_types=(tf.float32, tf.int32),\
                                            output_shapes=(tf.TensorShape([None,encoded_shape]),tf.TensorShape([])))\
                    .map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                    .shuffle(len(trainset)).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    ds_test=tf.data.Dataset.from_generator(generator=testgen, output_types=(tf.float32, tf.int32),\
                                            output_shapes=(tf.TensorShape([None,encoded_shape]),tf.TensorShape([])))\
                    .map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                    .batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    print('pipeline built')

    # load model
    model=AttMIL2()
    model.build(input_shape = (None, args.K, encoded_shape),n_class = 2) # need to change shape and n_class
    log_dir='./results' + task +'/'+runningcode+'millogs/mil' + str(fold) + '/K' + str(args.K) + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print('log saving in ', log_dir)

    METRICS = [
    metrics.BinaryAccuracy(name='accuracy'),
    metrics.Precision(name='precision'),
    metrics.Recall(name='recall'),
    metrics.AUC(name='auc')]


    print('model built')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                    loss=tf.keras.losses.binary_crossentropy,
                    metrics=METRICS)
    
    # model training
    history = model.fit(ds_train,validation_data=ds_test,epochs=args.epochs,callbacks=get_callbacks(log_dir))
    model.save(save_dir + 'fold' + str(fold) + '/model')

    # saving testing CM to .xlsx
    print('train result')
    predictions=[]
    gts=[]
    probs = []
    for idx,(x,y) in enumerate(ds_train):   
        prob=model.predict_on_batch(x)[0,0]
        probs.append(prob)
        predictions.append(np.uint8(np.around(prob)))
        gts.extend(np.round(y).tolist())
    mat=confusion_matrix(gts,predictions, labels=[0, 1])
    prf = precision_recall_fscore_support(gts, predictions, average='binary')
    print(mat)
    print('AUC: ', roc_auc_score(gts, probs))
    print('precision: {:.4}, recall: {}, f1: {}'.format(prf[0], prf[1], prf[2]))

    print('test result')
    predictions=[]
    gts=[]
    probs = []
    for idx,(x,y) in enumerate(ds_test):   
        prob=model.predict_on_batch(x)[0,0]
        probs.append(prob)
        predictions.append(np.uint8(np.around(prob)))
        gts.extend(np.round(y).tolist())        
    mat=confusion_matrix(gts, predictions, labels=[0, 1])
    prf = precision_recall_fscore_support(gts, predictions, average='binary')
    print(mat)
    print('AUC: ', roc_auc_score(gts, probs))
    print('acc: ', accuracy_score(gts, predictions))
    print('precision: {:.4}, recall: {}, f1: {}, spe:{}'.format(prf[0], prf[1], prf[2], mat[0, 0] / mat[0].sum()))

    return accuracy_score(gts, predictions), roc_auc_score(gts, probs), prf[0], prf[1], prf[2], mat[0, 0] / mat[0].sum()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='HE')
    parser.add_argument('--runningcode', type=str, default='bcrnet')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--K', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=150)

    args = parser.parse_args()

    print('task: ', args.task)
    print('experiment: ', args.runningcode)
    
    print('fold: ', args.fold)
    acc, auc, precision, recall, f1, spe = run(args.task, args.runningcode, args.fold, args)

                
                    