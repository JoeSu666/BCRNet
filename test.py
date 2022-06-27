import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import h5py
import os
from tensorflow.keras import datasets,layers,models
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import openpyxl
import random
import glob
import argparse
from util import *


def run(task, runningcode, fold, args):
    encoded_shape = 256
    embedding_dir='./data/embedded' + task # embedding saving dir

    # building dataset
    tf.keras.backend.clear_session()
    trainset, trainlabel = createbags_oneside(embedding_dir, 'train', fold, args.K, encoded_shape)
    testset, testlabel = createbags_oneside(embedding_dir, 'test', fold, args.K, encoded_shape)

    def traingen():
        for xy in zip(trainset,trainlabel):
            yield xy
    def testgen():
        for xy in zip(testset,testlabel):
            yield xy

    ds_train=tf.data.Dataset.from_generator(generator=traingen, output_types=(tf.float32, tf.int32),\
                                            output_shapes=(tf.TensorShape([None,encoded_shape]),tf.TensorShape([])))\
                    .map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                    .batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    ds_test=tf.data.Dataset.from_generator(generator=testgen, output_types=(tf.float32, tf.int32),\
                                            output_shapes=(tf.TensorShape([None,encoded_shape]),tf.TensorShape([])))\
                    .map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                    .batch(1).prefetch(tf.data.experimental.AUTOTUNE)

    model_dir = './results' + task +'/'+runningcode + 'K' + str(args.K) + '/' + 'fold' + str(fold) + '/model'
    model = models.load_model(model_dir)
    model.trainable = False

    report = []
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
    print('AUC: ', roc_auc_score(gts, probs))
    print('acc: ', accuracy_score(gts, predictions))
    print('precision: {:.4}, recall: {}, f1: {}, spe:{}'.format(prf[0], prf[1], prf[2], mat[0, 0] / mat[0].sum()))
    report.append([accuracy_score(gts, predictions), roc_auc_score(gts, probs), prf[0], prf[1], prf[2], mat[0, 0] / mat[0].sum()])

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
    print('AUC: ', roc_auc_score(gts, probs))
    print('acc: ', accuracy_score(gts, predictions))
    print('precision: {:.4}, recall: {}, f1: {}, spe:{}'.format(prf[0], prf[1], prf[2], mat[0, 0] / mat[0].sum()))
    report.append([accuracy_score(gts, predictions), roc_auc_score(gts, probs), prf[0], prf[1], prf[2], mat[0, 0] / mat[0].sum()])

    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='HE')
    parser.add_argument('--runningcode', type=str, default='bcrnet')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--K', type=int, default=5000)

    args = parser.parse_args()
    
    print('task: ', args.task)
    print('experiment: ', args.runningcode)

    train_folds, train_acc, train_auc, train_pre, train_recall, train_f1, train_spe = [], [], [], [], [], [], []
    test_folds, test_acc, test_auc, test_pre, test_recall, test_f1, test_spe = [], [], [], [], [], [], []
    for i in range(36):
        print('*****************')
        args.fold = i
        print('fold: ', args.fold)
        report = run(args.task, args.runningcode, args.fold, args)
        train_folds.append(str(args.fold))
        train_acc.append(report[0][0])
        train_auc.append(report[0][1])
        train_pre.append(report[0][2])
        train_recall.append(report[0][3])
        train_f1.append(report[0][4])
        train_spe.append(report[0][5])

        test_folds.append(str(args.fold))
        test_acc.append(report[1][0])
        test_auc.append(report[1][1])
        test_pre.append(report[1][2])
        test_recall.append(report[1][3])
        test_f1.append(report[1][4])
        test_spe.append(report[1][5])

    final_df = pd.DataFrame({'folds': train_folds, 'acc': train_acc, 'auc': train_auc, 'precision': train_pre, \
    'recall': train_recall, 'f1': train_f1, 'specificity': train_spe})
    final_df.to_csv('./results' + args.task +'/'+args.runningcode + 'K' + str(args.K) + '/' + 'trainsummary.csv')
    print(final_df.describe()[1:3])

    final_df = pd.DataFrame({'folds': test_folds, 'acc': test_acc, 'auc': test_auc, 'precision': test_pre, \
    'recall': test_recall, 'f1': test_f1, 'specificity': test_spe})
    final_df.to_csv('./results' + args.task +'/'+args.runningcode + 'K' + str(args.K) + '/' + 'testsummary.csv')
    print(final_df.describe()[1:2])


