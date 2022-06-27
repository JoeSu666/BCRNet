import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras import datasets, layers, models


INIT = 'glorot_uniform'
class gatedattention(layers.Layer):
    def __init__(self, channels=64, **kwargs):
        super(gatedattention, self).__init__(**kwargs)
        self.channels = channels
        self.V = tfa.layers.WeightNormalization(layers.Dense(channels,use_bias=False, kernel_initializer=INIT))
        self.U = tfa.layers.WeightNormalization(layers.Dense(channels,use_bias=False, kernel_initializer=INIT))
        
        self.Wa = layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(1e-5),use_bias=False, kernel_initializer=INIT)
        self.softmax = layers.Softmax(axis=1)
        self.dot = layers.Dot(axes=1)

    def call(self, x):
        x = x[0]
        V = tf.keras.activations.tanh(self.V(x))
        U = tf.keras.activations.sigmoid(self.U(x))
        energy = tf.math.multiply(V,U)

        x = tf.expand_dims(x,0)
        att = tf.expand_dims(self.Wa(energy),0)
        att = self.softmax(att)  

        hs = self.dot([att,x]) # 1,vector_size       
        hs = tf.squeeze(hs,1)
        return att, hs
        
    def get_config(self):
        config = super(gatedattention, self).get_config()
        config.update({'channels':self.channels})
        return config
        
class AttMILbinary(models.Model):
    def __init__(self):
        super(AttMILbinary, self).__init__()

    def build(self, inputshape):
        self.inputshape = inputshape

        self.gatedattention = gatedattention(inputshape[-1]//2, name='attention')
        # self.dot = layers.Dot(axes=1)

        self.WC = layers.Dense(1,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.00001), kernel_initializer=INIT)

        super(AttMILbinary,self).build(inputshape)

    def call(self, x):
        att, hs = self.gatedattention(x)
        hs = layers.Dropout(rate=0.1)(hs)
        s = self.WC(hs)
        
        return s
        
    def get_config(self):
        config = super(AttMILbinary, self).get_config()
        config.update({'inputshape':self.inputshape})
        return config