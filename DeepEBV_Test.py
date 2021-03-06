from __future__ import print_function, division
import numpy as np
import h5py
import scipy.io
import random
import sys, os
import itertools
import numbers
from collections import Counter
from warnings import warn
from abc import ABCMeta, abstractmethod
import tensorflow as tf

# from keras.layers import merge  # works
from keras.layers import merge
from keras.optimizers import RMSprop, SGD
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation, Flatten
import keras.layers.core as core
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, multiply, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding
from sklearn.metrics import fbeta_score, roc_curve, auc, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from keras.regularizers import l2, l1, l1_l2
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec
from keras.utils import CustomObjectScope
from keras.layers.merge import concatenate
import keras
from keras.callbacks import TensorBoard
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Choise GPU


class Attention(Layer):
    def __init__(self, hidden, init='glorot_uniform', activation='linear', W_regularizer=None, b_regularizer=None,
                 W_constraint=None, **kwargs):
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.hidden = hidden
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.input_length = input_shape[1]
        self.W0 = self.add_weight(name='{}_W1'.format(self.name), shape=(input_dim, self.hidden),
                                  initializer='glorot_uniform', trainable=True)  # Keras 2 API
        self.W = self.add_weight(name='{}_W'.format(self.name), shape=(self.hidden, 1), initializer='glorot_uniform',
                                 trainable=True)
        self.b0 = K.zeros((self.hidden,), name='{}_b0'.format(self.name))
        self.b = K.zeros((1,), name='{}_b'.format(self.name))
        self.trainable_weights = [self.W0, self.W, self.b, self.b0]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W0] = self.W_constraint
            self.constraints[self.W] = self.W_constraint

        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        attmap = self.activation(K.dot(x, self.W0) + self.b0)
        attmap = K.dot(attmap, self.W) + self.b
        attmap = K.reshape(attmap, (-1, self.input_length))  # Softmax needs one dimension
        attmap = K.softmax(attmap)
        dense_representation = K.batch_dot(attmap, x, axes=(1, 1))
        out = K.concatenate([dense_representation,
                             attmap])  # Output the attention maps but do not pass it to the next layer by DIY flatten layer
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1] + input_shape[1])

    def get_config(self):
        config = {'init': 'glorot_uniform',
                  'activation': self.activation.__name__,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'hidden': self.hidden if self.hidden else None}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class attention_flatten(Layer):  # Based on the source code of Keras flatten
    def __init__(self, keep_dim, **kwargs):
        self.keep_dim = keep_dim
        super(attention_flatten, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to "Flatten" '
                            'is not fully defined '
                            '(got ' + str(input_shape[1:]) + '. '
                                                             'Make sure to pass a complete "input_shape" '
                                                             'or "batch_input_shape" argument to the first '
                                                             'layer in your model.')
        return (input_shape[0], self.keep_dim)  # Remove the attention map

    def call(self, x, mask=None):
        x = x[:, :self.keep_dim]
        return K.batch_flatten(x)


def set_up_model_up():
    print('building model')

    seq_input_shape = (2000, 4)
    nb_filter = 128
    nb_filter1 = 128
    filter_length = 8
    filter_length1 = 6
    attentionhidden = 256

    seq_input = Input(shape=seq_input_shape, name='seq_input')
    convul1 = Convolution1D(filters=nb_filter, kernel_size=filter_length, padding='valid', activation='relu',
                            kernel_constraint=maxnorm(3), subsample_length=1)
    convul11 = Convolution1D(filters=nb_filter1, kernel_size=filter_length1, padding='valid', activation='relu',
                             kernel_constraint=maxnorm(3), bias_constraint=maxnorm(3), subsample_length=1)

    pool_ma1 = MaxPooling1D(pool_size=2)
    pool_ma11 = MaxPooling1D(pool_size=3)

    dropout1 = Dropout(0.50)
    dropout2 = Dropout(0.45)
    decoder = Attention(hidden=attentionhidden, activation='linear')  # Attention decoder
    dense1 = Dense(1)  #
    dense2 = Dense(1)

    output_1 = pool_ma1(convul1(seq_input))
    output_12 = pool_ma11(convul11(output_1))

    output_2 = dropout1(output_12)

    att_decoder = decoder(output_2)
    output_3 = attention_flatten(output_2._keras_shape[2])(att_decoder)
    output_4 = dense1(dropout2(Flatten()(output_2)))

    # all_outp = merge([output_3, output_4], mode='concat') # ?????????
    # all_outp = merge([output_3, output_4], mode='Sum')    # ?????????????????????????????????
    all_outp = concatenate([output_3, output_4])
    output_5 = dense2(all_outp)
    output_f = Activation('sigmoid')(output_5)

    model = Model(inputs=seq_input, outputs=output_f)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    return model


def test_model():
    Data_test = np.load('Data/Data_test.npy')  # Test_data
    Label_test = np.load('Data/Label_test.npy')  # Test_label
    Label_test.astype(int)  # Change the data format

    print('testing_model')
    model = set_up_model_up()
    model.load_weights('Model/DeepEBV_model.hdf5')

    model.summary()  # Show model structure

    print('Predicting...')
    Label_pred = model.predict(Data_test)
    Labels_pred = np.round(Label_pred)
    Labels_pred = Labels_pred.astype(int)

    # print('loss_and_acc...')
    loss_and_metrics = model.evaluate(Data_test, Label_test, batch_size=128)
    print('Test loss', loss_and_metrics[0])  # loss and acc
    print('Test acc:', loss_and_metrics[1])

    test_pred = Label_pred
    test_label = Label_test
    np.save('Pred_Result/test_label', test_label)
    np.save('Pred_Result/test_pred', test_pred)

    rocauc_score = roc_auc_score(Label_test, Label_pred)
    AveragePrecision_score = average_precision_score(Label_test, Label_pred)

    print('auroc', rocauc_score)
    print('aupr', AveragePrecision_score)

    # Save the excel
    output_directory = 'Pred_Result/'
    model_test_results = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                                      columns=['model_Test_loss', 'model_Test_acc', 'rocauc_score',
                                               'AveragePrecision_score'])
    model_test_results['best_model_Test_loss'] = loss_and_metrics[0]
    model_test_results['best_model_Test_acc'] = loss_and_metrics[1]
    model_test_results['rocauc_score'] = rocauc_score
    model_test_results['AveragePrecision_score'] = AveragePrecision_score

    model_test_results.to_csv(output_directory + 'model_test_results.csv', index=False)


if __name__ == '__main__':
    test_model()
