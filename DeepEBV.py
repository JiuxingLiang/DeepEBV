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
np.random.seed(1337)  # for reproducibility
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
from sklearn.metrics import fbeta_score, roc_curve, auc, roc_auc_score, average_precision_score,precision_recall_curve
import matplotlib.pyplot as plt
from keras.regularizers import l2, l1, l1_l2
from keras.models import Model, load_model
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
from pandas import merge
from keras.engine import InputSpec
from keras.utils import CustomObjectScope
from keras.layers.merge import concatenate
import keras
from keras.callbacks import TensorBoard
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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




def Build_model():
    print('building model')
    
    #Define_hyperparameters
    seq_input_shape = (2000, 4)
    nb_filter = 64
    nb_filter1 = 128
    filter_length = 8
    filter_length1 = 7
    input_shape = (2000, 4)
    attentionhidden = 256

    #Define_Input_layer
    seq_input = Input(shape=seq_input_shape, name='seq_input')
    
    #DEfine_Convolution_Layer
    convol_1 = Convolution1D(filters=nb_filter, kernel_size=filter_length, padding='valid', activation='relu',
                            kernel_constraint=maxnorm(4), subsample_length=1)
    convol_2 = Convolution1D(filters=nb_filter, kernel_size=filter_length, padding='valid', activation='relu',
                            kernel_constraint=maxnorm(4), subsample_length=1)
    convol_3 = Convolution1D(filters=nb_filter1, kernel_size=filter_length1, padding='valid', activation='relu',
                            kernel_constraint=maxnorm(4), bias_constraint = maxnorm(4),subsample_length=1)
    
    #Define_Pooling_Layer
    pooling_1 = MaxPooling1D(pool_size=3)
    pooling_2 = MaxPooling1D(pool_size=3)
    
    #Define_Dropout_Layer
    dropout_1 = Dropout(0.50)  
    dropout_2 = Dropout(0.45)
    
    #Define_Attention_Layer
    decoder = Attention(hidden=attentionhidden, activation='linear')  # attention_layer
    
    #Define_Dense_Layer
    dense_1 = Dense(1)   
    dense_2 = Dense(1)
    
    #Feature_Extraction_Module
    output_1 = pooling_1(convol_2(convol_1(seq_input)))  
    output_12 = pooling_2(convol_3(output_1))
    output_2 = dropout_1(output_12)
    output_3 = dense_1(dropout_2(Flatten()(output_2)))

    #Attention_Module
    att_decoder = decoder(output_2)  
    output_4 = attention_flatten(output_2._keras_shape[2])(att_decoder)
      
    #concatenate_Layer
    all_outp = merge([output_3, output_4], mode='concat')
    output_5 = dense_2(all_outp)

    #Classify_Layer
    output_f = Activation('sigmoid')(output_5)


    model = Model(inputs=seq_input, outputs=output_f)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    return model


def load_Data():
    #load Data
    dsVIS_test_Data = np.load('EBVdata/dsVIS_Data/dsVIS_Test_Data.npy')
    dsVIS_test_Label = np.load('EBVdata/dsVIS_Data/dsVIS_Test_label.npy')
    dsVIS_test_Label.astype(int)#Change the data format
    VISDB_test_Data = np.load('EBVdata/VISDB_Data/VISDB_Test_Data.npy')
    VISDB_test_Label = np.load('EBVdata/VISDB_Data/VISDB_Test_Label.npy')
    VISDB_test_Label.astype(int)  # Change the data format 
    return dsVIS_test_Data, dsVIS_test_Label, VISDB_test_Data, VISDB_test_Label


def test_model():

       

    print('testing_model')
    
    #build Model structure 
    model = Build_model()
    #Read model parameters and integrate them into the model
    model.load_weights('Model/DeepEBV_with_EBV_integration_sequences.hdf5')

    #Show model structure
    model.summary()

    dsVIS_test_Data, dsVIS_test_Label, VISDB_test_Data, VISDB_test_Label = load_Data()
    
    print('Predicting...')
    dsVIS_test_pred = model.predict(dsVIS_test_Data)
    VISDB_test_pred = model.predict(VISDB_test_Data)
    #Labels_pred = np.round(Label_pred)
    #Labels_pred = Labels_pred.astype(int)
    
    #Accuracy and loss_value
    dsVIS_Acc_loss = model.evaluate(dsVIS_test_Data, dsVIS_test_Label, batch_size=128) 
    VISDB_Acc_loss = model.evaluate(VISDB_test_Data, VISDB_test_Label, batch_size=128) 
    #ROC
    dsVIS_rocauc_score = roc_auc_score(dsVIS_test_Label, dsVIS_test_pred)
    VISDB_rocauc_score = roc_auc_score(VISDB_test_Label, VISDB_test_pred)
    #Data_process
    dsVIS_Test_Quantity = pd.value_counts(dsVIS_test_Label)
    # print('Test_neg_Quantity and Test_pos_Quantity', Test_Quantity._ndarray_values)
    dsVIS_neg_Quantity = dsVIS_Test_Quantity._ndarray_values[0]
    dsVIS_pos_Quantity = dsVIS_Test_Quantity._ndarray_values[1]
    VISDB_Test_Quantity = pd.value_counts(VISDB_test_Label)
    # print('Test_neg_Quantity and Test_pos_Quantity', Test_Quantity._ndarray_values)
    VISDB_neg_Quantity = VISDB_Test_Quantity._ndarray_values[0]
    VISDB_pos_Quantity = VISDB_Test_Quantity._ndarray_values[1]

    #AUPR    
    dsVIS_AveragePrecision_score = average_precision_score(dsVIS_test_Label, dsVIS_test_pred)
    VISDB_AveragePrecision_score = average_precision_score(VISDB_test_Label, VISDB_test_pred)
    """
        #pre,rec,t = precision_recall_curve(Label_test, Label_pred)
        #print('AUC',auc(rec, pre))
        Test_Quantity = pd.value_counts(Label_test)
        #print('Test_neg_Quantity and Test_pos_Quantity', Test_Quantity._ndarray_values)
        Test_neg_Quantity = Test_Quantity._ndarray_values[0]
        Test_pos_Quantity = Test_Quantity._ndarray_values[1]
        print('HPV_pos', Test_pos_Quantity)
        print('HPV_neg',Test_neg_Quantity)
        #A = pd.count(Labels_pred[0:Test_pos_Quantity])
        pos_pred = Labels_pred[0:Test_pos_Quantity-1]                   
        neg_pred = Labels_pred[Test_pos_Quantity:Test_pos_Quantity+Test_neg_Quantity-1]
        Testpos_Quantity = np.sum(pos_pred)
        Testneg_Quantity = np.sum(neg_pred)
        print('FP：', Testneg_Quantity)
        Testneg_Quantity = Test_neg_Quantity-Testneg_Quantity
        #neg_pred.astype(float)
        #A = neg_pred[0]
        #neg_pred[neg_pred==1] = 1.0
        #pos_Quantity = pd.value_counts(pos_pred)
        #Testpos_Quantity = pos_Quantity._ndarray_values[1]
        #neg_Quantity = pd.value_counts(neg_pred)
        #Testneg_Quantity = neg_Quantity._ndarray_values[1]
        True_Posatives = Testpos_Quantity/Test_pos_Quantity
        True_Negatives = Testneg_Quantity/Test_neg_Quantity
        Faste_Posatives = 1-True_Posatives
        Faste_Negatives = 1-True_Negatives
        print('True_Posatives：', True_Posatives)
        print('Faste_Posatives：', Faste_Posatives)
        print('True_Negatives：',True_Negatives)
        print('Faste_Negatives：', Faste_Negatives)
        # Label_pred = []
        # for item in y_score:
        # Label_pred.append(item[0])
        # Label_pred =  np.array(Label_pred)
        # ensemble += Label_pred
        # #true_negatives = tf.metrics.true_negatives_at_thresholds(Label_test,Labels_pred,thresholds=[0, 1])
        #print('true_negatives',true_negatives)
        #true_positives = tf.metrics.true_positives(Label_test,Labels_pred)
        #print('true_positives',true_positives)
        # ensemble /= n_estimators

        np.save('test_result/Label_test', Label_test)
        # np.save('test_result/Label_pred', ensemble)
        #np.save('test_result/Label_test', Labels_pred)
        #np.save('test_result/Label_test', Label_test)
        np.save('test_result/Label_pred', Label_pred)
    """

    #Results
    print('-------dsVIS_test_result------------------')
    print('dsVIS_Test_pos_Quantity:',dsVIS_pos_Quantity)
    print('dsVIS_Test_neg_Quantity:',dsVIS_neg_Quantity)
    print('Test acc:', dsVIS_Acc_loss[1])
    print('Test loss:', dsVIS_Acc_loss[0])
    print('auroc:', dsVIS_rocauc_score)
    print('aupr:', dsVIS_AveragePrecision_score)
    print('-------VISDB_test_result------------------')
    print('dsVIS_Test_pos_Quantity:',VISDB_pos_Quantity)
    print('dsVIS_Test_neg_Quantity:',VISDB_neg_Quantity)
    print('Test acc:', VISDB_Acc_loss[1])
    print('Test loss:', VISDB_Acc_loss[0])
    print('auroc:', VISDB_rocauc_score)
    print('aupr:', VISDB_AveragePrecision_score)



    #save_data
    output_directory = 'Test_Result/dsVIS_Result/'
    model_test_results = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['dsVIS_Test_pos_Quantity', 'dsVIS_Test_neg_Quantity', 
                                 'Test_acc','Test_loss','rocauc_score', 'AveragePrecision_score'])
    model_test_results['dsVIS_Test_pos_Quantity'] = dsVIS_pos_Quantity
    model_test_results['dsVIS_Test_neg_Quantity'] = dsVIS_neg_Quantity
    model_test_results['Test_acc'] = dsVIS_Acc_loss[1]
    model_test_results['Test_loss'] = dsVIS_Acc_loss[0]
    model_test_results['rocauc_score'] = dsVIS_rocauc_score
    model_test_results['AveragePrecision_score'] = dsVIS_AveragePrecision_score
    model_test_results.to_csv(output_directory + 'model_test_results.csv', index=False)

    output_directory = 'Test_Result/VISDB_Result/'
    model_test_results = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                      columns=['VISDB_Test_pos_Quantity', 'VISDB_Test_neg_Quantity', 
                                      'Test_acc','Test_loss', 'rocauc_score', 'AveragePrecision_score'])
    model_test_results['VISDB_Test_pos_Quantity'] = VISDB_pos_Quantity
    model_test_results['VISDB_Test_neg_Quantity'] = VISDB_neg_Quantity
    model_test_results['Test_acc'] = VISDB_Acc_loss[1]
    model_test_results['Test_loss'] = VISDB_Acc_loss[0]
    model_test_results['rocauc_score'] = VISDB_rocauc_score
    model_test_results['AveragePrecision_score'] = VISDB_AveragePrecision_score
    model_test_results.to_csv(output_directory + 'model_test_results.csv', index=False)



if __name__ == '__main__':


    test_model()
