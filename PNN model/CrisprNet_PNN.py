# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:56:36 2023

@author: xincao
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from tensorflow.keras.layers import Input, InputLayer, Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.layers import Reshape, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import bootstrap
print(tf.keras.backend.image_data_format())

data2=np.load('crisprSQL_723_format.npz')
# data2=np.load('Cpf1Offtarget_723_format.npz') # EN=90, BS=100
inputRow = 7
inputCol = 23
data2.files
X=data2['d1']
Y=data2['d2']
X = X.reshape((len(X),1,inputCol,inputRow))
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')

def conv2d_bn(x, filters, kernel_size, strides=1,
              padding='same', activation='relu',
              use_bias=True, name=None, trainable=True):
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding=padding, use_bias=use_bias,
                      name=name, trainable=trainable)(x)
    x = layers.BatchNormalization(axis=-1,scale=True)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x

loss = []
validation_loss = []
pearson = []
spearman = []
# epoch number (EN); batch size (BS); learning rate (LR)
EN = 100
BS = 200
LR = 0.0001

def Model():    
    input_layer = Input(shape=(1,23,7), name='main_input')
    branch_0 = conv2d_bn(input_layer, 10, (1, 1))
    branch_1 = conv2d_bn(input_layer, 10, (1, 2))
    branch_2 = conv2d_bn(input_layer, 10, (1, 3))
    branch_3 = conv2d_bn(input_layer, 10, (1, 5))
    branches = [input_layer, branch_0, branch_1, branch_2, branch_3]
    mixed = layers.Concatenate(axis=-1)(branches)
    mixed = Reshape((23, 47))(mixed)
    x = Dropout(0.25)(mixed)
    x = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 47), name="LSTM_out"))(x)
    # x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(80, activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='linear', name='main_output')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)

model = Model()
model.summary()
model.compile(
            optimizer=keras.optimizers.Adam(learning_rate = LR),
            loss = 'mse',
            )
hist = model.fit(xtrain, ytrain,
                     batch_size = BS,
                     epochs = EN,
                     validation_data = (xtest, ytest),
                     verbose = 1 )#, callbacks=[es]) 

loss.append(hist.history['loss'])
validation_loss.append([hist.history['val_loss']])
predictions = model.predict(xtest)
coef, p = spearmanr(np.squeeze(ytest), np.squeeze(predictions))
corr, p1 = pearsonr(np.squeeze(ytest), np.squeeze(predictions))
pearson.append(corr)
spearman.append(coef)

plt.figure()
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('mcdCNet Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

def MCD(X, model, num_samples):
    preds = [model(X, training=True) for _ in range(num_samples)]
    mcSamples = np.hstack(preds)
    return mcSamples 

def conf_interval(mcd_prediction, conf_level_index):
    bootstrapLowerBound = np.quantile(mcd_prediction, 0.5*(1-conf_level_index), axis=1)-stats.tstd(mcd_prediction,axis=1)
    bootstrapUpperBound = np.quantile(mcd_prediction, 1-(0.5*(1-conf_level_index)), axis=1)+stats.tstd(mcd_prediction,axis=1)
    for i in range(len(bootstrapLowerBound)):
        if bootstrapLowerBound[i] < -4:
            bootstrapLowerBound[i] = -4
        if bootstrapUpperBound[i] > 4:
            bootstrapUpperBound[i] = 4
    return bootstrapLowerBound, bootstrapUpperBound

def conf_int_counter(mcd_prediction, conf_level_index, y_label):
    counter = 0
    boundaryLow, boundaryUp = conf_interval(mcd_prediction, conf_level_index)
    for i in range(len(y_label)):
        if boundaryLow[i] <= y_label[i] and boundaryUp[i] >= y_label[i]:
            counter += 1   
    percentage = counter / len(y_label) * 100
    print(percentage)
    return percentage

# how many times forward passes (BNN samples number)
num_samples = 1000
mcPred = MCD(xtest, model, num_samples)

Expected_Confidence_Level = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.683,0.7,0.8,0.9,0.955,0.9973]
ob_conf_level_counter = []
for i in range(len(Expected_Confidence_Level)):
    ob_conf_level_counter.append(conf_int_counter(mcPred, Expected_Confidence_Level[i], ytest))

plt.style.use('seaborn')
fig = plt.figure()
ax = fig.add_subplot(111)
a1 = np.arange(0,101,1)
plt.plot(a1,a1,'k--')
xaxis = [i*100 for i in Expected_Confidence_Level]

plt.plot(xaxis, ob_conf_level_counter,color="blue",marker='*')
plt.plot([xaxis[7],xaxis[11],xaxis[12]],
            [ob_conf_level_counter[7],
              ob_conf_level_counter[11],
              ob_conf_level_counter[12]],'D',color="red")
ax.set_aspect('equal', adjustable='box')
plt.xlabel('Expected Confidence Level')
plt.ylabel('Observed Confidence Level')
plt.title('BNN confidence plot based on CRISPR-Net/LSTM')
plt.legend(['Ground Truth', 'BNN performance', 'Three Sigma'], loc='upper left')

# define how many data points want to show (10)
data_num = 10
designed_conf_level=0.8
plt.figure()
dp_ini = int(len(ytest)/data_num)+0
dp_index = [i*dp_ini for i in range(1,data_num+1,1)] # i=1~10
dp_y_label = ytest[dp_index]
dp_y_pred = mcPred[dp_index]
dplow, dphigh = conf_interval(dp_y_pred,designed_conf_level)
xaxis2=np.linspace(1,data_num,data_num) # 1~10
plt.plot(xaxis2,dp_y_label, color='green', marker='o', linestyle='--')
plt.fill_between(xaxis2,dplow,dphigh,color='green',alpha=0.2)
plt.legend(['Label', 'Predicted conf_interval'], loc='upper left')
plt.xlabel('Individual samples')
plt.ylabel("Sample's cleavage frequency")
plt.title('BNN forcasets with ' + '%s'%designed_conf_level +' confidence level (based on CRISPR-Net/LSTM)')
