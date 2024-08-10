# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:42:48 2023

@author: xincao
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from keras.models import Model
from tensorflow.keras.layers import Input, InputLayer, Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import scipy.stats as st
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import bootstrap
print(tf.keras.backend.image_data_format())
import os
import keras
import keras.backend as K
from keras.layers import Conv2D,MaxPool1D, MaxPooling1D, AveragePooling1D, MaxPool2D, GlobalAveragePooling1D, GlobalMaxPool1D, Input, Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, BatchNormalization, concatenate, Activation, Multiply, Permute, Reshape, Lambda, Add,multiply,Flatten
from keras.models import Model
from keras.initializers import RandomUniform,glorot_normal,VarianceScaling
from tensorflow.keras.layers import Attention
from keras.callbacks import ReduceLROnPlateau

data2=np.load('crisprSQL_723_format.npz')
# data2=np.load('Cpf1Offtarget_723_format.npz') #
inputRow = 7
inputCol = 23
data2.files
X=data2['d1']
Y=data2['d2']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

def transformImages(
        xtrain, xtest,
        ytrain, ytest,
        imgrows, imgcols,
                    ):
    if tf.keras.backend.image_data_format() == 'channels_first':
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (1, imgrows, imgcols)
    else:
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (imgrows * imgcols)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    return xtrain, xtest, ytrain, ytest, input_shape
xtrain, xtest, ytrain, ytest, input_shape = transformImages(xtrain,xtest,ytrain,ytest,inputRow,inputCol)
X2 = X.transpose(0,2,1)
X = X2.reshape((len(X), inputRow*inputCol))

VOCAB_SIZE = 7
EMBED_SIZE = 90
MAXLEN = 161

def residual_block(x, num_filters):
    res = x
    x = Conv1D(num_filters, kernel_size=5, padding='same', activation='relu')(x)
    # x = Conv1D(num_filters, kernel_size=5, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(num_filters, kernel_size=5, padding='same', activation='relu')(x)
    # x = Conv1D(num_filters, kernel_size=5, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([res, x])
    x = Activation('relu')(x) 
    return x

def CRISPR_HW():
    input = Input(shape=(input_shape,)) #(161,)
    embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)
    conv1 = Conv1D(70, 4, activation="relu", name="conv1")(embedded)
    batchnor1 = BatchNormalization()(conv1)
    conv2 = Conv1D(40, 6, activation="relu", name="conv2")(batchnor1)
    batchnor2 = BatchNormalization()(conv2)
    batchnor2 = Dropout(0.35)(batchnor2)

    #---------Resnet---------#
    c1 = residual_block(batchnor2, 40)
    c11 = BatchNormalization()(c1)

    # ---------LSTM---------#
    c22 = Bidirectional(LSTM(20, return_sequences=True, activation='relu'))(batchnor2)
    c22 = BatchNormalization()(c22)

    # ---------Attention---------#
    c31 = Conv1D(40, 9, activation='relu', name="c31")(embedded)
    batchnor3 = BatchNormalization()(c31)
    c32 = Attention()([batchnor2, batchnor3])


    merged = concatenate([c11, c22, c32])
    flat = Flatten()(merged)

    dense1 = Dense(600, activation="relu", name="dense1")(flat)
    # drop1 = Dropout(0.2)(dense1)
    drop1 = (dense1)

    dense2 = Dense(200, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.5)(dense2)

    output = Dense(1, activation="linear", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])
    return model

loss = []
validation_loss = []
pearson = []
spearman = []
# epoch number (EN); batch size (BS); learning rate (LR)
EN = 30
BS = 200
LR = 0.001

model = CRISPR_HW()
model.summary()
model.compile(
            optimizer=keras.optimizers.Adam(learning_rate = LR),
            loss = 'mse',
            # metrics=['mse']
            )
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)
hist = model.fit(xtrain, ytrain,
                     batch_size = BS,
                     epochs = EN,
                     validation_data = (xtest, ytest),
                     verbose = 1,
                     callbacks=[reduce_lr]
                     )

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
plt.title('FNN5 Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
print("pearson: ", np.squeeze(pearson))
print("spearman: ", np.squeeze(spearman))

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
num_samples = 100
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
