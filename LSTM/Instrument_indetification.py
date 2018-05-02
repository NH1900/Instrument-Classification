# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:23:22 2017

@author: Naussica28
"""
from skimage.io import imshow, imread
import matplotlib.pyplot as plt

import numpy as np
import random
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Activation, Dense,LSTM
from keras.optimizers import Adam


TIME_STEPS = 5     # same as the height of the image
INPUT_SIZE = 21     # same as the width of the image
BATCH_SIZE = 10
BATCH_INDEX = 0
OUTPUT_SIZE = 3
CELL_SIZE = 20
LR = 0.001

def data_splitting(data):
    lenx,leny = data.shape
    length = leny - 900
    for i in range(length):
        data = np.delete(data,-1,1)
    return data

def convert2batch(data,frame_number):
    lenx,leny = data.shape
    number_instances = int(leny / frame_number)
    data_matrix = np.zeros(((number_instances,lenx,frame_number)))
    for i in range(number_instances):
        data_matrix[i,:,:] = data[:,(0 +  frame_number * i) : (frame_number + frame_number * i) ]
    return data_matrix
    
def training_tunning_data(data_batch,tune_size):
    len_batch,lenx,leny = data_batch[:,:,:].shape
    tune_matrix = np.zeros(((tune_size,lenx,leny)))
    train_matrix = np.zeros((( (len_batch - tune_size),lenx,leny )))
    for i in range(tune_size):
        tune_matrix[i,:,:] = data_batch[len_batch - (tune_size - i),:,:]
    for j in range(len_batch - tune_size):
        train_matrix[j,:,:] = data_batch[j,:,:]
    return train_matrix,tune_matrix

def combine_labelize(data1,data2,data3):#output randomly distributed dataset
    len_batch,lenx,leny = data1.shape
    batch = len_batch*3
    combine_matrix = np.zeros(((batch,lenx,leny)))
    combine_matrix_random = np.zeros(((batch,lenx,leny)))
    label_matrix_random = np.zeros(batch)
    data_map = dict()
    for m in range(3):
        if m not in data_map.keys():
            if m == 0:
                data_map[m] = data1
            elif m == 1:
                data_map[m] = data2
            elif m == 2:
                data_map[m] = data3
        for i in range(len_batch):
            combine_matrix[(i + len_batch * m),:,:] = data_map[m][i,:,:]
                
    index = np.array(range(3 * len_batch))    
    data_dict = dict()
    for j in index:#do a mapping of id and actual label 
        if j not in data_dict.keys():
            if j <= (len_batch - 1):
                data_dict[j] = 1
            elif j <= (2*len_batch - 1):
                data_dict[j] = 2
            elif j <= (3*len_batch - 1):
                data_dict[j] = 3
    random.shuffle(index)
    for n in range(batch):
        combine_matrix_random[n,:,:] = combine_matrix[(index[n]),:,:]
        label_matrix_random[n] = data_dict[index[n]]
    return label_matrix_random,combine_matrix_random,index

def convert2int(label):#for label
    if len(label.shape) == 1:
        label_matrix = np.zeros(len(label))
        label_matrix.dtype = 'int64'
        for i in range(len(label)):
            label_matrix[i] = int(label[i])
    elif len(label.shape) == 2:
        lenx,leny = label.shape
        label_matrix = np.zeros(label.shape)
        label_matrix.dtype = 'int64'
        for ii in range(lenx):
            for jj in range(leny):
                label_matrix[ii,jj] = int(label[ii,jj])
    return label_matrix

def normalize(data):
    len_batch,lenx,leny = data.shape
    for i in range(len_batch):
        for j in range(leny):
            (data[i,:,j] - data[i,:,j].mean()) / data[i,:,j].var()
    return data
                
def one_hot(label,instance_size,onehot_number):#onehot number equals to the number of classes
    onehot_matrix = np.zeros((instance_size,onehot_number))
    for i in range(instance_size):
        if label[i] == 1:
            onehot_matrix[i,0] = 1
        elif label[i] == 2:
            onehot_matrix[i,1] = 1
        elif label[i] == 3:
            onehot_matrix[i,2] = 1    
    return onehot_matrix

def batch_transpose(data):
    len_batch,lenx,leny = data.shape
    new_data = np.zeros(((len_batch,leny,lenx)))
    for i in range(len_batch):
        new_data[i,:,:] = np.transpose(data[i,:,:])
    return new_data

def test_predict(test):
    len_batch,lenx,leny = test.shape
    preds = np.zeros((len_batch,3))
    for step2 in range(int(len_batch/BATCH_SIZE)):
        preds[0 + step2 * BATCH_SIZE : 10 + step2 * BATCH_SIZE,:] = model.predict(batch_transpose(test))#[0 + step2 * BATCH_SIZE : 10 + step2 * BATCH_SIZE,:]
    return preds

def show_image(preds):
    number_instance,onehot = preds.shape
    for i in range(number_instance):
        plt.figure(i,figsize =  (2,2))
        plt.axis("off")
        vector = preds[i,:]
        index = np.where(vector[:] == vector.max())
        if index[0] == 0:
            instrument = imread("Clarinet.jpg")
            imshow(instrument,aspect = "auto")
        elif index[0] == 1:
            instrument = imread("Flute.jpg")
            imshow(instrument,aspect = "auto")
        elif index[0] == 2:
            instrument = imread("Trumpet.jpg")
            imshow(instrument,aspect = "auto")
      
def test(T,F,B):
    test = np.zeros(((10,21,5)))
    test[0,:,:] = T[171,:,:]
    test[1,:,:] = B[159,:,:]
    test[2,:,:] = B[164,:,:]
    test[3,:,:] = T[174,:,:]
    test[4,:,:] = B[152,:,:]
    test[5,:,:] = F[161,:,:]
    test[6,:,:] = F[154,:,:]
    test[7,:,:] = T[166,:,:]
    test[8,:,:] = B[162,:,:]
    test[9,:,:] = F[159,:,:]
    return test

#load data
train_BbC = np.loadtxt('BbCldat.txt')
train_Flute = np.loadtxt('Flutedat.txt')
train_Trump = np.loadtxt('Trumpetdat.txt')

#dataset splitting
train_Flute = data_splitting(train_Flute)
train_Trump = data_splitting(train_Trump)

#convert to 3 dimensional matrix
Bbc_batch = convert2batch(train_BbC,5)
Flute_batch = convert2batch(train_Flute,5)
Trump_batch = convert2batch(train_Trump,5)

#create tune&train data 
tune_size = 30
Bbc_train,Bbc_tune = training_tunning_data(Bbc_batch,tune_size)
Flute_train,Flute_tune = training_tunning_data(Flute_batch,tune_size)
Trump_train,Trump_tune = training_tunning_data(Trump_batch,tune_size)

#combine dataset
train_label,train_data,train_index = combine_labelize(Bbc_train,Flute_train,Trump_train)
tune_label,tune_data,tune_index = combine_labelize(Bbc_tune,Flute_tune,Trump_tune)
train_data_transpose = batch_transpose(train_data)
tune_data_transpose = batch_transpose(tune_data)
train_label_int = convert2int(train_label)
tune_label_int = convert2int(tune_label)

#normalize the data
train_norm = normalize(train_data_transpose)
tune_norm = normalize(tune_data_transpose)

#one hot key
train_onehot = one_hot(train_label,len(train_label),3)
tune_onehot = one_hot(tune_label,len(tune_label),3)

#build RNN model
model = Sequential()

##RNN cell
model.add(LSTM(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim = CELL_SIZE,
    unroll=True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
for step in range(4501):
    # data shape = (batch_num, steps, inputs/outputs)
    train_batch = train_norm[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    train_label_batch = train_onehot[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(train_batch, train_label_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= train_norm.shape[0] else BATCH_INDEX
 
    if step % 500 == 0:
        cost, accuracy = model.evaluate(batch_transpose(tune_data), tune_onehot, batch_size = tune_onehot.shape[0], verbose=False)
        print('tune cost: ', cost, 'tune accuracy: ', accuracy)


test_data = test(Trump_batch,Flute_batch,Bbc_batch)
preds = test_predict(test_data)
show_image(preds)



