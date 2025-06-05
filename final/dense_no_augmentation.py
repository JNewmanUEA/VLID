import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
from tensorflow.keras import mixed_precision
from tensorflow.keras.initializers import Constant
from tensorflow.data import Dataset
import random
import os
from collections import Counter
import sys

repeat = int(sys.argv[1])
idx = int(sys.argv[2])
mixed_precision.set_global_policy('mixed_float16')

num_epochs = 50
num_batch_size = int(8)

def makemodel_dense():
    numClasses=3
    model = Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(1./0.5, offset=-1,  input_shape=(600, 90, 150, 1)))
    model.add(layers.Conv3D(8,(1,3,3),kernel_regularizer='l2',padding='same',activation=layers.LeakyReLU(alpha=0.01)))
    model.add(layers.Conv3D(8,(3,1,1),kernel_regularizer='l2',padding='same',activation=layers.LeakyReLU(alpha=0.01)))
    model.add(layers.MaxPooling3D((3,3,3)))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv3D(128,(1,3,3),padding='same',activation=layers.LeakyReLU(alpha=0.01)))
    model.add(layers.Conv3D(128,(3,1,1),padding='same',activation=layers.LeakyReLU(alpha=0.01)))
    model.add(layers.MaxPooling3D((3,3,3)))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv3D(32,(1,3,3),padding='same',activation=layers.LeakyReLU(alpha=0.01)))
    model.add(layers.Conv3D(32,(3,1,1),padding='same',activation=layers.LeakyReLU(alpha=0.01)))
    model.add(layers.MaxPooling3D((3,3,3)))
    model.add(layers.Dropout(0.1))

    model.add(layers.Flatten())
    model.add(layers.Dense(8,'relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(8,'relu'))
    model.add(layers.Dense(numClasses,'softmax', dtype='float32'))
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    model.summary()
    return model

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()

from sklearn.model_selection import GroupKFold,train_test_split,GroupShuffleSplit,LeavePGroupsOut
from sklearn import preprocessing
import datetime

x = datetime.datetime.now()
dateTimeStr = str(x)

nsplits = 5
results = np.zeros((nsplits))

def batch_generator(xIn, yIn, num_batch_size):
    while True:
        num_list = random.sample(range(0, yIn.shape[0]), num_batch_size)
        outputX = xIn[num_list,:,:,:,:]
        num_list2 = random.sample(range(0, int(num_batch_size)), num_batch_size)
        outputX[num_list2,:,:,:,:] = np.flip(outputX[num_list2,:,:,:,:],axis=3)
        yOut = yIn[num_list,:]
        yield outputX.astype(np.float16), yOut.astype(np.float16)

def batch_val_generator(xValIn, yValIn, num_batch_size):
    count = 0
    tot = yValIn.shape[0] // num_batch_size
    num_list = random.sample(range(0, yValIn.shape[0]), yValIn.shape[0])
    while True:
        count = count + 1
        stIdx = (count - 1) * num_batch_size
        edIdx = count * num_batch_size
        if count == tot:
            count = 0
        yOut = yValIn[num_list[stIdx:edIdx],:]
        yield [xValIn[num_list[stIdx:edIdx],:,:,:].astype(np.float16), yOut.astype(np.float16)]

def batch_test_generator(testXIn, testYIn, num_batch_size):
    for index3 in range(0,testYIn.shape[0],num_batch_size):
        yOut = testYIn[index3:index3+num_batch_size,:]
        yield testXIn[index3:index3+num_batch_size,:,:,:,:].astype(np.float16), yOut.astype(np.float16)

path=''
modeltype='paper_dense_1'
print(f'{modeltype} repeat {repeat}')
for i in range(0,5):
    if i != idx:
        continue

    mcp_save = ModelCheckpoint(f'train_{modeltype}_{repeat}_fold_{i}.h5', save_best_only=True, monitor='val_loss', mode='min')
    fitcomplete = False
    
    if os.path.isfile(f'train_{modeltype}_{repeat}_fold_{i}.h5'):
        fitcomplete = True
    
    xValIn = np.load(f'{path}{i}_valX_max.npy')
    yValIn = np.load(f'{path}{i}_valY_max.npy')
    
    yIn = np.load(f'{path}{i}_trainY_max.npy')
    xIn = np.load(f'{path}{i}_trainX_max.npy')

    print(xIn.shape)
    while not fitcomplete:
        model=makemodel_dense()
        history = model.fit(batch_generator(xIn, yIn, num_batch_size),validation_data=batch_val_generator(xValIn,yValIn,num_batch_size),
                    steps_per_epoch=yIn.shape[0]//num_batch_size,validation_steps=yValIn.shape[0]//num_batch_size,epochs=num_epochs, verbose=1, callbacks=[mcp_save])
        best_val = np.argmin(history.history['val_loss'])
        if history.history['val_accuracy'][best_val] > 0.80:
            fitcomplete = True
    del yIn, xIn, xValIn, yValIn
    model=makemodel_dense()
    model.load_weights(f'train_{modeltype}_{repeat}_fold_{i}.h5')
    testXIn = np.load(f'{path}{i}_testX_max.npy')
    testYIn = np.load(f'{path}{i}_testY_max.npy')
    test_accuracy=model.evaluate(batch_test_generator(testXIn,testYIn,num_batch_size),verbose=0,batch_size=num_batch_size)
    del testXIn, testYIn
    print(test_accuracy[1])
    results[i] = test_accuracy[1]
np.savetxt(f"results_{modeltype}_{repeat}_{idx}_{dateTimeStr}.csv", results, delimiter=",")
