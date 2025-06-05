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
from scipy import stats

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

def batch_test_generator(testXIn, testYIn, num_batch_size):
    for index3 in range(0,testYIn.shape[0],num_batch_size):
        yOut = testYIn[index3:index3+num_batch_size,:]
        yield testXIn[index3:index3+num_batch_size,:,:,:,:].astype(np.float16), yOut.astype(np.float16)

exp='paper_dense_1'
path=''
for i in range(0,5):
    model1=makemodel_dense()
    model2=makemodel_dense()
    model3=makemodel_dense()
    model4=makemodel_dense()
    model5=makemodel_dense()

    model1.load_weights(f'train_{exp}_{1}_fold_{i}.h5')
    model2.load_weights(f'train_{exp}_{2}_fold_{i}.h5')
    model3.load_weights(f'train_{exp}_{3}_fold_{i}.h5')
    model4.load_weights(f'train_{exp}_{4}_fold_{i}.h5')
    model5.load_weights(f'train_{exp}_{5}_fold_{i}.h5')
 
    testXIn = np.load(f'{path}{i}_testX_max.npy')
    testYIn = np.load(f'{path}{i}_testY_max.npy')
    pred1 = model1.predict(batch_test_generator(testXIn,testYIn,num_batch_size))
    pred2 = model2.predict(batch_test_generator(testXIn,testYIn,num_batch_size))
    pred3 = model3.predict(batch_test_generator(testXIn,testYIn,num_batch_size))
    pred4 = model4.predict(batch_test_generator(testXIn,testYIn,num_batch_size))
    pred5 = model5.predict(batch_test_generator(testXIn,testYIn,num_batch_size))
    
    allpreds = pred1 + pred2 + pred3 + pred4 + pred5
    np.savetxt(f"results_{exp}_{dateTimeStr}_comb_fold_{i}.csv", allpreds, delimiter=",")
 
    predidx1 = np.argmax(pred1, axis = 1)
    predidx2 = np.argmax(pred2, axis = 1)
    predidx3 = np.argmax(pred3, axis = 1)
    predidx4 = np.argmax(pred4, axis = 1)
    predidx5 = np.argmax(pred5, axis = 1)

    print(predidx1.shape)
    allpreds = np.stack((predidx1,predidx2,predidx3,predidx4,predidx5),axis=1)
    print(allpreds.shape)
    predidx = stats.mode(allpreds,axis=1)[0]
    print(predidx.shape)

    testYIn = np.hstack((testYIn,np.zeros((testYIn.shape[0], 1))))
    testYIn = np.argmax(testYIn, axis = 1)
    accuracy = metrics.accuracy_score(testYIn, predidx)
    print ("Accuracy = ", accuracy)
    confusion_matrix = metrics.confusion_matrix(testYIn, predidx)
    print(confusion_matrix)
    
    del testXIn, testYIn
    results[i] = accuracy
np.savetxt(f"results_{exp}_{dateTimeStr}.csv", results, delimiter=",")
