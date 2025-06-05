import numpy as np
transformedData= np.load('/gpfs/home/w0454974/scratch/processed_data_resized.npy').astype(np.float16)
print(transformedData.shape)
transformedDataLanguageLabel = np.load('/gpfs/home/w0454974/scratch/processed_subjectLabels_resized.npy')
transformedDataGroupLabel = np.load('/gpfs/home/w0454974/scratch/processed_langLabels_resized.npy')

uniqueSubjects = np.unique(transformedDataGroupLabel)
print(uniqueSubjects)

transformedData = np.expand_dims(transformedData, axis=-1)
print(transformedData.shape)
print(transformedDataGroupLabel.shape)
print(transformedDataLanguageLabel.shape)

print(np.unique(transformedDataGroupLabel))
print(np.unique(transformedDataLanguageLabel))

for i in range(0,transformedData.shape[0]):
    if i % 500 == 0:
        print(i)
    maxval = transformedData[i,:].max()
    transformedData[i,:] = transformedData[i,:] / maxval
transformedData = np.nan_to_num(transformedData)

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
from collections import Counter
from skimage.transform import resize

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()

from sklearn.model_selection import GroupKFold,train_test_split,GroupShuffleSplit,LeavePGroupsOut
from sklearn import preprocessing
import datetime

x = datetime.datetime.now()
dateTimeStr = str(x)

nsplits = 5
group_kfold = GroupKFold(n_splits=nsplits)
group_kfold.get_n_splits(transformedData, transformedDataLanguageLabel, transformedDataGroupLabel)
print(np.unique(transformedDataLanguageLabel))
new_labels=to_categorical(labelencoder.fit_transform(transformedDataLanguageLabel))
results = np.zeros((nsplits))
for i, (train_index, test_index) in enumerate(group_kfold.split(transformedData, transformedDataLanguageLabel, transformedDataGroupLabel)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={transformedDataGroupLabel[train_index]}")
    print(f"  Test: group={transformedDataGroupLabel[test_index]}")
    print(np.unique(transformedDataGroupLabel[test_index]))
    yIn = new_labels[train_index,:]
    xIn = transformedData[train_index,:,:,:,:]
    gIn = transformedDataGroupLabel[train_index]
    
    lpgo = GroupShuffleSplit(test_size=3,n_splits=1)
    new_train_index, new_val_index = next(lpgo.split(xIn, yIn, gIn))
    print(f"  Train: group={gIn[new_train_index]}")
    print(np.unique(gIn[new_train_index]))
    print(f"  Val: group={gIn[new_train_index]}")
    print(np.unique(gIn[new_val_index]))
    xValIn = xIn[new_val_index,:,:,:,:]
    yValIn = yIn[new_val_index,:]
    iValIn = gIn[new_val_index]
    
    newxIn = xIn[new_train_index,:,:,:,:]
    newyIn = yIn[new_train_index,:]
    newiIn = gIn[new_train_index]
    
    
    testYIn = new_labels[test_index,:]
    testiIn = transformedDataGroupLabel[test_index]
    testXIn = transformedData[test_index,:,:,:,:]
    np.save(f'{i}_testi_max',testiIn)
    np.save(f'{i}_traini_max',newiIn)
    np.save(f'{i}_vali_max',iValIn)
    np.save(f'{i}_testY_max',testYIn)
    np.save(f'{i}_testX_max',testXIn)
    np.save(f'{i}_trainY_max',newyIn)
    np.save(f'{i}_trainX_max',newxIn)
    np.save(f'{i}_valY_max',yValIn)
    np.save(f'{i}_valX_max',xValIn)

