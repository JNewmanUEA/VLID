import os
from pathlib import Path
import scipy.interpolate as interp
from sklearn.preprocessing import normalize
import IPython
from IPython import display
import glob
import numpy as np
from skimage.transform import resize


def window(data):
    windowLength=600
    numSamples=len(data)
    numOfWindows=np.floor(numSamples / windowLength)
    data=data[0:int(numOfWindows*windowLength),:,:]
    data=data.reshape(( int(numOfWindows), windowLength,  data.shape[1], data.shape[2]))
    return data

transformedData = np.ubyte([])
transformedDataGroupLabel = []
transformedDataLanguageLabel = []

ids = []
subjects = []
npyPath = '/gpfs/scratch/tracking/'
for filename in sorted(glob.glob(os.path.join(npyPath, 'S*_mouth.npy'))):
    langID = -1
    if "Ara" in filename:
        langID = 0
    if "Eng" in filename:
        langID = 1
    if "Man" in filename:
        langID = 2
    if langID == -1:
        continue
    print(langID)
    
    actual_filename = os.path.basename(filename)
    id = int(''.join(filter(str.isdigit, actual_filename[0:3])))
    print(int(''.join(filter(str.isdigit, actual_filename[0:3]))))
    if id not in ids:
        ids.append(id)
    
    print(filename)
    
    data = np.load(filename,allow_pickle=True)
    
    newdat = np.zeros((data.shape[0],90,150,1))
    for i in range(0, data.shape[0]):
        newdat[i,:,:,:] = np.ubyte(resize(data[i,:,:], (90,150,1),preserve_range=True))
    
    tdd = window(newdat)
    
    print(tdd.shape)
    
    if len(transformedData) == 0:
        transformedData = tdd
        transformedDataGroupLabel = np.full((len(tdd)), id)
        transformedDataLanguageLabel = np.full((len(tdd)), langID)
    else:
        transformedData = np.vstack((transformedData, tdd))
        transformedDataGroupLabel = np.hstack((transformedDataGroupLabel, np.full((len(tdd)), id)))
        transformedDataLanguageLabel = np.hstack((transformedDataLanguageLabel, np.full((len(tdd)), langID)))   
        
np.save('all_ids.npy',ids)
np.save('processed_data_resized',transformedData)
np.save('processed_langLabels_resized',transformedDataGroupLabel)
np.save('processed_subjectLabels_resized',transformedDataLanguageLabel)
