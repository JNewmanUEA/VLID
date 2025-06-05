from scipy.fft import dctn
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import sys

idx = int(sys.argv[1])

from scipy.fft import dctn
import cv2
from PIL import Image
import numpy as np

def extractFeats(videoPath):
    features = []
    cap = cv2.VideoCapture(videoPath)
    mouthW = 250
    mouthH = 150
    frameCount = -1
    stemFilename = (Path(os.path.basename(videoPath)).stem)
    allresults = np.load(stemFilename+"_track.npy",allow_pickle=True)
    
    numOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while frameCount < numOfFrames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameCount)
        res, frame = cap.read()
        frameCount = frameCount + 1
        if res == False:
            print('oh no!')
            break

        result = allresults[frameCount]
        if result.shape[0] > 0:
            if not np.any(result):
                if previousdat == []:
                    dat = [900, 500, 500, 700]
                else:
                    dat = previousdat
            else:
                dat = np.int32(result[0].round())
        else:
            if previousdat == []:
                dat = [900, 500, 500, 700]
            else:
                dat = previousdat

        pt1 = np.int32(np.round(dat[2]/2))
        pt2 = np.int32(np.round(dat[3]/2))
        x1 = max(0, dat[1]-pt2)
        x2 = min(frame.shape[0], dat[1]+pt2)
        x3 = max(0, dat[0]-pt1)
        x4 = min(frame.shape[1], dat[0]+pt1)
        frame = frame[x1:x2,x3:x4,:]

        ratio = frame.shape[0] / frame.shape[1]

        newy = int(np.round(mouthW * ratio))
        newx = mouthW
        widthpad = 0
        heightpad = (mouthH - newy) // 2

        if newy > mouthH:
            newx = int(np.round(mouthH * 1/ratio))
            newy = mouthH
            widthpad = (mouthW - newx) // 2
            heightpad = 0

        widthpad2 = widthpad
        heightpad2 = heightpad

        if 2 * widthpad + newx < mouthW:
            widthpad2 = widthpad2 + (mouthW - 2 * widthpad - newx)

        if 2 * heightpad + newy < mouthH:
            heightpad2 = heightpad2 + (mouthH - 2 * heightpad - newy)

        img = cv2.resize(frame,(newx, newy))

        color = [0, 0, 0]
        frame = cv2.copyMakeBorder(img, heightpad, heightpad2, widthpad, widthpad2, cv2.BORDER_CONSTANT,
            value=color)
        
        if not frame.shape[0] == 150 or  not frame.shape[1] == 250:
            print('ouch')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frameCount % 500 == 0:
            print(frameCount)
        features.append(frame)
        previousdat = dat
    np.save(stemFilename+"_mouth", features)
    return features

import os
from pathlib import Path
import glob

videoPath = '/gpfs/home/videos/'
for filename in sorted(glob.glob(os.path.join(videoPath, '*.MP4'))):
    print(filename)
    extractDCTFeats(filename)

