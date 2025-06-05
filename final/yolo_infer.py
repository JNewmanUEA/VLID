from scipy.fft import dctn
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import sys

idx = int(sys.argv[1])

def extractFeats(videoPath):
    features = []
    cap = cv2.VideoCapture(videoPath)
    frameCount = -1
    stemFilename = (Path(os.path.basename(videoPath)).stem)
    model = YOLO('tracking.pt')
    previousdat = []
    allresults = []
    numOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while frameCount < numOfFrames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameCount)
        res, frame = cap.read()
        frameCount = frameCount + 1

        if res == False:
            print(f'problem at {frameCount}')
            print(result.shape)
            if result.shape[0] == 0:
                result = np.zeros((1,4))
            allresults.append(result)
            continue
        
        results = model(frame,max_det=1,boxes=True,verbose=False, device=0, half=True)
        
        result = results[0].boxes.xywh.data.cpu().numpy()
        if result.shape[0] == 0:
            result = np.zeros((1,4))
        allresults.append(result)
        
        if frameCount % 500 == 0:
            print(frameCount)
    np.save(stemFilename+"_track", allresults)

import os
from pathlib import Path
import glob

videoPath = '/gpfs/home/videos/'
for filename in sorted(glob.glob(os.path.join(videoPath, '*.MP4'))):
    print(filename)
    extractFeats(filename)

