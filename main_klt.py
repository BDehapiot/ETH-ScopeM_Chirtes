#%% Imports -------------------------------------------------------------------

import cv2
import time
import numpy as np
from pathlib import Path
from pystackreg import StackReg
from czitools import extract_data
from joblib import Parallel, delayed

#%% Comments ------------------------------------------------------------------

'''

Comment #1
----------
Something is wrong with:
- 230616_RPE1_cycling_04_SIM².czi
- 230616_RPE1_glucosestarvation_06_SIM².czi
Acquisition was probably stopped prematurely

Comment #2
----------
File name format has been modified for convenience

'''

#%% Parameters ----------------------------------------------------------------

zoom = 0.2
threshCoeff = 2
minSize = 256

#%% Initialize ----------------------------------------------------------------

dataPath = Path('D:/local_Chirtes/data')

# Get paths
cziPaths = []
for path in dataPath.iterdir():
    if path.suffix == '.czi':
        cziPaths.append(path)

#%% Classes & functions -------------------------------------------------------

class ImageData:
    def __init__(self, path):
        self.path = path
        self.name = path.name
        self.cond = path.name[12:17]
        self.image = self.load_image()
        self.reg = self.get_reg()
                
    def load_image(self):            
        data = extract_data(str(self.path), zoom=zoom)
        image = data[1].squeeze()
        return image       
    
    def get_reg(self):
        sr = StackReg(StackReg.TRANSLATION)
        reg = sr.register_transform_stack(self.image, reference='previous')
        return reg
        
#%% Process -------------------------------------------------------------------

start = time.time()
print('Process')

image_data = Parallel(n_jobs=-1)(
    delayed(ImageData)(path) for path in cziPaths
    )
    
end = time.time()
print(f'  {(end-start):5.3f} s')

#%% 

mov = image_data[1].image
mov = (mov / 256).astype('uint8')

# -----------------------------------------------------------------------------

# Feature detection
feat_params = dict(
    maxCorners=1000,
    qualityLevel=0.0001,
    minDistance=7,
    blockSize=7,
	useHarrisDetector=True
    )

# Optical flow
flow_params = dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

# -----------------------------------------------------------------------------

klt_data = []
features = np.zeros_like(mov)
tracks = np.zeros_like(mov)

# Get frame & features (t0)
frm0 = mov[0,:,:]
f0 = cv2.goodFeaturesToTrack(
    frm0, mask=None, **feat_params)

for i in range(1, mov.shape[0]):
    
    # Get current image
    img1 = mov[i,:,:]
    
    # Calculate optical flow (between t0 and current)
    f1, st, err = cv2.calcOpticalFlowPyrLK(
        frm0, img1, f0, None, **flow_params)
    
    # Select good features
    valid_f1 = f1[st==1]
    valid_f0 = f0[st==1]
    
    # Make a display
    for j,(new,old) in enumerate(zip(valid_f1,valid_f0)):
        
        a, b = new.ravel().astype('int')
        c, d = old.ravel().astype('int')
        
        tracks[i,:,:] = cv2.line(tracks[i,:,:], (a,b), (c,d), 255, 2)
        tracks[i,:,:] = tracks[i,:,:] + tracks[i-1,:,:] // 1.5
        features[i,:,:] = cv2.circle(features[i,:,:], (a,b), 1, 255, 1)
        # tracks[i,:,:] = cv2.line(tracks[i,:,:], (a,b), (c,d), (255,255,255), 1)
        # features[i,:,:] = cv2.circle(features[i,:,:], (a,b), 1, (255,255,255), 1)
        
    # Update previous image & features 
    frm0 = img1
    f0 = valid_f1.reshape(-1,1,2) 

    klt_data.append((f1.shape[0]))
    
tracks = tracks
features = features

import napari
viewer = napari.Viewer()
viewer.add_image(mov)
viewer.add_image(features, blending='additive')
viewer.add_image(tracks, blending='additive')
