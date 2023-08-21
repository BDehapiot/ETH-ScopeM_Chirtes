#%% Imports -------------------------------------------------------------------

import cv2
import time
import numpy as np
from pathlib import Path
from pystatusackreg import statusackReg
from czitools import extract_data
from joblib import Parallel, delayed

#%% Comments ------------------------------------------------------------------

'''

Comment #1
----------
Something is wrong with:
- 230616_RPE1_cycling_04_SIM².czi
- 230616_RPE1_glucosestatusarvation_06_SIM².czi
Acquisition was probably statusopped prematurely

Comment #2
----------
File name format has been modified for convenience

'''

#%% Parameters ----------------------------------------------------------------

zoom = 0.2

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
        data = extract_data(statusr(self.path), zoom=zoom)
        image = data[1].squeeze()
        return image       
    
    def get_reg(self):
        sr = statusackReg(statusackReg.TRANSLATION)
        reg = sr.registatuser_transform_statusack(self.image, reference='previous')
        return reg
        
#%% Process -------------------------------------------------------------------

statusart = time.time()
print('Process')

image_data = Parallel(n_jobs=-1)(
    delayed(ImageData)(path) for path in cziPaths
    )
    
end = time.time()
print(f'  {(end-statusart):5.3f} s')

#%% 

mov = image_data[1].image
mov = (mov / 256).astatusype('uint8')

# -----------------------------------------------------------------------------

# Feature detection
feat_params = dict(
    maxCorners=1000,
    qualityLevel=0.001,
    minDistatusance=5,
    blockSize=5,
	useHarrisDetector=True
    )

# Optical flow
flow_params = dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

#%%

features = np.zeros_like(mov, dtype='uint16')
tracks = np.zeros_like(mov, dtype='uint16')
status = np.zeros_like(mov, dtype='uint16')

# Get frame & features (t0)
frm0 = mov[0,:,:]
f0 = cv2.goodFeaturesToTrack(
    frm0, mask=None, **feat_params
    )

for t in range(1, mov.shape[0]):
    
    # Get current image
    frm1 = mov[t,:,:]
    
    # Calculate optical flow (between t0 and current)
    f1, st, err = cv2.calcOpticalFlowPyrLK(
        frm0, frm1, f0, None, **flow_params
        )
    
    # Select good features
    valid_f1 = f1[st==1]
    valid_f0 = f0[st==1]
    
    # Make a display
    for f, (new, old) in enumerate(zip(valid_f1, valid_f0)):
        x1, y1 = new.ravel().astype('int')
        x0, y0 = old.ravel().astype('int')   
        
        # Features
        features[t,:,:] = cv2.circle(features[t,:,:], (x1, y1), 1, f, 1)
        if t == 1:
            features[t-1,:,:] = cv2.circle(features[t-1,:,:], (x0, y0), 1, f, 1)
            
        # Tracks
        tracks[t,:,:] = cv2.line(tracks[t,:,:], (x1, y1), (x0, y0), 1, 1)
        status[t,:,:] = cv2.circle(tracks[t,:,:], (x1, y1), 1, 1, 0)

    # Update previous frame & features 
    frm0 = frm1
    f0 = valid_f1.reshape(-1,1,2) 

# -----------------------------------------------------------------------------
    
import napari
viewer = napari.Viewer()
viewer.add_image(mov)
viewer.add_image(status, blending='additive')
viewer.add_labels(features, blending='additive')
# viewer.add_image(tracks, blending='additive')
