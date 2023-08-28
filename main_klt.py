#%% Imports -------------------------------------------------------------------

import cv2
import time
import numpy as np
from pathlib import Path
from pystackreg import StackReg
from czitools import extract_data
from joblib import Parallel, delayed

from klt import klt, klt_display

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

# Feature detection
feat_params = dict(
    maxCorners=1000,
    qualityLevel=0.001,
    minDistance=5,
    blockSize=5,
	useHarrisDetector=True
    )

# Optical flow
flow_params = dict(
    winSize=(11, 11),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.01)
    )

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

#%% Format results ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Results DataFrame
names, conds, dXY = [], [], []
for data in image_data:
    names.append(data.name) 
    conds.append(data.cond) 
    stack = data.image
    stack = (stack / 256).astype('uint8')
    klt_data = klt(stack, feat_params, flow_params)
    dXY.append(np.nanmean(klt_data['dXY']))

df = pd.DataFrame({
    'Name': names,
    'Condition': conds,
    'dXY': dXY
})

# Boxplot
df.boxplot(column='dXY', by='Condition')
plt.suptitle('')  # This removes the default title pandas adds
plt.xlabel('Condition')
plt.ylabel('dXY')
plt.show()   

#%% Display KLT ---------------------------------------------------------------

idx = 7
stack = image_data[idx].image
stack = (stack / 256).astype('uint8')
klt_data = klt(stack, feat_params, flow_params)
diplays = klt_display(stack, klt_data)

ftsRaw = diplays[0]
tksRaw = diplays[1]
ftsLab = diplays[2]

import napari
viewer = napari.Viewer()
viewer.add_image(stack)
# viewer.add_image(ftsRaw, blending='additive')
viewer.add_image(tksRaw, blending='additive')
viewer.add_labels(ftsLab, blending='additive')