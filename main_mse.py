#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from pathlib import Path
from pystackreg import StackReg
from czitools import extract_data
from joblib import Parallel, delayed
from skimage.metrics import mean_squared_error
from skimage.filters import threshold_triangle
from skimage.morphology import remove_small_objects

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
        self.mask = self.get_mask()
        self.mse = self.get_mse()
                
    def load_image(self):            
        data = extract_data(str(self.path), zoom=zoom)
        image = data[1].squeeze()
        return image       
    
    def get_mask(self):
        sr = StackReg(StackReg.TRANSLATION)
        reg = sr.register_transform_stack(self.image, reference='previous')
        mask = reg > threshold_triangle(reg) * threshCoeff
        mask = remove_small_objects(mask, min_size=minSize*zoom)
        return mask
    
    def get_mse(self):
        mse = []
        for t in range(1, self.image.shape[0]):
            mse.append(mean_squared_error(
                self.mask[t-1,...], self.mask[t,...]))
        mse = np.mean(mse)
        return mse
        
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
names, conds, mses = [], [], []
for data in image_data:
    names.append(data.name) 
    conds.append(data.cond) 
    mses.append(data.mse)
df = pd.DataFrame({
    'Name': names,
    'Condition': conds,
    'MSE': mses
})

# Boxplot
df.boxplot(column='MSE', by='Condition')
plt.suptitle('')  # This removes the default title pandas adds
plt.xlabel('Condition')
plt.ylabel('MSE')
plt.show()    