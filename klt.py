#%% Imports -------------------------------------------------------------------

import cv2
import numpy as np
from pystackreg import StackReg
from skimage.draw import line
from skimage.morphology import dilation, square

#%% Functions -----------------------------------------------------------------

def klt(stack, feat_params, flow_params):
    
    klt_data = {
        'xCoords': [],
        'yCoords': [],
        'dXY': [],
        'status': [],
        'errors': [],
        }

    # Get frame & features (t0)
    frm0 = stack[0,:,:]
    f0 = cv2.goodFeaturesToTrack(
        frm0, mask=None, **feat_params
        )

    for t in range(1, stack.shape[0]):
        
        # Get current image
        frm1 = stack[t,:,:]
        
        # Compute optical flow (between f0 and f1)
        f1, status, errors = cv2.calcOpticalFlowPyrLK(
            frm0, frm1, f0, None, **flow_params
            )
        
        # Format outputs
        errors = errors.squeeze().astype(float);
        status = status.squeeze().astype(float); 
        f0 = f0.squeeze(); f1 = f1.squeeze()
        f0[f0[:,0] >= frm0.shape[1]] = np.nan
        f0[f0[:,1] >= frm0.shape[0]] = np.nan
        f1[f1[:,0] >= frm1.shape[1]] = np.nan
        f1[f1[:,1] >= frm1.shape[0]] = np.nan
        f1[status == 0] = np.nan
        
        # Measure distances
        dXY = np.linalg.norm(f1 - f0, axis=1) 
            
        # Append klt_data
        if t == 1:
            nan = np.full_like(status, np.nan)
            klt_data['xCoords'].append(f0[:,0])
            klt_data['yCoords'].append(f0[:,1])
            klt_data['status'].append(nan)
            klt_data['errors'].append(nan)
            klt_data['dXY'].append(nan)
        klt_data['xCoords'].append(f1[:,0])
        klt_data['yCoords'].append(f1[:,1])
        klt_data['status'].append(status)
        klt_data['errors'].append(errors)
        klt_data['dXY'].append(dXY)
            
        # Update previous frame & features 
        frm0 = frm1
        f0 = f1.reshape(-1,1,2)
        
    return klt_data

# -----------------------------------------------------------------------------

def klt_display(stack, klt_data):
    
    # Create empty diplay arrays
    ftsRaw = np.zeros_like(stack, dtype=bool)
    tksRaw = np.zeros_like(stack, dtype=bool)
    ftsLab = np.zeros_like(stack, dtype='uint16')
    ftsdXY = np.zeros_like(stack, dtype=float)
    ftsErr = np.zeros_like(stack, dtype=float)

    for t in range(stack.shape[0]):

        # Extract variables   
        x1s = klt_data['xCoords'][t]
        y1s = klt_data['yCoords'][t]
        dXY = klt_data['dXY'][t]
        errors = klt_data['errors'][t]
        labels = np.arange(x1s.shape[0]) + 1
        
        # Remove non valid data
        valid_idx = ~np.isnan(x1s)
        x1s = x1s[valid_idx].astype(int)
        y1s = y1s[valid_idx].astype(int)
        dXY = dXY[valid_idx]
        errors = errors[valid_idx]
        labels = labels[valid_idx]
        
        # Fill features display arrays
        ftsRaw[t, y1s, x1s] = True
        ftsLab[t, y1s, x1s] = labels
        ftsdXY[t, y1s, x1s] = dXY
        ftsErr[t, y1s, x1s] = errors
        
        # Fill tracks display arrays
        if t > 0:
            x0s = klt_data['xCoords'][t-1]
            y0s = klt_data['yCoords'][t-1]
            x0s = x0s[valid_idx].astype(int)
            y0s = y0s[valid_idx].astype(int)
            for x0, y0, x1, y1 in zip(x0s, y0s, x1s, y1s):
                rr, cc = line(y0, x0, y1, x1)
                tksRaw[t,rr,cc] = True

        # Dilate display arrays
        dilateSize = 3
        ftsRaw[t,...] = dilation(ftsRaw[t,...], footprint=square(dilateSize))
        ftsLab[t,...] = dilation(ftsLab[t,...], footprint=square(dilateSize))
        ftsdXY[t,...] = dilation(ftsdXY[t,...], footprint=square(dilateSize))
        ftsErr[t,...] = dilation(ftsErr[t,...], footprint=square(dilateSize))
        
    return ftsRaw, tksRaw, ftsLab, ftsdXY, ftsErr