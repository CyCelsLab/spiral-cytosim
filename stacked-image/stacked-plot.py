#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:07:31 2024

@author: aman
"""
#%% library import
import matplotlib as mpl
from  matplotlib import pyplot as plt
import numpy as np
import os

#%% color scheme
cmap = mpl.colormaps.get("turbo")

#%% data loading
path = (os.path.join("stacked-plot-data","lmt-15-density-100-vpl","points.dat"))
data = np.loadtxt(path,comments="%")


totalFrames = 1201
Nseg = int(data.shape[0]/totalFrames)

fig,ax = plt.subplots(figsize=(5,5))

frameSkip = 25       # frame difference between plotting 
numberOfFrame  = 8   # no of frames to be plotted 
verticalShift = 4    # shift in stacking 

for j,i in enumerate(range(0,numberOfFrame*Nseg*frameSkip +1,frameSkip*Nseg)):
    X = data[i:i+Nseg,1]
    Y = data[i:i+Nseg,2]
    plt.plot(X,Y+verticalShift*j,color=cmap(i/Nseg/(frameSkip*numberOfFrame)))
    print(j,i/Nseg/(frameSkip*numberOfFrame))
    print(i//Nseg)

cm = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,i//Nseg)),ax=ax)
cm.ax.set_ylabel(r"Time (s)",fontsize=15)
cm.ax.tick_params(labelsize=15)

plt.tick_params(labelsize=15)
plt.xlabel("X $(\mu m)$",fontsize=15)

plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join("stacked-plot.pdf"))
