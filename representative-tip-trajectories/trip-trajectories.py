#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:23:10 2025

@author: aman
"""
#%%
import matplotlib as mpl
from  matplotlib import pyplot as plt
import numpy as np
import os

#%% plottig x and y coordinate as function of time
fig,ax   = plt.subplots(figsize=(4,3))
fig1,ax1 = plt.subplots(figsize=(4,3))

color=["red","green","blue"]

for j,i in enumerate(range(10,13)):
    path = (os.path.join(f"../../variable-lp/rep-{i}/Density-100/run0010/","points.dat"))
    data = np.loadtxt(path,comments="%")

    totalFrames = 1201
    Nseg = int(data.shape[0]/totalFrames)

    x,y = data[::Nseg,1:3].T
    t = np.linspace(0,1200,1201)

    x = x[351:]
    y = y[351:]
    y -= y.mean()
    t = t[351:]
    x -=x.mean()

    ax.plot(t[:],x[:],ls="--",markersize=1.5,color=color[j],lw=1)
    ax1.plot(t[:],y[:],ls="-.",markersize=1.5,color=color[j],lw=1)

j = 2
ax.plot(t[:],x[:],ls="--",markersize=1.5,color=color[j],lw=1)
ax1.plot(t[:],y[:],ls="-.",markersize=1.5,color=color[j],lw=1)

ax.tick_params(labelsize=15)
ax1.tick_params(labelsize=15)

ax.set_xlim(351,1200)
ax.set_ylim(-2.5,2.5)
ax1.set_xlim(351,1200)
ax1.set_ylim(-2.5,2.5)

ax.set_xlabel("Time $(s)$",fontsize=15)
ax1.set_xlabel("Time $(s)$",fontsize=15)
ax.set_ylabel("X ($\mu$m)",fontsize=15)
ax1.set_ylabel("Y ($\mu$m)",fontsize=15)

fig.tight_layout()
fig1.tight_layout()

ax.legend(frameon=False)
ax1.legend(frameon=False)
fig.savefig("x-coor.pdf")
fig1.savefig("y-coor.pdf")
