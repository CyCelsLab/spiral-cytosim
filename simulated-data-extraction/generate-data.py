#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:41:57 2024

@author: aman
"""
#%% importing libraries
import numpy as np
import pandas as pd
from scipy.fft import rfft,rfftfreq
import matplotlib.pyplot as plt
import os
#%% fixed variables
totalFrames  = 1201 # total frames; included zero as well.
samplingRate = 1    # sampling rate of output.
exclude = 351       # frames to be excluded.
#%% test case
def getData(filePath:str,totalFrames:int,exclude:int , samplingRate:float):

    # reading data
    data = np.loadtxt(os.path.join(filePath,"points.dat"),comments="%")

    Nseg = len(data) // totalFrames
    distmat = np.zeros((totalFrames,Nseg))

    # pairwise distance among countour segments
    for i in range(0,len(data),Nseg):
        distmat[int(i/Nseg)] = (np.linalg.norm(data[i:i+Nseg,1:3][:] - data[i:i+Nseg,1:3][-1],axis=1))
        distmat[int(i/Nseg)] = distmat[int(i/Nseg)][::-1]

    # mid point of contour
    ll = Nseg//(2)

    # end point
    lr = Nseg

    yield distmat[exclude:,ll:lr].mean()

    ## fft estimation
    xfree ,yfree  = data[::Nseg,1:3].T
    
    x = xfree[exclude:]
    y = yfree[exclude:]
    dx = x[1:]-x[:-1]
    dy = y[1:]-y[:-1]
    
    velocity = np.sqrt(dx**2+dy**2).mean()
    

    ## xtip fft
    signalx = xfree[exclude:]-xfree[exclude:].mean()
    #plt.plot(signalx)
    fftTipx = rfft(signalx)
    freqx   = rfftfreq(len(signalx),1.0)
    # ytip fft
    signaly = yfree[exclude:]-yfree[exclude:].mean()
    #plt.plot(signaly)
    fftTipy = rfft(signaly)
    freqy   = rfftfreq(len(signaly),1.0)

    # maximum frequency x and y
    xfreqmax = freqx[np.argmax(np.abs(fftTipx)*2/len(signalx))]
    yfreqmax = freqy[np.argmax(np.abs(fftTipy)*2/len(signaly))]

    yield xfreqmax
    yield yfreqmax
    yield velocity


for i in getData("constant-lp/rep-3/Density-150/run0020",1201,exclude,1): print(i)

# manual estimation
filePath = "constant-lp/rep-3/Density-150/run0020"

data = np.loadtxt(os.path.join(filePath,"points.dat"),comments="%")

Nseg = len(data) // totalFrames
distmat = np.zeros((totalFrames,Nseg))

for i in range(0,len(data),Nseg):
    distmat[int(i/Nseg)] = (np.linalg.norm(data[i:i+Nseg,1:3][:] - data[i:i+Nseg,1:3][-1],axis=1))
    distmat[int(i/Nseg)] = distmat[int(i/Nseg)][::-1]


ll = Nseg//(2)

lr = Nseg

print(distmat[exclude:,ll:lr].mean())

## fft estimation
xfree ,yfree  = data[::Nseg,1:3].T

## xtip fft
signalx = xfree[exclude:]-xfree[exclude:].mean()
#plt.plot(signalx)
fftTipx = rfft(signalx)
freqx   = rfftfreq(len(signalx),samplingRate)
# ytip fft
signaly = yfree[exclude:]-yfree[exclude:].mean()
#plt.plot(signaly)
fftTipy = rfft(signaly)
freqy   = rfftfreq(len(signaly),samplingRate)

# maximum frequency x and y
xfreqmax = freqx[np.argmax(np.abs(fftTipx)*2/len(signalx))]
yfreqmax = freqy[np.argmax(np.abs(fftTipy)*2/len(signaly))]

#plt.figure()
#plt.plot(np.abs(fftTipx)*2/len(signalx))
plt.show()
#plt.plot(yfree)
#plt.show()
# %% Analysis for constant persistence length
columns = ["rep","length","density","radius","x-freq","y-freq","velocity"]
df = pd.DataFrame(columns=columns)

temp = {}
for i in columns: temp.update({i:0})
for rep in range(2,22):
    for density in np.arange(50,201,25)[:]:
        for length,folder in enumerate(np.arange(26),5):
            temp["length"] = length
            temp["density"] = density
            temp["rep"] = rep
            filePath = os.path.join("constant-lp","rep-%d"%rep,"Density-%d"%density,"run%04d"%folder)
            print(filePath)
            for j,i in enumerate(getData(filePath, totalFrames, exclude, samplingRate),3):
                temp[columns[j]] = i
            df.loc[len(df)] = temp
df.to_csv("radius-freq-cpl.csv")

# %% Performing analysis on variable persistence length simulations
columns = ["rep","length","density","radius","x-freq","y-freq","velocity"]
df = pd.DataFrame(columns=columns)

temp = {}
for i in columns: temp.update({i:0})
for rep in range(2,22):
    for density in np.arange(50,201,25)[:]:
        for length,folder in enumerate(np.arange(26),5):
            temp["length"] = length
            temp["density"] = density
            temp["rep"] = rep
            filePath = os.path.join("variable-lp","rep-%d"%rep,"Density-%d"%density,"run%04d"%folder)
            print(filePath)
            for j,i in enumerate(getData(filePath, totalFrames, exclude, samplingRate),3):
                temp[columns[j]] = i
            df.loc[len(df)] = temp
df.to_csv("radius-freq-vpl.csv")
