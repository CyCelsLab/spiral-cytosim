#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:20:05 2024

@author: aman
"""
#%% importing libraries
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
#%% scaling varibles0
width      = 0.025     # micron
dutyRatio  = 1         # dimension less
stallForce = 5         # pico-Newton
Kbt        = 4.3*1e-3  # pN micron
motorVelocity = 0.1    # micron/s
#%% figure folder
figureFolder = "radius-density"
#%%
def compute_standard_regression(y_true,y_pred,n,p):
    ss_res = np.sum((y_true - y_pred)**2)
    return(np.sqrt(ss_res/(n-p-1)))
#%% taken from pampaloni et al.
lpinf = 6300
pl = lambda l: lpinf/(1+441/l**2)
#%% color map
cmap = matplotlib.colormaps.get("turbo")
#%% Analysis of varible lp data
#%%%% reading data
dfvl = pd.read_csv("radius-freq-vpl.csv",index_col=0)
dfvl["lfd"] = dfvl["density"]*(width*stallForce*dutyRatio)
dfvl["pl"] = pl(dfvl.length)
dfvl_M = dfvl.groupby(["length","density"]).mean().reset_index()
dfvl_S = dfvl.groupby(["length","density"]).std().reset_index()
#%%%% plotting raw data
fig,ax = plt.subplots(figsize=(4,3))
minl =  dfvl_M.length.min()
maxl =  dfvl_M.length.max()
diff = maxl-minl
for length in dfvl_M.length.unique():
    tempdfm = dfvl_M[dfvl_M.length == length]
    tempdfs = dfvl_S[dfvl_S.length == length]
    plt.errorbar(tempdfm.density,tempdfm.radius,yerr=tempdfs.radius,color=cmap((length-minl)/diff),lw=1)
plt.xlabel(r"$\rho_{m}$ (Motors/$\mu$m$^{2})$",fontsize=15)
plt.ylabel(r"R ($\mu$m)",fontsize=15)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$L$ ($\mu$m)",fontsize=15)
cm.ax.set_yticks(ticks=[7,14,21,28])
cm.ax.tick_params(labelsize=15)

plt.tick_params(labelsize=15)
plt.tight_layout()

plt.savefig(f"{figureFolder}/raw-vl-radius-density.pdf")
#%%%% rescaling data
dfvl["lfd"] =  dfvl.lfd
dfvl["A"] = dfvl.radius/(dfvl.pl*Kbt)**(1/3)
#%%%% plotting rescaled data
tempdf = dfvl
tempdf = tempdf.groupby(["length","density"]).mean().reset_index()

fig,ax = plt.subplots(figsize=(4,3))
plt.scatter(tempdf.lfd,tempdf.A,c=tempdf.length,cmap="turbo")

# power law fitting
x  = tempdf.lfd
y  = tempdf.A
fit = lambda x,a,b: a*x**b
(a,b),(_) = curve_fit(fit,x,y)
SER = compute_standard_regression(y,a*x**b,len(y),2)
x = np.sort(x.unique())
plt.plot(x,a*x**b,ls="--",color="k",lw=1)
plt.text(10,8e-1,r"$f^{%s}$ [%s]"%(round(b,3),round(SER,2)),fontsize=10)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$L$  ($\mu$m)",fontsize=15)
cm.ax.set_yticks(ticks=[7,14,21,28])
cm.ax.tick_params(labelsize=15)

plt.ylabel(r"$A_v$ (pN/$\mu$m)$^{-1/3}$",fontsize=15)
plt.xlabel(r"$f$ (pN/$\mu$m)",fontsize=15)
plt.tick_params(labelsize=15)
plt.xscale("log")
plt.yscale("log")
plt.ylim(0.2,2)
plt.tight_layout()
#plt.show()
plt.savefig(f"{figureFolder}/scaled-vl-A-epsilon.pdf")
#%% Analysis of constant lp data
dfcl = pd.read_csv("radius-freq-cpl.csv",index_col=0)
dfcl["lfd"] = dfcl["density"]*(width*stallForce*dutyRatio)
dfcl["pl"]  = 5000  #gittes et al.
dfcl_M = dfcl.groupby(["length","density"]).mean().reset_index()
dfcl_S = dfcl.groupby(["length","density"]).std().reset_index()
#%%%% plotting raw data
minl =  dfcl_M.length.min()
maxl =  dfcl_M.length.max()
diff = maxl-minl

fig,ax = plt.subplots(figsize=(4,3))
for length in dfcl_M.length.unique():
    tempdfm = dfcl_M[dfcl_M.length == length]
    tempdfs = dfcl_S[dfcl_S.length == length]
    plt.errorbar(tempdfm.density,tempdfm.radius,tempdfs.radius,color=cmap((length-minl)/diff),lw=1)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$L$ ($\mu$m)",fontsize=15)
cm.ax.set_yticks(ticks=[7,14,21,28])
cm.ax.tick_params(labelsize=15)

plt.xlabel(r"$\rho_{m}$ $(Motors/\mu m^{2})$",fontsize=15)
plt.ylabel(r"R ($\mu$m)",fontsize=15)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.savefig(f"{figureFolder}/raw-cl-radius-density.pdf")
#%%%% rescaling data
dfcl["lfd"] =  dfcl.lfd
dfcl["A"] = dfcl.radius/(dfcl.pl*Kbt)**(1/3)
#%%%% plotting rescaled data
tempdf = dfcl
tempdf = tempdf.groupby(["length","density"]).mean().reset_index()

fig,ax = plt.subplots(figsize=(4,3))
plt.scatter(tempdf.lfd,tempdf.A,c=tempdf.length,cmap="turbo")

# power law fitting
x  = tempdf.lfd
y  = tempdf.A
fit = lambda x,a,b: a*x**b
(a,b),(_) = curve_fit(fit,x,y)
SER = compute_standard_regression(y,a*x**b,len(y),2)
x = np.sort(x.unique())
plt.plot(x,a*x**b,ls="--",color="k",lw=2)
plt.text(10,8e-1,r"$f^{%s}$ [%s]"%(round(b,3),round(SER,2)),fontsize=10)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$L$ ($\mu$m)",fontsize=15)
cm.ax.set_yticks(ticks=[7,14,21,28])
cm.ax.tick_params(labelsize=15)

plt.ylabel(r"$A_c$ (pN/$\mu$m)$^{-1/3}$",fontsize=15)
plt.xlabel(r"$f$ (pN/$\mu$m)",fontsize=15)
plt.tick_params(labelsize=15)
plt.xscale("log")
plt.yscale("log")
plt.ylim(0.2,2)
plt.tight_layout()
#plt.show()
plt.savefig(f"{figureFolder}/scaled-cl-A-epsilon.pdf")