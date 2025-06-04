#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:06:25 2024

@author: aman
"""
#%% importing libraries
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FixedLocator
#%%
def compute_standard_regression(y_true,y_pred,n,p):
    ss_res = np.sum((y_true - y_pred)**2)
    return(np.sqrt(ss_res/(n-p)))
#%% scaling varibles
width      = 0.025     # micron
dutyRatio  = 1         # dimension less
stallForce = 5         # pico-Newton
Kbt        = 4.3*1e-3  # pN micron
motorVelocity = 0.1    # micron/s
#%% figure folder
figureFolder = "freq-density"
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
#%%%% raw data
fig,ax = plt.subplots(figsize=(4,3))
minl =  dfvl_M.length.min()
maxl =  dfvl_M.length.max()
diff = maxl-minl
for length in dfvl_M.length.unique():
    tempdfm = dfvl_M[dfvl_M.length == length]
    tempdfs = dfvl_S[dfvl_S.length == length]
    #plt.errorbar(tempdfm.density,tempdfm["y-freq"],tempdfs["y-freq"],color=cmap((length-minl)/diff))
    plt.errorbar(tempdfm.density,(tempdfm["y-freq"]+tempdfm["x-freq"])/2,(tempdfs["y-freq"]+tempdfs["x-freq"])/2,color=cmap((length-minl)/diff),lw=1)

plt.xlabel(r"$\rho_{m}$ (Motors/$\mu$m$^{2}$)",fontsize=15)
plt.ylabel(r"$\nu$ ($1/s$)",fontsize=15)
plt.tick_params(labelsize=15)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$L$ ($\mu$m)",fontsize=15)
cm.ax.tick_params(labelsize=15)
cm.ax.set_yticks(ticks=[7, 14, 21, 28])

plt.tight_layout()
plt.savefig(f"{figureFolder}/raw-vl-freq-density.pdf")
plt.close("all")
#%%%% rescaling variables
dfvl["chi"] = ((dfvl["y-freq"]+dfvl["x-freq"])/2*(dfvl.pl*Kbt)**(1/3))/dfvl.velocity
#%%%% plotting trasformed data
tempdf = dfvl
tempdf = tempdf.groupby(["length","density"]).mean().reset_index()

maxl = dfvl.length.max()
minl = dfvl.length.min()

fig,ax = plt.subplots(figsize=(4,3))
plt.scatter(tempdf.lfd,tempdf.chi,edgecolor=cmap((tempdf.length-minl)/(maxl-minl)),facecolor="none")

x  = tempdf.lfd
y  = tempdf.chi
m,c = np.polyfit(np.log(x),np.log(y),1)
plt.plot(x,np.exp(c)*x**m,ls="--",color="k",lw=3)
SER = compute_standard_regression(y,np.exp(c)*x**m,len(y),2)
plt.text(10,3e-1,r"$f^{%s}$ [%s]"%(round(m,3),round(SER,3)),fontsize=10)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$L$ $(\mu m)$",fontsize=15)
cm.ax.set_yticks(ticks=[7, 14, 21, 28])
cm.ax.tick_params(labelsize=15)

plt.ylabel(r"$\chi_v$ (pN/$\mu$m)$^{1/3}$",fontsize=15)
plt.xlabel(r"$f$ (pN/$\mu$m)",fontsize=15)
plt.tick_params(labelsize=15)
plt.xscale("log")
plt.yscale("log")

ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_minor_formatter(FormatStrFormatter('%g'))
ax.tick_params(axis='both', which='minor', labelsize=15, label1On=True)
ax.xaxis.set_minor_locator(FixedLocator([5,10,15,20,25,30]))
ax.yaxis.set_minor_locator(FixedLocator([0.25,0.5,0.75,1,2]))
ax.tick_params(which='minor', length=3.5,width=1, color='black')

plt.ylim(1e-1,1)

plt.tight_layout()
#plt.show()
plt.savefig(f"{figureFolder}/scaled-vl-f-epsilon.pdf")
plt.close("all")
#%% Analysis of constant lp data
dfcl = pd.read_csv("radius-freq-cpl.csv",index_col=0)
dfcl["lfd"] = dfcl["density"]*(width*stallForce*dutyRatio)
dfcl["pl"]  = 5000
dfcl_M = dfcl.groupby(["length","density"]).mean().reset_index()
dfcl_S = dfcl.groupby(["length","density"]).std().reset_index()
#%%%% raw data
fig,ax = plt.subplots(figsize=(4,3))
minl =  dfcl_M.length.min()
maxl =  dfcl_M.length.max()
diff = maxl-minl

for length in dfcl_M.length.unique():
    tempdfm = dfcl_M[dfcl_M.length == length]
    tempdfs = dfcl_S[dfcl_S.length == length]
    plt.errorbar(tempdfm.density,(tempdfm["y-freq"]+tempdfm["x-freq"])/2,(tempdfs["y-freq"]+tempdfs["x-freq"])/2,color=cmap((length-minl)/diff),lw=1)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$L$ $(\mu m)$",fontsize=15)
cm.ax.set_yticks(ticks=[7, 14, 21, 28])
cm.ax.tick_params(labelsize=15)

plt.xlabel(r"$\rho_{m}$ (Motors/$\mu$m$^{2}$)",fontsize=15)
plt.ylabel(r"$\nu$ (1/s)",fontsize=15)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.savefig(f"{figureFolder}/raw-cl-freq-density.pdf")
plt.close("all")
#%%%% rescaling variables
dfcl["chi"] = ((dfcl["y-freq"]+dfcl["y-freq"])/2*(dfcl.pl*Kbt)**(1/3))/dfcl.velocity
#%%%% plotting trasformed data
tempdf = dfcl
tempdf.groupby(["length","density"]).mean().reset_index()

maxl = dfcl.length.max()
minl = dfcl.length.min()

fig,ax = plt.subplots(figsize=(4,3))
plt.scatter(tempdf.lfd,tempdf.chi,edgecolor=cmap((tempdf.length-minl)/(maxl-minl)),facecolor="none")

x  = tempdf.lfd
y  = tempdf.chi
m,c = np.polyfit(np.log(x),np.log(y),1)
plt.plot(x,np.exp(c)*x**m,ls="--",color="k",lw=3)
SER = compute_standard_regression(y,np.exp(c)*x**m,len(y),2)
plt.text(10,3.5e-1,r"$f^{%s}$ [%s]"%(round(m,3),round(SER,3)),fontsize=10)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$L$ ($\mu$m)",fontsize=15)
cm.ax.set_yticks(ticks=[7, 14, 21, 28])
cm.ax.tick_params(labelsize=15)

plt.ylabel(r"$\chi_c$ (pN/$\mu$m)$^{1/3}$",fontsize=15)
plt.xlabel(r"$f$ (pN/$\mu$m)",fontsize=15)
plt.tick_params(labelsize=15)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-1,1)

ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_minor_formatter(FormatStrFormatter('%g'))
ax.tick_params(axis='both', which='minor', labelsize=15, label1On=True)
ax.xaxis.set_minor_locator(FixedLocator([5,10,15,20,25,30]))
ax.yaxis.set_minor_locator(FixedLocator([0.25,0.5,0.75,1,2]))
ax.tick_params(which='minor', length=3.5,width=1, color='black')

plt.tight_layout()
plt.savefig(f"{figureFolder}/scaled-cl-f-epsilon.pdf")
plt.close("all")
