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
    plt.errorbar(tempdfm.density,tempdfm["y-freq"],tempdfs["y-freq"],color=cmap((length-minl)/diff))

plt.xlabel(r"$\rho_{m}$ $(Motors/\mu m^{2})$",fontsize=15)
plt.ylabel(r"$\nu$ ($1/s$)",fontsize=15)
plt.tick_params(labelsize=15)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$\ell_{MT}$  $(\mu m)$",fontsize=15)
cm.ax.tick_params(labelsize=15)
cm.ax.set_yticks(ticks=[7, 14, 21, 28])

plt.tight_layout()
plt.savefig(f"{figureFolder}/raw-vl-freq-density.pdf")
#%%%% rescaling variables
dfvl["chi"] = (dfvl["y-freq"]*dfvl.pl)/motorVelocity
dfvl["epsilon"] = dfvl.lfd*(dfvl.pl**2)/(Kbt)
#%%%% plotting trasformed data
tempdf = dfvl
tempdf = tempdf.groupby(["length","density"]).mean().reset_index()

fig,ax = plt.subplots(figsize=(4,3))
plt.scatter(tempdf.epsilon,tempdf.chi,c=tempdf.length,cmap="turbo")
plt.ylabel(r"$\chi$",fontsize=15)
plt.xlabel(r"$\epsilon$",fontsize=15)
plt.tick_params(labelsize=15)
plt.xscale("log")
plt.yscale("log")

x  = tempdf.epsilon
y  = tempdf.chi

fit = lambda x,a,b: a*x**b
(a,b),(_) = curve_fit(fit,x,y)
x = np.sort(x.unique())
plt.plot(x,a*x**b,ls="--",label=rf"$\alpha$:{round(b,4)}",color="k",lw=3)
plt.text(0.2e10,2.6e2,r"$\epsilon^{%s}$"%(round(b,3)),fontsize=15)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$\ell_{MT}$  $(\mu m)$",fontsize=15)
cm.ax.set_yticks(ticks=[7, 14, 21, 28])
cm.ax.tick_params(labelsize=15)

plt.tight_layout()
#plt.show()
plt.savefig(f"{figureFolder}/scaled-vl-chi-epsilon.pdf")
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
    plt.errorbar(tempdfm.density,tempdfm["y-freq"],yerr=tempdfs["y-freq"],color=cmap((length-minl)/diff))
plt.xlabel(r"$\rho_{m}$ $(Motors/\mu m^{2})$",fontsize=15)
plt.ylabel(r"$\nu$ ($1/s$)",fontsize=15)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$\ell_{MT}$  $(\mu m)$",fontsize=15)
cm.ax.set_yticks(ticks=[7, 14, 21, 28])
cm.ax.tick_params(labelsize=15)

plt.tick_params(labelsize=15)
plt.tight_layout()
#plt.show()
plt.savefig(f"{figureFolder}/raw-cl-freq-density.pdf")
#%%%% rescaling variables
dfcl["chi"] = (dfcl["y-freq"]*dfcl.pl)/motorVelocity
dfcl["epsilon"] = dfcl.lfd*(dfcl.pl**2)/(Kbt)
#%%%% plotting trasformed data
tempdf = dfcl
tempdf.groupby(["length","density"]).mean().reset_index()

fig,ax = plt.subplots(figsize=(4,3))
plt.scatter(tempdf.epsilon,tempdf.chi,c=tempdf.length,cmap="turbo")
plt.ylabel(r"$\chi$",fontsize=15)
plt.xlabel(r"$\epsilon$",fontsize=15)
plt.tick_params(labelsize=15)
plt.xscale("log")
plt.yscale("log")

x  = tempdf.epsilon
y  = tempdf.chi

fit = lambda x,a,b: a*x**b
(a,b),(_) = curve_fit(fit,x,y)

x = np.sort(x.unique())
plt.plot(x,a*x**b,ls="--",label=rf"$\alpha$:{round(b,4)}",color="k",lw=3)
plt.text(0.65e11,4.8e2,r"$\epsilon^{%s}$"%(round(b,3)),fontsize=15)
plt.xticks(ticks=[5e10],minor=True)

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(minl,maxl)),ax=ax)
cm.ax.set_ylabel(r"$\ell_{MT}$  $(\mu m)$",fontsize=15)
cm.ax.set_yticks(ticks=[7, 14, 21, 28])
cm.ax.tick_params(labelsize=15)

plt.tight_layout()
plt.savefig(f"{figureFolder}/scaled-cl-chi-epsilon.pdf")
