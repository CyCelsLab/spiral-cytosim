#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:40:52 2024

@author: aman
"""
#%% importing libraries
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import matplotlib
#%% scaling varibles
width      = 0.025     # micron
dutyRatio  = 1         # dimension less
stallForce = 5         # pico-Newton
Kbt        = 4.3*1e-3  # pN micron
motorVelocity = 0.1    # micron/s
#%% figure folder
figureFolder = "freq-length"
#%%
def compute_standard_regression(y_true,y_pred,n,p):
    ss_res = np.sum((y_true - y_pred)**2)
    return(ss_res)
#    return(np.sqrt(ss_res/(n-p-1)))
#%% taken from pampaloni et al.
lpinf = 6300
pl = lambda l: lpinf/(1+441/l**2)
#%%
plc = lambda l: np.ones(len(l))*5000
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
mind =  dfvl_M.density.min()
maxd =  dfvl_M.density.max()
diff = maxd-mind

fig,ax = plt.subplots(figsize=(4,3))
for density in dfvl_M.density.unique():

    tempdfm = dfvl_M[dfvl_M.density == density]
    tempdfs = dfvl_S[dfvl_S.density == density]

    plt.errorbar(tempdfm.length,tempdfm["y-freq"],tempdfs["y-freq"],color=cmap((density-mind)/diff),lw=1)

cm=plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  (Motors/($\mu$m$^{2}$)",fontsize=15)
cm.ax.tick_params(labelsize=15)

plt.xlabel(r"$L$ ($\mu$m)",fontsize=15)
plt.ylabel(r"$\nu$ (1/s)",fontsize=15)
plt.tick_params(labelsize=15)

plt.tight_layout()
#plt.show()
plt.savefig(f"{figureFolder}/raw-vl-freq-length.pdf")
plt.close("all")
#%%%% rescaling variables
dfvl['theta'] =  (dfvl["y-freq"])*(Kbt/dfvl.lfd)**(1/3)/motorVelocity
#%%%% plotting trasformed data
tempdf = dfvl
tempdf = tempdf.groupby(["length","density"]).mean().reset_index()
dmin = tempdf.density.min()
dmax = tempdf.density.max()

fig,ax = plt.subplots(figsize=(4.1,3))
plt.scatter(tempdf.length,tempdf.theta,color=cmap((tempdf.density-dmin)/(dmax-dmin)))

# theoretical fitting to constant persistence length model 
y = tempdf.theta
x = tempdf.length
fit1 = lambda x,a: a/(pl(x))**(1/3)
(a),(_) = curve_fit(fit1,x,y,method="trf")
SER = compute_standard_regression(y,fit1(x,a),len(y),1)
x = np.linspace(4,31)
plt.plot(x,fit1(x,a),ls="--",label=r"%s$/\ell_{p}^{v~1/3}$ [%4.2e]"%(np.round(a[0],3),SER),lw=2,color="black")

# theoretical fitting to variable persistence length model 
y = tempdf.theta
x = tempdf.length
fit1 = lambda x,a: a/(plc(x))**(1/3)
(a),(_) = curve_fit(fit1,x,y,method="trf")
SER = compute_standard_regression(y,fit1(x,a),len(y),1)
x = np.linspace(4,31)
plt.plot(x,fit1(x,a),ls="-.",label=r"%s$/\ell_{p}^{c~1/3}$ [%4.2e]"%(np.round(a[0],3),SER),lw=2,color="black")


#colorbar
cm=plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  (Motors/$\mu$m$^{2}$)",fontsize=15)
cm.ax.tick_params(labelsize=15)

plt.ticklabel_format(style='sci',axis='y', scilimits=(0, 0))
plt.ylabel(r"$\Theta$ ($\mu$m$^{1/3}$)",fontsize=15)
plt.xlabel(r"$L$ ($\mu$m)",fontsize=15)
plt.tick_params(labelsize=15)
plt.ylim(2e-3,2e-2)
plt.legend(frameon=False,fontsize=9)
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"{figureFolder}/scaled-vl-theta-length-with-overlay.pdf")
plt.close("all")
#%% Analysis of constant lp data
dfcl = pd.read_csv("radius-freq-cpl.csv",index_col=0)
dfcl["lfd"] = dfcl["density"]*(width*stallForce*dutyRatio)
dfcl["pl"] = 5000
dfcl_M = dfcl.groupby(["length","density"]).mean().reset_index()
dfcl_S = dfcl.groupby(["length","density"]).std().reset_index()
#%%% frequency vs length
#%%%% raw data
mind =  dfcl_M.density.min()
maxd =  dfcl_M.density.max()
diff = maxd-mind

fig,ax = plt.subplots(figsize=(4,3))
for density in dfcl_M.density.unique():
    tempdfm = dfcl_M[dfcl_M.density == density]
    tempdfs = dfcl_S[dfcl_S.density == density]
    plt.errorbar(tempdfm.length,tempdfm["y-freq"],tempdfs["y-freq"],color=cmap((density-mind)/diff),lw=1)

cm=plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  Motors/($\mu$m$^{2}$)",fontsize=15)
cm.ax.tick_params(labelsize=15)

plt.xlabel(r"$L$ ($\mu$m)",fontsize=15)
plt.ylabel(r"$\nu$ (1/s)",fontsize=15)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.savefig(f"{figureFolder}/raw-cl-freq-length.pdf")
plt.close("all")
#%%%% rescaling variables
dfcl['theta'] =  (dfcl["y-freq"])*(Kbt/dfcl.lfd)**(1/3)/motorVelocity
#%%%% plotting trasformed data
tempdf = dfcl
tempdf = tempdf.groupby(["length","density"]).mean().reset_index()
dmin = tempdf.density.min()
dmax = tempdf.density.max()

fig,ax = plt.subplots(figsize=(4.1,3))
plt.scatter(tempdf.length,tempdf.theta,color=cmap((tempdf.density-dmin)/(dmax-dmin)))

# theoretical fitting to constant persistence length model 
x = tempdf.length
y = (tempdf.theta)
fit1 = lambda x,a: a/(np.zeros(len(x))+5000)**(1/3)
(a),(_) = curve_fit(fit1,x,y)
SER = compute_standard_regression(y,fit1(x,a),len(y),1)
x = np.linspace(4,31)
plt.plot(x,fit1(x,a),ls="-.",label=r"%s$/\ell_{p}^{c~1/3}$ [%4.2e]"%(np.round(a[0],3),SER),lw=2,color="black")

# theoretical fitting to variable persistence length model 
x = tempdf.length
y = (tempdf.theta)
fit1 = lambda x,a: a/(pl(x))**(1/3)
(a),(_) = curve_fit(fit1,x,y,method="trf")
SER = compute_standard_regression(y,fit1(x,a),len(y),1)
x = np.linspace(4,31)
plt.plot(x,fit1(x,a),ls="--",label=r"%s$/\ell_{p}^{v~1/3}$ [%4.2e]"%(np.round(a[0],3),SER),lw=2,color="black")

#colorbar
cm=plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  (Motors/$\mu$m$^{2}$)",fontsize=15)
cm.ax.tick_params(labelsize=15)

plt.ticklabel_format(style='sci',axis='y', scilimits=(0, 0))
plt.ylabel(r"$\Theta$ ($\mu$m$^{1/3}$)",fontsize=15)
plt.xlabel(r"$L$ ($\mu$m)",fontsize=15)
plt.tick_params(labelsize=15)
plt.ylim(2e-3,2e-2)
plt.legend(frameon=False,fontsize=9)
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"{figureFolder}/scaled-cl-theta-length-with-overlay.pdf")
plt.close("all")
