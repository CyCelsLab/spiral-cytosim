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
mind =  dfvl_M.density.min()
maxd =  dfvl_M.density.max()
diff = maxd-mind

fig,ax = plt.subplots(figsize=(4,3))
for density in dfvl_M.density.unique():

    tempdfm = dfvl_M[dfvl_M.density == density]
    tempdfs = dfvl_S[dfvl_S.density == density]

    plt.errorbar(tempdfm.length,tempdfm["y-freq"],tempdfs["y-freq"],color=cmap((density-mind)/diff),lw=1)

cm=plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  $Motors/(\mu m^{2})$",fontsize=15)
cm.ax.tick_params(labelsize=15)

plt.xlabel(r"$\ell_{MT}$ $(\mu m)$",fontsize=15)
plt.ylabel(r"$\nu$ ($1/s$)",fontsize=15)
plt.tick_params(labelsize=15)

plt.tight_layout()
plt.savefig(f"{figureFolder}/raw-vl-freq-length.pdf")
#%%%% rescaling variables
dfvl['theta'] = (dfvl["y-freq"])/dfvl.lfd**(1/3)
#%%%% plotting trasformed data

tempdf = dfvl
tempdf = tempdf.groupby(["length","density"]).mean().reset_index()

fig,ax = plt.subplots(figsize=(4,3))
plt.scatter(tempdf.length,tempdf.theta,c=tempdf.density,cmap="turbo")
plt.ylabel(r"$\Theta$ $\left(s  \left(\frac{\mu m}{pN}\right)^{1/3}\right)$",fontsize=15)
plt.xlabel(r"$\ell_{MT}$ $(\mu m)$",fontsize=15)
plt.tick_params(labelsize=15)

x = tempdf.length
y = tempdf.theta
fit = lambda x,a,b: a*x**b
(a,b),(_) = curve_fit(fit,x,y)
x = np.sort(x.unique())
plt.plot(x,a*x**b,ls="--",lw=3,color="k")
plt.text(20,0.0055,r"$\ell_{MT}^{%s}$"%(round(b,3)),fontsize=15)

cm=plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  $Motors/(\mu m^{2})$",fontsize=15)
cm.ax.tick_params(labelsize=15)

ax.ticklabel_format(style='sci',axis='y', scilimits=(0, 0))
plt.tight_layout()

plt.savefig(f"{figureFolder}/scaled-vl-theta-length.pdf")
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
cm.ax.set_ylabel(r"$\rho_{m}$  $Motors/(\mu m^{2})$",fontsize=15)
cm.ax.tick_params(labelsize=15)

plt.xlabel(r"$\ell_{MT}$ $(\mu m)$",fontsize=15)
plt.ylabel(r"$\nu$ ($1/s$)",fontsize=15)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.savefig(f"{figureFolder}/raw-cl-freq-length.pdf")
#%%%% rescaling variables
dfcl['theta'] = (dfcl["y-freq"])/dfcl.lfd**(1/3)
#%%%% plotting trasformed data

tempdf = dfcl
tempdf = tempdf.groupby(["length","density"]).mean().reset_index()

fig,ax = plt.subplots(figsize=(4,3))
plt.scatter(tempdf.length,tempdf.theta,c=tempdf.density,cmap="turbo")
plt.ylabel(r"$\Theta$ $\left(s  \left(\frac{\mu m}{pN}\right)^{1/3}\right)$",fontsize=15)
plt.xlabel(r"$\ell_{MT}$ $(\mu m)$",fontsize=15)
plt.tick_params(labelsize=15)

x = tempdf.length
y = tempdf.theta
fit = lambda x,a,b: a*x**b
(a,b),(_) = curve_fit(fit,x,y)
x = np.sort(x.unique())
plt.plot(x,a*x**b,ls="--",lw=3,color="black")
plt.text(20,0.00329,r"$\ell_{MT}^{%s}$"%np.round(b,3),fontsize=15)

cm=plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  $Motors/(\mu m^{2})$",fontsize=15)
cm.ax.tick_params(labelsize=15)
ax.ticklabel_format(style='sci',axis='y', scilimits=(0, 0))

plt.tight_layout()
plt.savefig(f"{figureFolder}/scaled-cl-theta-length.pdf")
#%% plotting with theoretical fit and power law fit.
#%%%% plotting trasformed data
tempdf = dfvl
tempdf = tempdf.groupby(["length","density"]).mean().reset_index()
dmin = tempdf.density.min()
dmax = tempdf.density.max()

fig,ax = plt.subplots(figsize=(4.1,3))

plt.scatter(tempdf.length,tempdf.theta,facecolor="none",edgecolor=cmap((tempdf.density-dmin)/(dmax-dmin)))
plt.ylabel(r"$\Theta$ $\left(s  \left(\frac{\mu m}{pN}\right)^{1/3}\right)$",fontsize=15)
plt.xlabel(r"$\ell_{MT}$ $(\mu m)$",fontsize=15)
plt.tick_params(labelsize=15)

# power law fit
x = tempdf.length
y = tempdf.theta
fit = lambda x,a,b: a*x**b
(a,b),(_) = curve_fit(fit,x,y)
x = np.linspace(1,50)
plt.plot(x,a*x**b,ls="-.",label=r"$%s\ell_{{MT}}^{%s}$"%(round(a,3),round(b,3)),lw=2,color="red")

# theoretical fit
x = tempdf.length
y = tempdf.theta
fit1 = lambda x,a: a*motorVelocity/pl(x)**(1/3)/Kbt**(1/3)
(a),(_) = curve_fit(fit1,x,y)
x = np.linspace(1,50)
plt.plot(x,fit1(x,a),ls="-.",label=r"%s$v/(KT\ell_{p}(\ell))^{1/3}$"%np.round(a[0],3),lw=2,color="navy")

plt.legend(frameon=False)
cm=plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  $(Motors/\mu m^{2})$",fontsize=15)
cm.ax.tick_params(labelsize=15)

ax.ticklabel_format(style='sci',axis='y', scilimits=(0, 0))
plt.tight_layout()
plt.savefig(f"{figureFolder}/scaled-vl-theta-length-with-overlay.pdf")
