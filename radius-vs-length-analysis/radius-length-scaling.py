
#%% importing libraries
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
#%%
def compute_standard_regression(y_true,y_pred,n,p):
    ss_res = np.sum((y_true - y_pred)**2)
    return(np.sqrt(ss_res/(n-p)))
#%% scaling varibles
width      = 0.025     # microns
dutyRatio  = 1         # dimension less
stallForce = 5         # pico-Newton
Kbt        = 4.3*1e-3  # pN micron
#%% taken from pampaloni et al.
lpinf = 6300
pl = lambda l: lpinf/(1+441/l**2)
#%% color map
cmap = matplotlib.colormaps.get("turbo")
#%% figure folder
figureFolder = "radius-length"
#%% Analysis of varible lp data
#%%% reading data
dfvl = pd.read_csv("radius-freq-vpl.csv",index_col=0)
dfvl["lfd"] = dfvl["density"]*(width*stallForce*dutyRatio)
dfvl["pl"] = pl(dfvl.length)
#%%%% mean and std data frames
dfvl_M = dfvl.groupby(["length","density"]).mean().reset_index()
dfvl_S = dfvl.groupby(["length","density"]).std().reset_index()
#%%%% plotting raw data
mind =  dfvl_M.density.min()
maxd =  dfvl_M.density.max()
diff = maxd-mind

fig,ax = plt.subplots(figsize=(4,3))
for density in dfvl_M.density.unique():
    tempdfm = dfvl_M[dfvl_M.density == density]
    tempdfs = dfvl_S[dfvl_S.density == density]

    plt.errorbar(tempdfm.length,tempdfm.radius,yerr=tempdfs.radius,color=cmap((density-mind)/diff),lw=1)

cm=plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  (Motors/$\mu$m$^{2}$)",fontsize=15)
cm.ax.tick_params(labelsize=15)

plt.xlabel(r"$L$ ($\mu$m)",fontsize=15)
plt.ylabel(r"R ($\mu$m)",fontsize=15)
plt.tick_params(labelsize=15)
plt.xticks(ticks=[7, 14, 21, 28])
plt.ylim(0.5,2.5)
plt.tight_layout()
#plt.show()
plt.savefig(f"{figureFolder}/raw-vl-radius-length.pdf")
plt.close()
#%%%% rescaling data
dfvl["xi"] =  dfvl["radius"]*dfvl["lfd"]**(1/3)/Kbt**(1/3)
#%%%% plotting rescaled data with theoretical fit for variable persistence length
tempdf = dfvl
tempdf = tempdf.groupby(["length","density"]).mean().reset_index()
dmin = tempdf.density.min()
dmax = tempdf.density.max()

fig,ax = plt.subplots(figsize=(4,3))
plt.scatter(tempdf.length,tempdf.xi,edgecolor=cmap((tempdf.density-dmin)/(dmax-dmin)),facecolor="none")

# theoretical fitting using variable persistence length
x = tempdf.length
y = tempdf.xi
fit1 = lambda x,a: a*pl(x)**(1/3)
(a),(_) = curve_fit(fit1,x,y)
SER = compute_standard_regression(y,fit1(x,a),len(y),1)
plt.plot(x,fit1(x,a),ls="--",label=rf"{np.round(a[0],3)}$\ell_{{p}}^{{v~1/3}}$ [{round(SER,2)}]",lw=3,color="black")

# theoretical fitting using constant persistence length
fit1 = lambda x,a: a*(np.zeros(len(x))+5000)**(1/3)
(a),(_) = curve_fit(fit1,x,y)
SER = compute_standard_regression(y,fit1(x,a),len(y),1)
plt.plot(x,fit1(x,a),ls="-.",label=rf"{np.round(a[0],3)}$\ell_{{p}}^{{c~1/3}}$ [{round(SER,2)}]",lw=3,color="black")

cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind-20,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  (Motors/$\mu$m$^{2}$)",fontsize=15)
cm.ax.tick_params(labelsize=15)

plt.ylabel(r"$\xi$ ($\mu$m$^{1/3}$)",fontsize=15)
plt.xlabel(r"$L$ ($\mu$m)",fontsize=15)
plt.tick_params(labelsize=15)
plt.ylim(5,34)
plt.legend(frameon=False,loc="upper left",fontsize=10)
plt.tight_layout()
#plt.show()
plt.savefig(f"{figureFolder}/scaled-vl-xi-length-with-overlay.pdf")
#%% Analysis of constant lp data
#%%% reading data
dfcl = pd.read_csv("radius-freq-cpl.csv",index_col=0)
dfcl["lfd"] = dfcl["density"]*(width*stallForce*dutyRatio)
dfcl["pl"] = 5000
#%%%% mean and std data frames
dfcl_M = dfcl.groupby(["length","density"]).mean().reset_index()
dfcl_S = dfcl.groupby(["length","density"]).std().reset_index()
#%%%% plotting raw data
mind =  dfcl_M.density.min()
maxd =  dfcl_M.density.max()+1
diff = maxd-mind

fig,ax = plt.subplots(figsize=(4,3))
for density in dfcl_M.density.unique():
    tempdfm = dfcl_M[dfcl_M.density == density]
    tempdfs = dfcl_S[dfcl_S.density == density]

    plt.errorbar(tempdfm.length,tempdfm.radius,yerr=tempdfs.radius,color=cmap((density-mind)/diff),lw=1)

cm=plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  (Motors/$\mu$m$^{2}$)",fontsize=15)
cm.ax.tick_params(labelsize=15)

plt.xlabel(r"$L$ ($\mu$m)",fontsize=15)
plt.ylabel(r"R ($\mu$m)",fontsize=15)
plt.tick_params(labelsize=15)
plt.xticks(ticks=[7, 14, 21, 28])
plt.ylim(0.5,2.5)
plt.tight_layout()
#plt.show()
plt.savefig(f"{figureFolder}/raw-cl-radius-length.pdf")
#%%%% rescaling data
dfcl["xi"] =  dfcl["radius"]*dfcl["lfd"]**(1/3)/Kbt**(1/3)
#%%%% plotting rescaled data with theoretical fit for variable persistence length
tempdf = dfcl
tempdf = tempdf.groupby(["length","density"]).mean().reset_index()
dmin = tempdf.density.min()
dmax = tempdf.density.max()

fig,ax = plt.subplots(figsize=(4,3))
plt.scatter(tempdf.length,tempdf.xi,edgecolor=cmap((tempdf.density-dmin)/(dmax-dmin)),facecolor="none")

# theoretical fitting using variable persistence length
x = tempdf.length
y = tempdf.xi
fit1 = lambda x,a: a*pl(x)**(1/3)
(a),(_) = curve_fit(fit1,x,y)
SER = (compute_standard_regression(y,fit1(x,a),len(y),1))
plt.plot(x,fit1(x,a),ls="--",label=rf"{np.round(a[0],3)}$\ell_{{p}}^{{v~1/3}}$ [{round(SER,2)}]",lw=3,color="black")


# theoretical fitting using constant persistence length
x = tempdf.length
y = tempdf.xi
fit1 = lambda x,a: a*(np.zeros(len(x))+5000)**(1/3)
(a),(_) = curve_fit(fit1,x,y)
SER = (compute_standard_regression(y,fit1(x,a),len(y),1))
plt.plot(x,fit1(x,a),ls="-.",label=rf"{np.round(a[0],3)}$\ell_{{p}}^{{c~1/3}}$ [{round(SER,2)}]",lw=3,color="black")


cm = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(mind-20,maxd)),ax=ax)
cm.ax.set_ylabel(r"$\rho_{m}$  (Motors/$\mu$m$^{2}$",fontsize=15)
cm.ax.tick_params(labelsize=15)


plt.ylabel(r"$\xi$ ($\mu$m$^{1/3})$",fontsize=15)
plt.xlabel(r"$L$ ($\mu$m)",fontsize=15)
plt.tick_params(labelsize=15)
plt.ylim(5,34)
plt.legend(frameon=False,loc="lower right",fontsize=10)
plt.tight_layout()
#plt.show()
plt.savefig(f"{figureFolder}/scaled-cl-xi-length-with-overlay.pdf")