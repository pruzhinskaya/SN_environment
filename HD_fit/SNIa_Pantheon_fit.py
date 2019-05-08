#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import quad
from math import *
from astropy.io import fits
import glob
import pylab 
from numpy.linalg import inv
from iminuit import Minuit, describe, Struct
import pandas
#from scipy.optimize import curve_fit
# from scipy.optimize import fmin


# In[2]:


# Physical constantes
clight = 299792.458
# Riess et al. 2016: H0 = 73.24 ± 1.74 km s−1 Mpc−1
H0 = 0.00007324

# Initial values of fitted parameters withb Pantheon results
# Omega_m   = 0.298 +- 0.022
# M_B       = ???
# alpha     = 0.154 +- 0.006
# beta      = 3.02 +- 0.06
# gamma     = 0.053 +- 0.009 <=> delta_M
Npar = 5
omgM0 = 0.298
M_b0 = -19.16
alpha0 = 0.154
beta0 = 3.030
gamma0 = 0.053


# In[3]:


# Comoving radial coordinate (c not included) corresponding to redshift z = c int_0^z dz/H(z)
# in LambdaCDM flat model
def intfun(z, omgM):
    return 1/sqrt(omgM*(1 + z)**3 + (1 - omgM))


# In[4]:


# Luminosity Distance function
def fitfundL(zcmb, omgM):
    mu = []
    for i in range (len(zcmb)): 
        zz = zcmb[i]
        mu.append(dL_z(zz, zz, omgM)) 
    return mu 


# In[5]:


# Luminosity Distance corresponding to a redshift z
def dL_z(zcmb, zhel, omgM):
    mu_zz = 5*log10((1 + zhel)*clight*(quad(intfun, 0, zcmb, args=(omgM))[0] / (10*H0)))
    return mu_zz


# In[6]:


# Distance modulus of observational data
def muexp(mb, x1, color, M_b, alpha, beta, gamma, Mstell, delta_mb, delta_c, delta_x1):
    mu=[]
    for i in range(len(mb)):
        delta_M = gamma / (1 + np.exp(-(Mstell[i]-10.13)/0.001))
        delta_b = delta_mb[i] - beta*delta_c[i] + alpha*delta_x1[i]
        mu.append(mb[i] - M_b + alpha*x1[i] - beta*color[i] + delta_M + delta_b)
        #mu.append(mb[i] - M_b)
    return mu


# In[7]:


# Uncertainty on distance modulus for an observational data ???
def dmuexp(dmb, db1, dcolor, alpha, beta):
    dmu=[]
    for i in range(len(dmb)):
        dmu.append(sqrt(dmb[i]**2+(alpha*dx1[i])**2+(beta*dcolor[i])**2))
    return dmu


# In[8]:


# Read covariance matrix 
# filecov = 'sys_full_long_G10.txt'
# filecov = 'cov.txt'
filecov = 'sys_full_long.txt'

SNcov = pandas.read_csv(filecov, comment='#', header=None)
#print('SNcov \n', SNcov)
Ndim = int(SNcov[0][0])
Cmat = np.zeros(Ndim*Ndim)
for i in range (len(SNcov)):
    if (i>0):
        Cmat[i-1] = SNcov[0][i]
Cmat = Cmat.reshape(Ndim,Ndim)
print('Cmat \n', Cmat)
for i in range(Ndim):
    for j in range(Ndim):
        if (i==j):
            Cmat[i,j] = 0.15**2
        else:
            Cmat[i,j] = 0
print('Cmat \n', Cmat)
Cinv = inv(Cmat)
print('Cinv \n', Cinv)


# In[9]:


# Definition of chi2 fucntion for minimization
class Chi2:

    def __init__(self, zcmb, zhel, mb, dmb, x1, dx1, color, dcolor, Mstell, delta_mb, delta_c, delta_x1): # Construct method
        '''Attributes'''
        self.chi2tot = 0.
        self.zcmb = zcmb 
        self.zhel = zhel
        self.mb = mb
        self.dmb = dmb
        self.x1 = x1
        self.dx1 = dx1
        self.color = color
        self.dcolor = dcolor
        self.Mstell = Mstell
        self.delta_mb = delta_mb
        self.delta_c = delta_c
        self.delta_x1 = delta_x1
        self.dL = np.zeros(shape=(len(zcmb))) 

    def chi2(self, omgM, M_b, alpha, beta, gamma):
        ''' Function to calculate the chi2 '''
        global Cinv
        mu_z = muexp(self.mb, self.x1, self.color, M_b, alpha, beta, gamma, self.Mstell, self.delta_mb, self.delta_c, self.delta_x1)
        
        # Loop for matrix construction
        for i in range(len(self.zcmb)):
            zz = self.zcmb[i]
            zzz = self.zhel[i]
            self.dL[i] = dL_z(zz, zzz, omgM)
        
        # Contruction of the chi2 by matrix product
        result =  pylab.dot( (mu_z-self.dL), pylab.dot((Cinv), (mu_z-self.dL)))
        self.chi2tot = result
        #print(result)
        return result


# In[10]:


if __name__=='__main__':
    
    # Read Pantheon data
    filename = 'Ancillary_G10.FITRES'
    filename2 = 'Ancillary_C11.FITRES'
    SNdata_G10 = pandas.read_csv(filename, comment='#', skiprows=6, header=0, delimiter=r"\s+")
    SNdata_C11 = pandas.read_csv(filename2, comment='#', skiprows=6, header=0, delimiter=r"\s+")

    zcmb = SNdata_G10['zCMB']
    zhel = SNdata_G10['zHD']
    mb = SNdata_G10['mB']
    dmb = SNdata_G10['mBERR']
    x1 = SNdata_G10['x1']
    dx1 = SNdata_G10['x1ERR']
    color = SNdata_G10['c']
    dcolor = SNdata_G10['cERR']
    Mstell = SNdata_G10['HOST_LOGMASS']
    dataset = SNdata_G10['IDSAMPLE']
    delta_mb = (SNdata_G10['biasCor_mB'] + SNdata_C11['biasCor_mB'])/2
    delta_c = (SNdata_G10['biasCor_c'] + SNdata_C11['biasCor_c'])/2
    delta_x1 = (SNdata_G10['biasCor_x1'] + SNdata_C11['biasCor_x1'])/2

#     filename = 'pant.txt'
#     dcol = ' '
#     ucol = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#     scol = ['name', 'zcmb', 'zhel', 'dz', 'mb', 'dmb', 'x1', 'dx1', 'color', 'dcolor', 'Mstell', 'dMstell', 'set']
#     SNdata = pandas.read_csv(filename, comment='#', header=None, delimiter=dcol, usecols=ucol) 
#     SNdata.columns = scol
#     zcmb = SNdata['zcmb']
#     zhel = SNdata['zhel']
#     mb = SNdata['mb']
#     dmb = SNdata['dmb']
#     x1 = SNdata['x1']
#     dx1 = SNdata['dx1']
#     color = SNdata['color']
#     dcolor = SNdata['dcolor']
#     Mstell = SNdata['Mstell']
#     dataset = SNdata['set']
    
    
    # Perform SN fit
    chi2function = Chi2(zcmb, zhel, mb, dmb, x1, dx1, color, dcolor, Mstell, delta_mb, delta_c, delta_x1)
    m = Minuit(chi2function.chi2, omgM=omgM0, M_b=M_b0, alpha=alpha0, beta=beta0, gamma=gamma0,
               limit_omgM=(0.2,0.4), limit_M_b=(-20.,-18.), limit_alpha=(0.1,0.2), limit_beta=(2.,3.5), limit_gamma=(-0.1,0.1),
               fix_omgM=False, fix_M_b=False, fix_alpha=False, fix_beta=False, fix_gamma=False, 
               print_level=1)
    m.migrad()
    #m.hesse
    
    # Extract fitted parameters
    omgM, M_b, alpha, beta, gamma = list(m.args)
    #domgM, dM_b, dalpha, dbeta, dgamma = list(np.sqrt(np.diag(m.matrix())))
    domgM, dM_b, dalpha, dbeta, dgamma = 0, 0, 0, 0, 0
    chi2ndf = chi2function.chi2(omgM, M_b, alpha, beta, gamma) / (len(mb)-Npar)
    print('chi2ndf = ', chi2ndf)


# In[11]:


# Experimental distance modulus
SNdata_G10['mu_exp'] = muexp(mb, x1, color, M_b, alpha, beta, gamma, Mstell, delta_mb, delta_c, delta_x1)
mu_exp = SNdata_G10['mu_exp']
SNdata_G10['dmu_exp'] = dmuexp(dmb, dx1, dcolor, alpha, beta)
dmu_exp = SNdata_G10['dmu_exp']

# Theoretical distance modulus (LambdaCDM)
SNdata_G10['mu_th'] = fitfundL(zcmb, omgM)
mu_th = SNdata_G10['mu_th']

# Residual = experimental - theoretical
mu_res = []
dmu_res = []
for i in range(len(mu_exp)):
    mu_res.append(mu_exp[i] - mu_th[i])
    dmu_res.append(dmu_exp[i])
SNdata_G10['res'] = mu_res
res = SNdata_G10['res']
SNdata_G10['dres'] = dmu_res
dres = SNdata_G10['dres']
    
# Theoretical curve (LambdaCDM)
xfunc = np.linspace(0.001, 2, 1000)
yfunc = np.zeros(len(xfunc))
yfunc = fitfundL(xfunc, omgM)

x0 = np.linspace(0.001, 2, 1000)
y0 = np.zeros(len(x0))
 
str0 = '{0:10s} {1:2.3f} \n'.format(r'$\chi^2_{/ndf}$ = ', chi2ndf)
str1 = '{0:10s} {1:2.3f} {2:4s} {3:2.3f} \n'.format(r'$\Omega_m$ =', omgM, r' $\pm$ ', domgM)
str2 = '{0:10s} {1:4.2f} {2:4s} {3:4.2f} \n'.format(r'$M_B$ = ', M_b, r' $\pm$ ', dM_b)
str3 = '{0:10s} {1:2.3f} {2:4s} {3:2.3f} \n'.format(r'$\alpha$ = ', alpha, r' $\pm$ ', dalpha)
str4 = '{0:10s} {1:3.2f} {2:4s} {3:3.2f} \n'.format(r'$\beta$ = ', beta, r' $\pm$ ', dbeta)
str5 = '{0:10s} {1:2.3f} {2:4s} {3:2.3f} \n'.format(r'$\gamma$ = ', gamma, r' $\pm$ ', dgamma)
str_fit = '{} {} {} {} {} {}'.format(str0, str1, str2, str3, str4, str5)
#print(str_fit)
    
fig = plt.figure(figsize=[8,8])
P1, P2 = fig.subplots(2, 1, sharex=True, sharey=False, gridspec_kw=dict(height_ratios=[3,1]))
fig.subplots_adjust(hspace=0)
P1.errorbar(zcmb, mu_exp, yerr=dmu_exp, marker='.', color="blue", linestyle="None", ecolor="blue", label='Pantheon data')
P1.plot(xfunc, yfunc, c="red", label="$\Lambda$CDM fit")
P1.set_xlabel('Redshift z')
P1.set_ylabel(r'Distance modulus $\mu = m_B - M_B + \alpha x_1 - \beta c + \Delta_M$')  
#P1.set_xscale('log')
P1.set_xlim(0., 2.3)
P1.set_ylim(30, 48)
P1.text(0.2, 31, str_fit, fontsize=12)
P1.legend(loc='best', shadow=True, fontsize='x-large')
P2.errorbar(zcmb, mu_res, yerr=dmu_res, marker='.', color="blue", linestyle="None", ecolor="blue")
P2.plot(x0, y0, c="grey", label="$\Lambda$CDM fit")
P2.set_xlabel('Redshift z')
P2.set_ylabel(r'$\mu - \mu_{\Lambda CDM}$')  
#P2.set_xscale('log')
#P2.set_xlim(0, 1.4)
#P2.set_ylim(-1, 1)
#fig.savefig("SNIa_lightcurve", dpi=150, bbox_inches='tight')


# In[49]:


setNB = dataset==0
setSDSS = dataset==2
setSNLS = dataset==3
setPS1 = dataset==1
setHST = dataset==4

fig = plt.figure(figsize=[8,8])
P1, P2 = fig.subplots(2, 1, sharex=True, sharey=False, gridspec_kw=dict(height_ratios=[3,1]))
fig.subplots_adjust(hspace=0)
P1.errorbar(zcmb, mu_exp, yerr=dmu_exp, marker='.', color="orange", linestyle="None", 
            ecolor="orange", label='Low and high-z')
P1.errorbar(zcmb[setSDSS], mu_exp[setSDSS], yerr=dmu_exp[setSDSS], marker='.', color="green", linestyle="None", 
            ecolor="green", label='SDSS')
P1.errorbar(zcmb[setSNLS], mu_exp[setSNLS], yerr=dmu_exp[setSNLS], marker='.', color="red", linestyle="None", 
            ecolor="red", label='SNLS')
P1.errorbar(zcmb[setPS1], mu_exp[setPS1], yerr=dmu_exp[setPS1], marker='.', color="blue", linestyle="None", 
            ecolor="blue", label='PS1')
P1.plot(xfunc, yfunc, c="grey", label="$\Lambda$CDM fit")
P1.set_xlabel('Redshift z')
P1.set_ylabel(r'Distance modulus $\mu = m_B - M_B + \alpha x_1 - \beta c + \Delta_M$')  
P1.set_xscale('log')
P1.set_xlim(5e-3, 2.3)
P1.set_ylim(30, 48)
P1.text(0.2, 31, str_fit, fontsize=12)
P1.legend(loc='best', shadow=True, fontsize='x-large')
P2.errorbar(zcmb, res, yerr=dres, marker='.', color="orange", linestyle="None", 
            ecolor="orange")
P2.errorbar(zcmb[setSDSS], res[setSDSS], yerr=dres[setSDSS], marker='.', color="green", linestyle="None", 
            ecolor="green")
P2.errorbar(zcmb[setSNLS], res[setSNLS], yerr=dres[setSNLS], marker='.', color="red", linestyle="None", 
            ecolor="red")
P2.errorbar(zcmb[setPS1], res[setPS1], yerr=dres[setPS1], marker='.', color="blue", linestyle="None", 
            ecolor="blue")
P2.plot(x0, y0, c="grey", label="$\Lambda$CDM fit")
P2.set_xlabel('Redshift z')
P2.set_ylabel(r'$\mu - \mu_{\Lambda CDM}$')  
P2.set_xscale('log')
P2.set_xlim(5e-3, 2)
P2.set_ylim(-1, 1)


# In[12]:


fig = plt.figure(figsize=[6,4])
ax  = fig.add_subplot(111)
ax.hist(dataset, 20, range=[0,20], color='blue', alpha=0.3)
ax.set_xlabel("set")


# In[52]:


Mlow = Mstell<=10
Mhigh = Mstell>10

fig = plt.figure(figsize=[10,4])
ax  = fig.add_subplot(121)
ax.hist(x1[Mlow], 20, range=[-2,2], color='blue', alpha=0.3, label='log(M) < 10')
ax.hist(x1[Mhigh], 20, range=[-2,2], color='red', alpha=0.3, label='log(M) > 10')
ax.set_xlabel("x1")
ax  = fig.add_subplot(122)
ax.hist(color[Mlow], 20, range=[-0.2,0.2], color='blue', alpha=0.3, label='log(M) < 10')
ax.hist(color[Mhigh], 20, range=[-0.2,0.2], color='red', alpha=0.3, label='log(M) > 10')
ax.set_xlabel("c")
ax.legend(loc='best', shadow=True, fontsize='large')


# In[ ]:




