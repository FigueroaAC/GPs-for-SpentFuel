#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:42:11 2020

@author: AFigueroa
"""


import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano
import matplotlib.pyplot as plt
# In[]


def flux(alpha1,alpha2):
    sigma1 = 0.3e-24
    dE1 = 1e-1
    sigma2 = 0.1e-24
    dE2 = 1
    return alpha1*sigma1*dE1 + alpha2*sigma2*dE2

def fp(alpha1,alpha2,y,t):
    sigma1 = 0.3e-24
    dE1 = 1e-1
    sigma2 = 0.1e-24
    dE2 = 1
    N235 = 20e23
    e = np.exp((alpha1*sigma1*dE1 + alpha2*sigma2*dE2)*t)    
    return N235*y*e

N235 = 20e23
y = [0.07,0.042]
t = 86400*30
N = [fp(1e10,2e14,i,t) for i in y]

def leftside(Ni,Nu,yi,ti):
    return np.log(Ni/(Nu*yi))*(1/ti)

Ymeas = leftside(N[0],N235,y[0],t)

def ll(amp,alpha1,alpha2,sigma,yobs):
    sigma1 = 0.3e-24
    dE1 = 1e-1
    sigma2 = 0.1e-24
    dE2 = 1
    mu = (amp*alpha1*dE1*sigma1) + (amp*alpha2*sigma2*dE2)
    return ((mu - yobs)/(2*sigma))**2

def sampler(points,sigma,yobs):
    #a1 = 40 - (25*np.random.random(points))
    #a2 = 40 - (25*np.random.random(points))
    amp = 60 - (25*np.random.random(points))
    a1 = np.random.random(points)
    a2 =1 - a1
    amp = np.exp(amp)
    #a1 = np.exp(a1)
    #a2 = np.exp(a2)
    l = [ll(amp[i],a1[i],a2[i],sigma,yobs) for i in range(len(a1))]
    l = np.array(l)
    
    plt.figure()
    plt.scatter(a1,l,s=2)
    plt.xlabel('alpha1')
    plt.yscale('log')
    plt.grid(True)
    
    plt.figure()
    plt.scatter(a2,l,s=2)
    plt.xlabel('alpha2')
    plt.yscale('log')
    plt.grid(True)
    
    plt.figure()
    plt.scatter(amp,l,s=2)
    plt.xlabel('Amplitude')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    
    idx = np.argsort(l)
    a1 = a1[idx]
    a2 = a2[idx]
    amp = amp[idx]
    return amp,a1,a2

Out = sampler(10000,1e-19,Ymeas)


#with pm.Model() as model:
#    # priors on the covariance function hyperparameters
#    #lE = pm.Uniform('lE', 0, 1e10)
#    #lT = pm.Uniform('lT', 0, 1e10)
#    #lP = pm.Uniform('lP', 0, 1e10)
#    log_alpha1 = pm.Uniform('log_alpha1', 15, 40)
#    lalpha1 = pm.Deterministic('scale_alpha1', tt.exp(log_alpha1))
#    log_alpha2 = pm.Uniform('log_alpha2', 15, 40)
#    lalpha2 = pm.Deterministic('scale_alpa2', tt.exp(log_alpha2))
#
#    sigma = 1e-17
#    mua = flux(lalpha1,lalpha2)    
#    Iso = pm.Normal('Likelihood', mu = mua, sd = sigma, observed = Ymeas) 
#    
#    #Joint_Determ(Isotopes,Iso)
#    
#    trace = pm.sample(1000,tune = 2000, cores = 4, chains = 2,nuts_kwargs=dict(target_accept=0.99))#, init = 'ADVI')
#    pm.traceplot(trace)#,lines = plot_lines)
#    pm.energyplot(trace)
#    