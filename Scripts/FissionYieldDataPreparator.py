#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:32:58 2020

@author: AFigueroa
"""
import numpy as np
import matplotlib.pyplot as plt
# In[]
Data = np.load('/Users/AFigueroa/Desktop/GIT/FissionYields/u-235n-fissionyields.npy',allow_pickle=True).item()

#Ind = Data['Cum']
#Energies = [energy for energy in Ind]
#A = [int(nuc) for nuc in Ind[Energies[0]]['Nuclides-ZA']]
#B = [int(nuc) for nuc in Ind[Energies[-1]]['Nuclides-ZA']]
#
#idxA  =np.where(np.logical_and(Ind[Energies[0]]['Yield'][:len(A)] > 0.01,
#                               Ind[Energies[0]]['IsoState'][:len(A)]==0))[0]
#idxB  =np.where(np.logical_and(Ind[Energies[-1]]['Yield'][:len(B)] > 0.01,
#                               Ind[Energies[-1]]['IsoState'][:len(B)]==0))[0]
#YieldA = Ind[Energies[0]]['Yield'][:len(A)][idxA]
#A = np.array(A)[idxA]
#YieldB = Ind[Energies[-1]]['Yield'][:len(B)][idxB]
#B = np.array(B)[idxB]
#
#plt.figure()
#plt.scatter(A,YieldA)
#plt.scatter(B,YieldB)
#
#idx = np.where(YieldB>0.01)[0]
def data_preparer(Data,E):
    Cum = Data['Cum']
    Energies = E#[energy for energy in Cum]
    Y = []
    As = []
    for energy in Energies:
        A = [int(nuc) for nuc in Cum[energy]['Nuclides-ZA']]
        idxA = np.where(np.logical_and(Cum[energy]['Yield'][:len(A)] > 0.01,
                               Cum[energy]['IsoState'][:len(A)]==0))[0]
        YieldA = Cum[energy]['Yield'][:len(A)][idxA]
        A = np.array(A)[idxA]
        As.append(A)
        Y.append(YieldA)
    Y = np.array([Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))])
    E = np.array([float(Energies[i]) for i in range(len(Energies)) for j in range(len(As[i]))])
    As = np.array([As[i][j] for i in range(len(As)) for j in range(len(As[i]))])
    X = np.vstack([E,As]).T
    return X,Y

E = [e for e in Data['Cum']][:2]+[e for e in Data['Cum']][2::2]
X,Y = data_preparer(Data,E)
        
np.save('/Users/AFigueroa/Desktop/GIT/FissionYields/pu239curatedX.npy',X)
np.save('/Users/AFigueroa/Desktop/GIT/FissionYields/pu239curatedY.npy',Y)
        

    
