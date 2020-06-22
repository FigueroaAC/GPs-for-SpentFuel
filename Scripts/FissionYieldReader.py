#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 22:57:14 2020

@author: AFigueroa
"""

import numpy as np
    

# In[]:
def ZZAAA_2_human(ZZAAA):
     Element_list = ['Zero','H',                                                                               'He',
                       'Li','Be',                                                     'B','C','N','O','F','Ne',
                       'Na','Mg',                                                  'Al','Si','P','S','Cl','Ar',
                       'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
                       'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
                       'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
                       'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm']
     A = ZZAAA[-3:]
     Z = int(ZZAAA[:-3])
     name = Element_list[Z] + A
     return name
     
# In[]:
file_name = 'cm-245n'
with open('/Users/AFigueroa/Desktop/GIT/{}.out'.format(file_name),'r') as f:
    Lines = f.readlines()
    Output = [ line[:66] for line in Lines]
    st = 22
    end = -5
    Output = Output[st:end]
    Output = [out.lstrip().rstrip().split() for out in Output]
    #OutpuOutput = [c for r in Output for c in r]
    Nenergies = int(Output[0][2])
    Dict = {}
    Dict['Ind'] = {}
    Dict['Cum'] = {}
    j = 0
    for i in range(Nenergies+1):
        Energy = Output[j][0]
        print(Energy)
        Dict['Ind'][Energy] = {}
        try:
            A = int(Output[j][4])
        except ValueError:
            j+=1
            A = int(Output[j][4])
            print(A)
        Nrows = A//6
        Data = Output[j+1:j+1+Nrows+1]
        Data = [c for r in Data for c in r]
        Dict['Ind'][Energy]['Nuclides-ZA'] = [str(int(float(d))) for d in Data[::4][:-2]]
        try:
            cutoff = Dict['Ind'][Energy]['Nuclides-ZA'].index('0')
            Dict['Ind'][Energy]['Nuclides-ZA'] = Dict['Ind'][Energy]['Nuclides-ZA'][:cutoff]
        except ValueError:
            pass 
        Dict['Ind'][Energy]['Nuclides-Hu'] = [ZZAAA_2_human(d) for d in \
            Dict['Ind'][Energy]['Nuclides-ZA']]
        Dict['Ind'][Energy]['IsoState'] = np.array(Data[1::4],dtype='float64')
        Dict['Ind'][Energy]['Yield'] = np.array(Data[2::4],dtype='float64')
        Dict['Ind'][Energy]['sigma'] = np.array(Data[3::4],dtype='float64')
        j += Nrows+1
    j += 2
    print('\n')
    for i in range(Nenergies+1):
        Energy = Output[j][0]
        print(Energy)
        Dict['Cum'][Energy] = {}
        try:
            A = int(Output[j][4])
        except ValueError:
            j+=1
            A = int(Output[j][4])
        Nrows = A//6
        Data = Output[j+1:j+1+Nrows+1]
        Data = [c for r in Data for c in r]
        Dict['Cum'][Energy]['Nuclides-ZA'] = [str(int(float(d))) for d in Data[::4][:-2]]
        try:
            cutoff = Dict['Cum'][Energy]['Nuclides-ZA'].index('0')
            Dict['Cum'][Energy]['Nuclides-ZA'] = Dict['Ind'][Energy]['Nuclides-ZA'][:cutoff]
        except ValueError:
            pass
        Dict['Cum'][Energy]['Nuclides-Hu'] = [ZZAAA_2_human(d) for d in \
        Dict['Cum'][Energy]['Nuclides-ZA']]
        Dict['Cum'][Energy]['IsoState'] = np.array(Data[1::4],dtype='float64')
        Dict['Cum'][Energy]['Yield'] = np.array(Data[2::4],dtype='float64')
        Dict['Cum'][Energy]['sigma'] = np.array(Data[3::4],dtype='float64')
        j += Nrows+1
        
np.save('/Users/AFigueroa/Desktop/GIT/{}-fissionyields.npy'.format(file_name),Dict)