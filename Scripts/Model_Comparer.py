#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:06:36 2020

@author: AFigueroa
"""

import numpy as np

from scipy import interpolate as intp
from scipy import linalg
from scipy.spatial import distance
from tqdm import tqdm
# In[]:    
def GPR_MODEL(Params,Lambda_Inv,Xtrain,Ytrain,alpha_,Xtest):
    """ Implementation of a GP mean (mu) predictor
        
        mu = Ktrans \dot alpha_
        
        inputs:
            * Params, Lambda_Inv: Trained kernel parameters.
            * alpha_: Precomputed product of (Kself+\sigma*I)^{-1} \dot Ytrain.
            * Xtrain, Ytrain: Training datasets.
            * Xtest: Test point.
        outputs:
            * Y_pred: Predicted mean of the GP at the queried point.
    """
    Constant = Params[0]
    sigma = Params[-1]
    Distance = Xtest - Xtrain
    Distance_sq = np.diag(np.dot(np.dot(Distance,Lambda_Inv),Distance.T))
    Ktrans = (Constant * np.exp(-0.5*Distance_sq)) + (sigma*np.ones(len(Xtrain)))
    Y_pred = np.dot(Ktrans,alpha_)
    Y_pred = (Y_pred*np.std(Ytrain))+np.mean(Ytrain)
    return Y_pred  

def alpha_calculator(Params,X,Y):
    """
    Precalculation of the alpha_ and Kinv constants used for the computation
    of the posterior GP mean and variance respectively.
    
        alpha_ = (Kself+\sigma*I)^{-1} \dot Ytrain
        K_inv = (Kself+\sigma*I)^{-1}
        
        inputs:
            * Params: Trained kernel parameters.
            * X,Y: Training datasets.
        outputs:
            * alpha_: Vector used for the computation of the GP mean.
            * K_inv: matrix used for the computation of the GP variance.
            
    
    """
    Cs = Params[0]
    P =  Params[1:-1]
    alpha = Params[-1]
    LAMBDA = np.eye(len(X[0]))
    length_scales = (1/P)
    np.fill_diagonal(LAMBDA,length_scales)
    Xtrain = np.dot(X,LAMBDA)
    distSself = distance.cdist(Xtrain , Xtrain, metric='sqeuclidean').T
    # Calculate Self Correlation:
    KSelf = Cs * np.exp(-.5 * distSself)
    Y_norm = (Y - np.mean(Y))/np.std(Y)
    KSelf = KSelf + alpha*np.eye(len(KSelf))

    L_ = linalg.cholesky(KSelf,lower=True)
    L_inv = linalg.solve_triangular(L_.T,np.eye(L_.shape[0]))
    K_inv = L_inv.dot(L_inv.T)
    alpha_ = linalg.cho_solve((L_,True),Y_norm)
    return alpha_, K_inv

def GPR_MODEL_w_Std(Params,Lambda_Inv,Xtrain,Ytrain,alpha_,K_inv,Xtest):
     """ Implementation of a GP mean (mu) and variance (var) predictor
        
        mu = Ktrans \dot alpha_
        var = Ktranstrans - Ktrans.T \dot K_inv \dot Ktrans
        
        inputs:
            * Params, Lambda_Inv: Trained kernel parameters.
            * alpha_: Precomputed product of (Kself+\sigma*I)^{-1} \dot Ytrain.
            * K_inv: Precomputed inverse matrix of self covariance.
            * Xtrain, Ytrain: Training datasets.
            * Xtest: Test point.
        outputs:
            * Y_pred: Predicted mean of the GP at the queried point.
            * Var: Predicted variance of the GP at the queried point
    """
    Constant = Params[0]
    sigma = Params[-1]
    Distance = Xtest - Xtrain
    Distance_sq = np.diag(np.dot(np.dot(Distance,Lambda_Inv),Distance.T))
    Ktrans = (Constant * np.exp(-0.5*Distance_sq)) + (sigma*np.ones(len(Xtrain)))
    Ktranstrans = Constant + sigma

    Y_pred = np.dot(Ktrans,alpha_)
    Y_pred = (Y_pred*np.std(Ytrain))+np.mean(Ytrain)

    Var = np.expand_dims(Ktranstrans,-1) -np.einsum(
            "ij,ij->i", np.dot(np.expand_dims(Ktrans,0), K_inv), 
            np.expand_dims(Ktrans,0))
    Var = np.sqrt(np.abs(Var[0]))

    return Y_pred, Var


def gen_mask(option,data,testtype,spacing,mode):
    """Experimental Design Generator for grid sampling and testing:
        
        Three options are available:
            * Even: Even selection of points in the grid.
            * Odd: Odd selection of points in the grid.
            * Checkerboard: Checkerboard Pattern.
        Furthermore, for the spacing variable adds more variety as it allows
        to change the spacing for Even and Odd patterns. 
        
        inputs:
            * option: Type of design (Even, Odd or Checkboard).
            * data: The array upon which the design is desired.
            * testtype: The design itself leaves many points unnacounted for.
                        Here one can choose if the test set would be composed
                        from the rest of the points not in the design (all),
                        the borders of the space only (borders), or only points
                        lying inside the space (inner) for pure interpolation.
            *spacing: The spacing desired between points for the designed
                      training set.
            *mode : 
                -normal: Used for calculations
                -testing: Used for plotting.
        outputs:
            * Train, Test: flattened but ordered arrays used for training
                           testing the interpolation methods.
        
    """
    
    def rest_array(array,testtype):
        if testtype == 'all':
            Rest = np.ones((array.shape[0],array.shape[1]))
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    if array[i][j] == 1:
                        Rest[i][j] = 0
        if testtype == 'inner':
            Rest = np.zeros((array.shape[0],array.shape[1]))
            for i in range(1,array.shape[0]-1):
                for j in range(1,array.shape[1]-1):
                    if array[i][j] != 1:
                        Rest[i][j] = 1
        if testtype == 'borders':
            Rest = np.ones((array.shape[0],array.shape[1]))
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    if array[i][j] == 1:
                        Rest[i][j] = 0
            Rest[1:-1,1:-1] = 0
        return Rest
# =============================================================================
#   Types of grid
# =============================================================================
    even = np.zeros((data.shape[0],data.shape[1]))
    for i in range(0,data.shape[0],spacing):
        for j in range(0,data.shape[1],spacing):
            even[i][j] = 1
    odd= np.zeros((data.shape[0],data.shape[1]))
    for i in range(1,data.shape[0],spacing):
        for j in range(1,data.shape[1],spacing):
            odd[i][j] = 1
    Sum = even+odd       
    if option == 'Even':
        Rest = rest_array(even,testtype)
        Train = data[even>0]
        Test = data[Rest>0]
    if option == 'Odd':
        Rest = rest_array(odd,testtype)
        Train = data[odd>0]
        Test = data[Rest>0]
    if option == 'Checkerboard':
        Rest = rest_array(Sum,testtype)
        Train = data[Sum>0]
        Test = data[Rest>0]
    if mode == 'normal':    
        return Train, Test
    elif mode == 'testing':
        return even,odd,Sum,Rest
    
# =============================================================================
# Statistical Functions:
# =============================================================================
def Calc_SStot(ydata):
    Ymean = np.mean(ydata)
    Diff = np.array([(y-Ymean)**2 for y in ydata])
    SStot = np.sum(Diff)
    return SStot
def Calc_SSres(yfunc,ydata):
    Diff = np.array([(yfunc[i]-ydata[i])**2 for i in range(len(ydata))])
    SSres = np.sum(Diff)
    return SSres
    
def Rsquare(ypred,ytest):
    SStot = Calc_SStot(ytest)
    SSres = Calc_SSres(ypred,ytest)
    Rsquare = 1- (SSres/SStot)
    return Rsquare

def get_Error(Ypred,Ytrue):
    Error = [(Ypred[i] - Ytrue[i]) for i in range(len(Ytrue))] 
    MAE = sum(np.abs(Error))/len(Error)
    MSE = sum([(Error[i])**2 for i in range(len(Error))])/len(Error)
    RMSE = np.sqrt(MSE)
    rel_Error = Error/np.array(Ytrue)
    mean_rel_error = np.mean(rel_Error)*100
    max_Rel_Error = np.max(rel_Error)
    return Error, MAE, MSE, RMSE, mean_rel_error, max_Rel_Error
# In[]:

# =============================================================================
# Isotope:
# =============================================================================

def intp_comparison(isotope,option,size,Grid_Setting,Test_type,space_factor,
                    op_mode,X,Y,sorting_index,Kernel_type,Xsobol,Ysobol):
    
    """
    Comparison between Cubic Splines and GP models.
    
    inputs:
        *isotope: nuclide to use for comparison
        *option: 
            -Grid: for comparisons on models trained on a Grid
            -Random: for comparisons on models trained on a random selection
                     of the datasets.
            -Sobol: for comparisons on models trained on a Sobol sequence.
        *size: The number of samples to consider. Only considered when selecting
               the option 'Sobol' or 'Random'.
        *Grid_Setting: Selects the option on the design generator (Even, Odd or
                       Checkerboard).
        *Test_type: Selects the option on the design generator(All, inner or
                    borders).
        *space_factor: Sets the spacing on the design generator.
        *op_mode: Sets the operation mode of the design generator (normal,
                  testing).
        * X,Y, Xsobol, Ysobol: Training / Testing sets
        * sorting_index: Only used for X and Y on a grid. Reorder Ys points to
                         correctly match the points of the sampling grid.
        * Kernel_type: Used to select the kernel type used (Kernels-Mass, 
                       Kernels-Adens. etc)
    outputs:
        * Values_Cubic,Values_GPR: Contain diagnostics such as MAE, RMSE, MSE,
                                   Rsquared, plus mean predictive variance and
                                   fractions of predicted points within 
                                   predictive variance for GPR.
    """
    
    Temperature = np.array(list(set(X[:,0])))
    Burnup = np.array(list(set(X[:,1])))
    Burnup = np.sort(Burnup)
    
    Ys = np.array(Y[isotope])[sorting_index]
    Ys = Ys.reshape((len(Temperature),len(Burnup))).T 
    Ydata = np.array(Y[isotope])
  
    if option == 'Grid':
    
    # =========================================================================
    # Cubic
    # =========================================================================
        Tmesh, Bmesh = np.meshgrid(Temperature,Burnup)
        BtrainCubic,BtestCubic = gen_mask(
                Grid_Setting,Bmesh,Test_type,space_factor,op_mode)
        TtrainCubic,TtestCubic = gen_mask(
                Grid_Setting,Tmesh,Test_type,space_factor,op_mode)
        YtrainCubic,YtestCubic = gen_mask(
                Grid_Setting,Ys,Test_type,space_factor,op_mode)
    
    # =========================================================================
    # GPR
    # =========================================================================
        XtrainGPR = np.vstack((TtrainCubic,BtrainCubic)).T
        YtrainGPR = YtrainCubic
        XtestGPR = np.vstack((TtestCubic,BtestCubic)).T
        YtestGPR = YtestCubic
    
    elif option == 'random':
        Indexes = np.arange(len(X))
        IdxTrain = np.random.choice(Indexes,size = size)
        IdxTest = np.array([idx for idx in Indexes if idx not in IdxTrain])

        XtrainGPR = X[IdxTrain]
        XtestGPR = X[IdxTest]
        BtrainCubic = XtrainGPR[:,0]
        TtrainCubic = XtrainGPR[:,1]
        BtestCubic = XtestGPR[:,0]
        TtestCubic = XtestGPR[:,1]
        YtrainGPR = Ydata[IdxTrain]
        YtestGPR = Ydata[IdxTest]
        YtrainCubic = YtrainGPR
        YtestCubic = YtestGPR
    
    elif option == 'Sobol':
        XtrainGPR = Xsobol[:size]
        YtrainGPR = Ysobol[:size]
        XtestGPR = X
        YtestGPR = Y[isotope]
        BtrainCubic = XtrainGPR[:,1]
        TtrainCubic = XtrainGPR[:,0]
        BtestCubic = XtestGPR[:,1]
        TtestCubic = XtestGPR[:,0]
        YtrainCubic = YtrainGPR
        YtestCubic = YtestGPR
        

    # =========================================================================
    # Interpolators
    # =========================================================================
    CubicInt = intp.SmoothBivariateSpline(BtrainCubic,TtrainCubic,YtrainCubic)
    Yintp_Cubic = CubicInt.ev(BtestCubic,TtestCubic)
    Cubic_Errors = get_Error(
            Yintp_Cubic,YtestCubic)
    
    # =========================================================================
    # GPR:
    # =========================================================================

    try:
        # =====================================================================
        # Load Kernel Params:
        # =====================================================================
        
        Kernel = np.load('Path/'+Kernel_type+\
                         '/{}/{}.npy'.format(option,isotope),
                         allow_pickle=True).item()
        Params = Kernel['Params']
        Lambda_Inv = Kernel['LAMBDA']
        
        # =====================================================================
        # Precalculate alpha_ and K_inv:
        # =====================================================================
        alpha_,K_inv = alpha_calculator(Kernel['Params'],XtrainGPR,YtrainGPR)
        
        # =====================================================================
        # Get predictions:
        # =====================================================================
        
        GPR = [GPR_MODEL_w_Std(Params,Lambda_Inv,XtrainGPR,YtrainGPR,alpha_,
                               K_inv,x) for x in XtestGPR]
        GPR_pred = np.array(GPR)[:,0]
        GPR_std = np.array(GPR)[:,1]
        GPR_Errors = get_Error(GPR_pred,YtestGPR)
    except FileNotFoundError:
        return 404
     
        
    Mean_GPR_std = np.mean(GPR_std)
    Max_GPR_std = np.max(GPR_std)
    RsquareGPR = Rsquare(GPR_pred,YtestGPR)
    RsquareCubic = Rsquare(Yintp_Cubic,YtestCubic)
    One_sigma = [1 if \
                 GPR_pred[i]-GPR_std[i] < YtestGPR[i] < GPR_pred[i]+GPR_std[i] \
                 else 0 for i in range(len(GPR_pred))]
    Two_sigma = [1 if \
                 GPR_pred[i]-2*GPR_std[i] < YtestGPR[i] < GPR_pred[i]+2*GPR_std[i] \
                 else 0 for i in range(len(GPR_pred))]
    f_1sigma = (sum(One_sigma)/len(One_sigma))*100
    f_2sigma = (sum(Two_sigma)/len(Two_sigma))*100
    # =========================================================================
    # Print Summary
    # =========================================================================

    print('Mean Y = ',np.mean(Ys))
    print('MAE (Cubic | GPR) = ','{:.3e}'.format(Cubic_Errors[1]),\
          '|','{:.3e}'.format(GPR_Errors[1]))
    print('MSE (Cubic | GPR) = ','{:.3e}'.format(Cubic_Errors[2]),\
          '|','{:.3e}'.format(GPR_Errors[2]))
    print('RMSE (Cubic | GPR) = ','{:.3e}'.format(Cubic_Errors[3]),\
          '|','{:.3e}'.format(GPR_Errors[3]))
    print('Mean Rel. Error (%) (Cubic | GPR) = ','{:.3e}'.format(Cubic_Errors[4])\
          ,'|','{:.3e}'.format(GPR_Errors[4]))
    print('Max Rel. Error (%) (Cubic | GPR) = ','{:.3e}'.format(Cubic_Errors[5])\
          ,'|','{:.3e}'.format(GPR_Errors[5]))
    print('R^2 Coeff. of Determination (Cubic | GPR) = ',\
          '{:.3e}'.format(RsquareCubic),'|','{:.3e}'.format(RsquareGPR))

    Values_Cubic = [error for error in Cubic_Errors]+[RsquareCubic]
    Values_GPR = [error for error in GPR_Errors] + [RsquareGPR, 
                 Mean_GPR_std,Max_GPR_std,f_1sigma,f_2sigma]
    return Values_Cubic,Values_GPR

    
# In[]:
# =============================================================================
# Load Data for interpolator comparison
# =============================================================================

X = np.load('/X_Candu_Grid625.npy',
            allow_pickle=True)

Xsobol = np.load('/X_Candu_Sobol625.npy',
            allow_pickle=True)
# =========================================================================
# Use this to perform the interpolation on the atom density data set
# =========================================================================
    
#Y = np.load('/Y_Candu_Grid625Adens.npy',
#           allow_pickle=True).item()
    
    
# =========================================================================
# Use this to perform the interpolation on the total mass data set
# =========================================================================
Y = np.load('/YCandu_Output_Grid.npy',
               allow_pickle=True).item()
Ysobol = np.load('/YCandu_Output_Sobol.npy',
                allow_pickle=True).item()

# In[]:
# =============================================================================
# Interpolation Options:
# =============================================================================
Isotope_List = [isotope for isotope in Y]
Grid_Setting = 'Sobol'#'Even'|'Odd'|'Checkerboard'|'Sobol'
Test_type = 'all' #'all'|'inner'|'borders'
op_mode = 'normal' # '
space_factor = 2 # Grid Spacing
size = 625 # Used when option != Grid
option = 'Sobol'#'Grid'|'random'|'Sobol'
sorting_index = np.load('/sorting_indexes.npy',allow_pickle=True)

Kernel_type = 'Kernels-Mass'#| 'Kernels-Mass' | 'Kernels-Adens' | -> Use this one for atom density
# Kernels-Mass = Kernels for total mass as outputs, uses the entire data set to train the GPR parameters
Path = ''
# =============================================================================
# Perform the comparison
# =============================================================================

Output = [intp_comparison(isotope,option,size,Grid_Setting,Test_type,space_factor,
                          op_mode,X,Y,sorting_index,Kernel_type,Xsobol,\
                          Ysobol[isotope]) for isotope in tqdm(Isotope_List)]

Output_dict = {}
Summary_Cubic = ['Errors','MAE','MSE','RMSE','meanRelE','maxRelE','Rsquare']
Summary_GPR = ['Errors','MAE','MSE','RMSE','meanRelE','maxRelE','Rsquare',
               'meanPredStd','maxPredStd','%with1std','%with2std']
for i, isotope in tqdm(enumerate(Isotope_List)):
    try:
        Output_dict[isotope] = {}
        Output_dict[isotope]['Cubic'] = dict(zip(Summary_Cubic,Output[i][0]))
        Output_dict[isotope]['GPR'] = dict(zip(Summary_GPR,Output[i][1]))
    except TypeError:
        continue
np.save(Path+'/Comparison-Summary-{}-{}-{}-{}space-Dict.npy'.format(
        Kernel_type,Grid_Setting,Test_type,space_factor),Output_dict)


