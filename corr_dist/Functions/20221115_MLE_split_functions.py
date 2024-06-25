# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:15:11 2022

@author: Shane Mannion
Code for various ideass used to try and fit multiple distributions to one graph

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import zeta
from scipy.special import factorial
from scipy.stats import poisson
from scipy.stats import kstest
import os
os.chdir('C:\\Users\\shane\\Desktop\\CRT\\IndividualProject\\Scripts and Data')  # Provide the new path here
from Functions.MLE_functions import (AIC, BIC, CCDF, CDF, exp_dist, logn, MLE, normal,
                                     PDF, poisson_dist, poisson_large_k, powerlaw,
                                     stretched_exp, trunc_powerlaw, weibull
                                     )
"""
______________________________________________________________________________
First, using KS test (incorrectly to do so).
"""


def find_split(result, X, N, P, Y):
    """
    Finds the value at which the distribution becomes a bad fit via KS test.
    Think this is different to how I originally used it. Here I am using CDF 
    functions fir the distribution (This is proper use) that I defined myself.
    This does not perform well though. The method that performed well was using
    2sample_kstest (from scipy, wrong method name here) with (X[0:i], Y[0:i])
    Parameters
    ----------
    X : array-like
        degree list
    P : array-like
        empirical CCDF of the degree list
    Y : array-like
        initial CCDF fit

    Returns
    -------
    int
        the value at which the initial distribution 
        begins to be a bad fit to the data
        
    """
    for i in range(1, max(X)):
        _, p_val = kstest(X[0:i], CDF, args=(result[0], N, P))
        if p_val < 0.05:
            break
    if i - 10 > 0:    
        return i - 10
    else:
        return i

def two_fits(X, k_star, initial_result):
    """
    Using kstar as found above, 
    Parameters
    ----------
    X : array-like
        degree list
    k_star : int
        value at which initial distribution becomes a poor fit
    initial_result : list
        output from MLE function

    Returns
    -------
    new_result : list
        parameters for two distributions fit to different portions of the dataset

    """
    result_1 = MLE(X, initial_result[0][0])
    result_2 = MLE(X, k_star)
    print(k_star)
    new_result = [result_1, result_2]
    return new_result


"""
______________________________________________________________________________
Next was Davids idea to fit to one distribution, and then split at kmax-1, then
fit to both. Get a loglikelihood for each and add together. Input this into AIC 
formula to get AIC that compares one to two (since it is ok once n_datapoints
is equal for both models). Repeat this for descending kstar (value to split at)
values. Choose the best distribution. Two methods for doing this outlined
below.
"""

def split_dist_m1(X, result):
    """
    Takes a degree sequence X as well as the result of applying MLE to it. Comparing AIC of original model to AIC of using two
    models (uses the original model for k_min <= k < kstar and the other for k >= kstar). 
    Returns the combination of models with lowest AIC in the same format as MLE result and a dictionary of {kmin: AIC}
    """
    split_results = {} # dictionary to save our results
    split_results['none'] = AIC(result[2][1],len(X[X >= result[0]]),len(result[2][0]) + 1) # aic of just one dist
    kmin = result[0]
    #so we have saved the AIC for original fit. Now we look at kstars
    kstars = np.unique(X[X >= np.percentile(X, 90)][::-1]) # split it at every value above the 90th percentile

    for i in kstars:
        XL = np.array([j for j in X if j >= kmin and j <= i]) 
        delta = (X[X < kmin].size/X.size)
        inf = np.arange(np.amax(X)) #Creates a sequence of numbers for infinite sums    
        sum_log = np.sum(np.log(XL))
        if result[1] == 'Powerlaw':
            lnl = -1*powerlaw(result[2][0], XL, sum_log, delta, kmin)
        if result[1] == 'Exponential':
            lnl = -1*exp_dist(result[2][0], XL, delta, kmin)
        if result[1] == 'Weibull':
            lnl = -1*weibull(result[2][0], XL, inf, sum_log, delta, kmin)
        if result[1] == 'Normal':
            lnl = -1*normal(result[2][0], XL, inf)
        if result[1] == 'Stretched_Exp':
            lnl = -1*stretched_exp(result[2][0], XL, inf, kmin)
        if result[1] == 'Trunc_PL':
            lnl = -1*trunc_powerlaw(result[2][0], XL, inf, delta, kmin)
        if result[1] == 'Lognormal':
            lnl = -1*logn(result[2][0], inf, sum_log, kmin)
        XU = np.array(X[X >= i])    
        rU = MLE(XU, i, 2)
        tot_lnl = lnl + rU[2][1]
        nparams = len(result[2][0]) + len(rU[2][0] + 2)
        split_results[i] = AIC(tot_lnl, len(X[X >= kmin]), nparams)
    opt_split = list(split_results.keys())[np.argmin(list(split_results.values()))]   
    if opt_split == 'none':
        opt_split = 1
    XU = np.array(X[X >= opt_split])
    rU = MLE(XU, opt_split, 2)
    LU = [result, rU]
    return LU, split_results

def split_dist_m2(X, result):
    """
    Takes a degree sequence X as well as the result of applying MLE to it. Comparing AIC of original model to AIC of using two
    models (one fitted to k < kstar and the other for k >= kstar). Returns the model with lowest AIC.
    Returns the combination of models with lowest AIC in the same format as MLE result and a dictionary of {kmin: AIC}
    """
    split_results = {} # dictionary to save our results
    split_results['none'] = AIC(result[2][1],len(X[X >= result[0]]),len(result[2][0]) + 1) # aic of just one dist
    kmin = result[0]
    #so we have saved the AIC for original fit. Now we look at kstars
    kstars = np.unique(X[X >= np.percentile(X, 90)][::-1]) # split it at every value above the 90th percentile
    for i in kstars:
        XU = np.array(X[X >= i])
        XL = np.array([j for j in X if j >= kmin and j <= i])    
        rL = MLE(XL, kmin, 2)
        rU = MLE(XU, i, 2)
        tot_lnl = rL[2][1] + rU[2][1]
        nparams = len(rL[2][0]) + len(rU[2][0] + 2)
        split_results[i] = AIC(tot_lnl, len(X[X >= kmin]), nparams)
    opt_split = list(split_results.keys())[np.argmin(list(split_results.values()))]    
    XU = np.array(X[X >= opt_split])
    XL = np.array([j for j in X if j >= kmin and j <= opt_split])   
    rL = MLE(XL, kmin, 2)
    rU = MLE(XU, opt_split, 2)
    LU = [rL, rU]
    return LU, split_results
    
def split_dist_all_k(X, result):
    """
    Takes a degree sequence X as well as the result of applying MLE to it. Comparing AIC of original model to AIC of using two
    models (uses the original model for k_min <= k < kstar and the other for k >= kstar). 
    This version checks all kmin values, not just the top 10%
    This version is likely redundant, percentile could be an input.
    Returns the combination of models with lowest AIC in the same format as MLE result and a dictionary of {kmin: AIC}    
    """
    split_results = {} # dictionary to save our results
    split_results['none'] = AIC(result[2][1],len(X[X >= result[0]]),len(result[2][0]) + 1) # aic of just one dist
    kmin = result[0]
    #so we have saved the AIC for original fit. Now we look at kstars
    kstars = np.unique(X) # split it at every value above the 90th percentile

    for i in kstars:
        XL = np.array([j for j in X if j >= kmin and j <= i]) 
        delta = (X[X < kmin].size/X.size)
        inf = np.arange(np.amax(X)) #Creates a sequence of numbers for infinite sums    
        sum_log = np.sum(np.log(XL))
        if result[1] == 'Powerlaw':
            lnl = -1*powerlaw(result[2][0], XL, sum_log, delta, kmin)
        if result[1] == 'Exponential':
            lnl = -1*exp_dist(result[2][0], XL, delta, kmin)
        if result[1] == 'Weibull':
            lnl = -1*weibull(result[2][0], XL, inf, sum_log, delta, kmin)
        if result[1] == 'Normal':
            lnl = -1*normal(result[2][0], XL, inf)
        if result[1] == 'Stretched_Exp':
            lnl = -1*stretched_exp(result[2][0], XL, inf, kmin)
        if result[1] == 'Trunc_PL':
            lnl = -1*trunc_powerlaw(result[2][0], XL, inf, delta, kmin)
        if result[1] == 'Lognormal':
            lnl = -1*logn(result[2][0], inf, sum_log, kmin)
        XU = np.array(X[X >= i])    
        rU = MLE(XU, i, 2)
        tot_lnl = lnl + rU[2][1]
        nparams = len(result[2][0]) + len(rU[2][0] + 2)
        split_results[i] = AIC(tot_lnl, len(X[X >= kmin]), nparams)
    opt_split = list(split_results.keys())[np.argmin(list(split_results.values()))]    
    XU = np.array(X[X >= opt_split])
    rU = MLE(XU, opt_split, 2)
    LU = [result, rU]
    return LU, split_results
        