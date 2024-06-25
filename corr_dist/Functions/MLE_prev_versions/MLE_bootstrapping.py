# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:42:36 2022

@author: shane
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import zeta
from scipy.special import factorial
from scipy.stats import poisson
from Functions.MLE_functions_edit import powerlaw, exp_dist, weibull, stretched_exp, trunc_powerlaw, logn, poisson_dist, poisson_large_k

def opt_single_dist(X, result, k_min):
   #k_min = result[0]
    x = X[X >= k_min]
    delta = (X[X < k_min].size/X.size)
    k_mean = x.mean()    
    try:
        inf = np.arange(np.amax(x))# + 1000)    
    except ValueError:  #raised if x is empty.
        inf = 1000
    sum_log = np.sum(np.log(x))
    if result[1] == 'Powerlaw':
        opt = minimize(powerlaw, (2), (x, sum_log, delta, k_min), method = 'SLSQP', bounds = [(0.5, 4)])
    if result[1] == 'Exponential':
        opt = minimize(exp_dist, (k_mean), (x, delta, k_min), method = 'SLSQP', bounds = ((0.5,k_mean + 20),))
    if result[1] == 'Weibull':
        opt = minimize(weibull, (k_mean,1),(x, inf, sum_log, delta, k_min), method = 'SLSQP', bounds=((0.05, None),(0.05, 4),))    
    if result[1] == 'Stretched_Exp':
        opt = minimize(stretched_exp,(k_mean,1),(x, inf, k_min), method='SLSQP',bounds=[(0.5,None),(0.05,4.)])
    if result[1] == 'Trunc_PL':
        opt = minimize(trunc_powerlaw,(k_mean,1),(x, inf, delta, k_min), method = 'SLSQP', bounds=((0.5, k_mean + 20),(0.5,4),))
    if result[1] == 'Lognormal':
        try: #prevents valueerror when value goes out of bounds given in function
            opt = minimize(logn, (np.log(k_mean), np.log(x).std()), (x, inf, sum_log, k_min), method='TNC',bounds=[(0.,np.log(k_mean)+10),(0.01,np.log(x.std())+10)])
        except ValueError:
            return [0,0]
    if result[1] == 'Poisson':
        try:
            poisson_max = np.amax(x)
        except ValueError:
            poisson_max = 1
        if poisson_max > 170: #different method used when k_max is large, due to infinity from factorial
            opt = minimize(poisson_large_k, x.mean(), (x), method='SLSQP')
        else:
            opt = minimize(poisson_dist, x.mean(), (x, delta, k_min), method='SLSQP', bounds = ((0.5, None),))
    return opt['x']



def summary_stats(Name, result, params):
    means = []
    devs = []
    perc1 = []
    perc2 = []
    for i in params:
        means.append(np.mean(i))
        devs.append(np.std(i))
        perc1.append(np.percentile(i, 2.5))
        perc2.append(np.percentile(i, 97.5))
    
    row = [Name, result[1], result[0], result[2][0], means[0], devs[0], perc1[0], perc2[0]]
    if len(result[2]) == 2:
        row.extend(result[2][1], means[1], devs[1], perc1[1], perc2[1]) 
    return row

def bootstrap(G_list, result): # NEW VERSION
    """
    Bootstraps a sample of data and using the established k_min and distribution
    Obtains 1,000 values for the parameter(s) of the distribution. Ignores cases where
    the bootstrapped sample is found to be a different distribution

    Parameters
    ----------
    G_list : list or np.array
        Degree list
    G_result : list
        Output from the MLE function

    Returns
    -------
    param1 : list
        list of parameter values obtained from bootstrapped samples
    param2 : list
        list of parameter values obtained from bootstrapped samples

    """
    params1 = []
    params2 = []
    while len(params1) < 1000:
        sample = np.array(np.random.choice(G_list, len(G_list), replace=True))
        opt = opt_single_dist(sample, result)
        if len(opt) == 1:
            if opt[0].isna == False:
                params1.append(opt[0])
		
        if len(opt) == 2:
            if opt[0].isna == False & opt[1].isna == False:
                if [0] != opt[1]:
                    params1.append(opt[0])
                    params2.append(opt[1])
	
		
    if len(params2) == 0: # graph has one distribution
    	parameters = [params1]
    else:
        parameters = [params1, params2]
	
    return parameters