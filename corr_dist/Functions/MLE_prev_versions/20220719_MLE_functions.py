# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:43:36 2022

@author: shane
"""
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from scipy.special import zeta
from scipy.special import factorial
from scipy.stats import poisson

def freqTable(G):
    
    """
    Parameters
    ----------
    G : networkx.graph OR degree list
        Graph to get freq table of

    Returns
    ----------
    np.array(degree_list): np.ndarray
        list of degrees
    unique_deg: np.ndarray
        list of unique degrees
    table: dict
        freq table of degrees

    """
    if type(G) == nx.classes.graph.Graph:
        degree_dict = dict(G.degree())
        degree_list = list(degree_dict.values())
    else:
        degree_list = G
    degree_list.sort()
    unique_deg = []
    
    table = {}
    for n in degree_list:   
        if n in table:
            table[n] += 1
        else:
            table[n] = 1    
            
    for n in degree_list:
        if n not in unique_deg:
            unique_deg += [n]
    return np.array(degree_list), np.array(unique_deg), table

def pdf_cdf(table: dict):
    
    """
    Parameters
    ----------
    table : dict
        Frequency table of degrees

    Returns
    -------
    Y_norm : np.ndarray
        Probabilities of degrees
    CY : np.ndarray
        Cumulative prob of degrees
    """

    Y = np.array(list(table.values()))
    Y_norm = Y/sum(Y)
    CY = np.cumsum(Y)/sum(Y)
    return Y_norm, CY


def empirical(result, X_list):
    params = result[2][0]
    k_min = result[0]
    N,f = np.unique(X_list, return_counts=True)
    p = f/X_list.size
    cumul = np.cumsum(f[::-1])[::-1]
    P = cumul/X_list.size
    Input = np.arange(result[0],np.amax(X_list)+1)
    return params, N, P, Input

"""
CDF functions: get CCDF for a degree distribution
    Parameters
    ----------
    params : np.ndarray 
        parameters determined by MLE function below
    Input : np.ndarray
        degree values above k_min
    P : np.ndarray
        pobabilities of each degree
    k_min : int
        value from which distribution is fitted
    inf: int
        large integer based on max degree for approximating infinite sums    

    Returns
    -------
    y : np.ndarray
        array of complementary cumulative probabilities
"""



def pl_cdf(params, Input, P, k_min):
    y_pl = (P[k_min - 1])*zeta(params[0], Input)/zeta(params[0], k_min)
    return y_pl

def exp_cdf(params, Input, P, k_min):
    C = P[k_min]#DOUBLE CHECK THIS
    y_exp = C*np.exp((-1/params[0])*(Input-k_min))
    return y_exp

def wb_cdf(params, Input, P, inf):
    sum1 = np.array([np.sum((((j+inf)/params[0])**(params[1]-1))*np.exp(-(((j+inf)/params[0])**params[1]))) for j in Input])
    inf_sum = np.sum((((inf + 2)/params[0])**(params[1]-1))*np.exp(-1*((inf + 2)/params[0])**params[1]))
    y_weibull = (P[1])*sum1/inf_sum
    return y_weibull

def logn_cdf(params, Input, P, k_min, inf):
    sum1 = np.array([np.sum( (1.0/(j+inf))*np.exp(-((np.log(j+inf)-params[0])**2)/(2*(params[1]**2)))) for j in Input])
    inf_sum = np.sum( (1.0/(inf+k_min)) * np.exp(-((np.log(inf+k_min)-params[0])**2)/(2*params[1]**2) ) )
    y_lognormal = (P[k_min-1])*sum1/(inf_sum)
    return y_lognormal

def AIC(LnL: float, N:int, params:int = 1):
    """
    AIC with correction for large sample sizes
    Parameters
    ----------
    LnL : float
        log_likelihood value
    params : int, optional
        Number of parameters in the distribution

    Returns
    -------
    float
        AIC for a given log-likelihood and distribution
    """
    AICc = -2*LnL + 2*params + ((2*params*(params + 1)))/(N - params - 1)
    return AICc

def BIC(LnL: float, N:int, params:int = 1):
    """
    Parameters
    ----------
    LnL : float
        log_likelihood value
    N : int
        number of nodes with degree > k_min        
    params : int, optional
        Number of parameters in the distribution

    Returns
    -------
    float
        BIC for a given log-likelihood and distribution
    """
    return params * np.log(N) - 2*LnL

"""
Log-Likelihood functions: Return negative log-likelihoods for the distributions:
    power-law
    exponential
    weibull
    truncated power-law
    log-normal
    poisson
Parameters
----------
params: np.ndarray
    array of distribution parameters
x: np.ndarray
    array of network degrees above k_min
delta: float
    fraction of degrees below k_min. default = 0
k_min: int
    value from which the distribution is fitted
sum_log: float
    sum of log values of x above    
inf: int
    large value to sum to for approximations of infinte sums
lam: float
    lambda parameter for poisson distribution
    
Returns
-------
NegLnL: float
    Negative of log-likelihood value for given distribution
    
"""
def powerlaw(params:np.ndarray, x:np.ndarray, sum_log, delta:float = 0, k_min:int = 1):
    NegLnL =  x.size*np.log(zeta(params[0], k_min)) + params[0]*(sum_log) 
    return NegLnL

def exp_dist(params:np.ndarray, x:np.ndarray, delta:float=0, k_min:int=1):
    NegLnL = -1 * x.size*(np.log(1-np.exp(-1/params[0]))) + (1/params[0])*(x.sum() - x.size*k_min)
    return NegLnL

def weibull(params, x:np.ndarray, inf, sum_log, delta:float=0, k_min:int=1):
    inf_sum = np.sum((((inf + k_min)/params[0])**(params[1]-1))*np.exp(-1*((inf + k_min)/params[0])**params[1]))
    LnL = -x.size * np.log(inf_sum) - x.size * (params[1] - 1) * np.log(params[0])\
        + (params[1] - 1) * sum_log - np.sum((x/params[0])**params[1])
    NegLnL = -1 * LnL    
    return NegLnL

def trunc_powerlaw(params, x:np.ndarray, inf, delta:float, k_min:int=1):
    inf_sum = np.sum((inf + k_min)**(-1*params[1]) * np.exp(-1*inf/params[0]))
    LnL = x.size * np.log(1 - delta) + x.size * k_min/params[0] - x.size*np.log(inf_sum)\
        - (params[1]*np.log(x) + x/params[0]).sum()
    NegLnL = -1*LnL
    return NegLnL

def logn(params, x, inf, sum_log, k_min=1):
    inf_sum = np.sum( (1.0/(inf+k_min)) * np.exp(-((np.log(inf+k_min)-params[0])**2)/(2*params[1]**2) ) )
    NegLnL = -1*( - x.size*np.log(inf_sum) - sum_log - np.sum( ((np.log(x)-params[0])**2)/(2*params[1]**2) ) )
    return NegLnL

def poisson_dist(lam, x:np.ndarray, delta:float, k_min:int=1):
    m = np.arange(k_min)
    LnL = x.size * np.log(1 - delta) - np.log(1 - np.exp(-1*lam) * np.sum((lam**m)/factorial(m)))\
        - x.size * lam + np.log(lam) * x.sum() - np.sum(np.log(factorial(x)))
    NegLnL = -1*LnL
    return NegLnL

def poisson_large_k(lam, x:np.ndarray):
    d1 = poisson.pmf(x, lam)
    d1 = d1[np.nonzero(d1)]
    NegLnL = -1 * np.sum(np.log(d1))    
    return NegLnL

def MLE(X:np.ndarray, k_min = 2):
    """
    Maximises the log-likelihood for each of the above distributions and chooses the best
    by minimising the AIC and BIC.
    
    Stopping Criteria: starts with k_min=2 and increases by 1 each time. Stops when the 
    same distribution is chosen for 5 consecutive k_min values, returns values obtained at 
    the smallest of these 5 k_min values.
    
    Parameters
    ----------
    X : np.ndarray
        array of degrees of network including below k_min.
    k_min : TYPE, optional
        value from which to fit the distribution

    Returns
    -------
    Final_dist : list
        [k_min, fitted distribution name (e.g. powerlaw), array of parameters, 
         negative log-likelihood]
    Delta : float
        fraction of nodes below final chosen k_min value

    """
    votes = [100,10,100,10,100]
    Results = {}
    Results['Powerlaw'] = {}
    Results['Exponential'] = {}
    Results['Weibull'] = {}
    Results['Trunc_PL'] = {}
    Results['Lognormal'] = {}
    Results['Poisson'] = {}
    
    while np.std(votes[-3:]) >= 0.1:   
        x = X[X >= k_min]
        delta = (X[X < k_min].size/X.size)
        k_mean = x.mean()    
        try:
            inf = np.arange(np.amax(x))# + 1000)    
        except ValueError:  #raised if x is empty.
            inf = 1000
        sum_log = np.sum(np.log(x))

        opt_pl = minimize(powerlaw, (2), (x, sum_log, delta, k_min), method = 'SLSQP', bounds = [(0.5, 4)])
        Results['Powerlaw'][k_min] = [opt_pl['x'], -1*opt_pl['fun']]
                
        opt_exp = minimize(exp_dist, (k_mean), (x, delta, k_min), method = 'SLSQP', bounds = ((0.5,k_mean + 20),))
        Results['Exponential'][k_min] = [opt_exp['x'], -1*opt_exp['fun']]
        
        opt_wb = minimize(weibull, (k_mean,1),(x, inf, sum_log, delta, k_min), method = 'SLSQP', bounds=((0.5, None),(0.05, 4),))
        Results['Weibull'][k_min] = [opt_wb['x'], - 1*opt_wb['fun']]
        
        opt_tpl = minimize(trunc_powerlaw,(k_mean,1),(x, inf, delta, k_min), method = 'SLSQP', bounds=((0.5, k_mean + 20),(0.5,4),))
        Results['Trunc_PL'][k_min] = [opt_tpl['x'], -1*opt_tpl['fun']]
        try: #prevents valueerror when value goes out of bounds given in function
            opt_logn = minimize(logn, (np.log(k_mean), np.log(x).std()), (x, inf, sum_log, k_min), method='TNC',bounds=[(0.,np.log(k_mean)+10),(0.01,np.log(x.std())+10)])
            Results['Lognormal'][k_min] = [opt_logn['x'], -1*opt_logn['fun']]
        except ValueError:
            Results['Lognormal'][k_min] = [[0,0], 0]
        try:
            poisson_max = np.amax(x)
        except ValueError:
            poisson_max = 1
        if poisson_max > 170: #different method used when k_max is large, due to infinity from factorial
            opt_p = minimize(poisson_large_k, x.mean(), (x), method='SLSQP')
        else:
            opt_p = minimize(poisson_dist, x.mean(), (x, delta, k_min), method='SLSQP', bounds = ((0.5, None),))
        Results['Poisson'][k_min] = [opt_p['x'], -1*opt_p['fun']]
        Distributions = list(Results.keys())    
        AICs = []
        BICs = []
        
        for i in Results.keys():
            if AIC(Results[i][k_min][1], x.size, len(Results[i][k_min][0])) == float("-inf"):
                AICs.append(float("inf"))
            else:
                AICs.append(AIC(Results[i][k_min][1], x.size, len(Results[i][k_min][0])))
                
            if BIC(Results[i][k_min][1], x.size, len(Results[i][k_min][0])) == float("-inf"):    
                BICs.append(float("inf"))
            else:
                BICs.append(BIC(Results[i][k_min][1], x.size, len(Results[i][k_min][0])))
                                
        if Distributions[np.argmin(AICs)] == Distributions[np.argmin(BICs)]:
            votes.append(np.argmin(BICs).astype(np.int32))
            
        else:
            votes += [10] #if BIC and AIC do not choose the same distribution, move to next k_min
        k_min += 1
    Final_dist = [k_min-3, Distributions[np.argmin(AICs)],Results[Distributions[np.argmin(AICs)]][k_min-3]]
    Delta = (X[X < (k_min-3)]).size/X.size
    return Final_dist, Delta
