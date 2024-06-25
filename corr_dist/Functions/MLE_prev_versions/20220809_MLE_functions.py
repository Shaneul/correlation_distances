# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:43:36 2022

@author: shane
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import zeta
from scipy.special import factorial
from scipy.stats import poisson
from scipy.stats import ks_2samp
def freqTable(G): #likely redundant
    
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

def degree_list(G):
    """
    Parameters
    ----------

    G : networkx.graph OR list


    Returns
    -------
    np.ndarray
        array of degrees

    """
    if type(G) == nx.classes.graph.Graph:
        degree_dict = dict(G.degree())
        degree_list = list(degree_dict.values())
    else:
        degree_list = G
    degree_list.sort()
    return np.array(degree_list)

def empirical(X_list):
    """
    Takes the MLE result and degree list and returns cumulative probability, unique degrees,
    sequence of integers from kmin to max degree
    Parameters
    ----------
    result : list
        result from MLE function
    X_list : list
        degree list

    Returns
    -------
    params : np.ndarray
        parameters of the distribution
    N : list
        unique degrees of the duistribution
    P : TYPE
        cumulative prob of the degrees
    Input : list
        sequence of integers kmin to max degree (Input to plot vs fitted curve)

    """
    N,f = np.unique(X_list, return_counts=True)
    cumul = np.cumsum(f[::-1])[::-1]
    P = cumul/X_list.size
    return N, P


def CCDF(result, Input, N, P):
    """
    Uses parameters from the distribution to get the fitted CCDF

    Parameters
    ----------
    result : list
        output from MLE function. [kmin, dist, params, delta]
    Input : list
        The degree list
    P : TYPE
        The empirical CCDF

    Returns
    -------
    
    
    """
    k_min = result[0]
    distribution = result[1]
    params = result[2]
    C_index = np.where(N == k_min)[0]
    C = P[C_index]
    try:
        inf = np.arange(np.amax(Input))
    except ValueError:
        inf = np.arange(1000)
    if distribution == 'Powerlaw':
        y = C*zeta(params[0], Input)/zeta(params[0], k_min)
    
    if distribution == 'Exponential':
        y = C*np.exp((-1/params[0])*(Input-k_min))
    
    if distribution == 'Weibull':
        sum1 = np.array([np.sum((((j+inf)/params[0])**(params[1]-1))*np.exp(-(((j+inf)/params[0])**params[1]))) for j in Input])
        inf_sum = np.sum((((inf + k_min)/params[0])**(params[1]-1))*np.exp(-1*((inf + k_min)/params[0])**params[1]))
        y = C*sum1/inf_sum 
    
    if distribution == 'Lognormal':
        sum1 = np.array([np.sum( (1.0/(j+inf))*np.exp(-((np.log(j+inf)-params[0])**2)/(2*(params[1]**2)))) for j in Input])
        inf_sum = np.sum( (1.0/(inf+k_min)) * np.exp(-((np.log(inf+k_min)-params[0])**2)/(2*params[1]**2) ) )
        y = C*sum1/(inf_sum)    
    
    return y
    

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
    if N < 4:
        AIC = -2*LnL + 2*params
    else:
        AIC = -2*LnL + 2*params + ((2*params*(params + 1)))/(N - params - 1)
    return AIC

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

def MLE(X:np.ndarray, k_min:int = 1, vt:int = 3):
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
    
    while np.std(votes[-vt:]) >= 0.1:   
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
        
        opt_wb = minimize(weibull, (k_mean,1),(x, inf, sum_log, delta, k_min), method = 'SLSQP', bounds=((0.05, None),(0.05, 4),))
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
            votes += [10] #if BIC and AIC do not choose the same distribution, 
                          #move to next k_min
        k_min += 1
    
    Delta = (X[X < (k_min-vt)]).size/X.size
    Final_dist = [k_min-vt, Distributions[np.argmin(AICs)],np.round(Results[Distributions[np.argmin(AICs)]][k_min-vt][0],2), np.round(Delta,2)]
    return Final_dist

def bootstrap(G_list, G_result):
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
    sample_dists = []
    param1 = []
    param2 = []
    while len(param1) < 1000:
        #get sample of size N with replacement and do MLE methodology on it
        sample = np.array(np.random.choice(G_list, len(G_list), replace=True))
        sample_result, _ = MLE(sample, G_result[0], 3)
        sample_dists.append(sample_result[1])
        if sample_result[1] == G_result[1]:
            #if there are two parameters
            if len(sample_result[2]) == 2:
                #if both parameters are not nan and nonzero then we include them both
                if np.isnan(sample_result[2][0]) == False and np.isnan(sample_result[2][1]) == False and sample_result[2][1] != 0:
                    param1.append(sample_result[2][0])
                    param2.append(sample_result[2][1])
            #if there is only one parameter, we only check one        
            if np.isnan(sample_result[2][0]) == False:
                param1.append(sample_result[2][0])
    return  param1, param2

def find_split(X, P, Y):
    """
    
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
        _, p_val = ks_2samp(P[0:i], Y[0:i])
        if p_val < 0.05:
            break
    if i - 10 > 0:    
        return i - 10
    else:
        return i

def two_fits(X, k_star, initial_result):
    """

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

def fit(Name, G, k_min:int=1):
     """
     Performs MLE on a degree sequence and performs goodness of fit KS-test.
     If the distribution is not a good fit it tries a higher k_min.
     If this still does not work it tries multiple distributions.
    
     Parameters
     ----------
     G : array-like
         degree list
     k_min : int, optional
         Initial k_min value. The default is 1.
     vt : int, optional
         Number of consecutive votes required for a distribution
         to be chosen. The default is 3.
         
     Returns
     -------
     result : list of lists
         List [kmin, distribution name, parameters, Delta] 
         for each regime in the distribution
        
     Input : list of lists
         X values for plotting fitted distributions
     Y : list of lists
         Y values for plotting fitted distributions

     """
     X = degree_list(G)
     if len(X) < 2500:
         vt = 2
     else:
         vt = 3
     result = [MLE(X, k_min, vt)]
     print(result)
     N, P = empirical(X)
     Initial_Y = CCDF(result[0], N,N, P)
     KS, p_val = ks_2samp(P, Initial_Y)
     if p_val < 0.5: #the distribution is not a good fit
         new_result = [MLE(X, result[0][0]+3)] #we try MLE with slightly larger k_min 
         Y = [CCDF(new_result[0], N,N, P)]
         _, p_val = ks_2samp(P, Y[0])
         if p_val < 0.5: # the distribution is still not a good fit
             k_star = find_split(X, P, Initial_Y)
             print('kstar =', k_star)
             new_result = two_fits(X, k_star, result)
         result = new_result
     if len(result) == 1:
         Input = [np.arange(result[0][0],np.amax(X)+1)]
         Y = [CCDF(result[0], Input[0],N, P)]
     if len(result) == 2:
         Input = [np.arange(result[0][0],result[1][0]), np.arange(result[1][0],np.amax(X)+1)]        
         Y = [CCDF(result[0], Input[0],N, P), CCDF(result[1], Input[1],N, P)]
     plt.step(N, P, '+', ms = 4, color ='k', label = 'Actual')
     for i in range(len(Y)):
         j = i+1
         plt.plot(Input[i], Y[i],label = 'Fit %s'%j)
     plt.xscale('log')
     plt.yscale('log')
     plt.title(Name)
     plt.legend()#
     plt.show()        
     if len(result) == 1:
         print('The degree distribution follows a', result[0][1], 'distribution with parameters', result[0][2])
     if len(result) == 2:
         print(Name,':For k less than', result[1][0], 'the degree distribution follows a', result[0][1],'\
               distribution with parameters', result[0][2],'.\n For k greater than', result[1][0], 'the degree\
               distrubtion follows a ', result[1][1], 'distribution with parameters', result[1][2])
     return result, Input, Y
    

