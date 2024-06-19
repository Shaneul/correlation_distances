"""
This script will create 10 instances of each of our 5 distributions.
It will attempt to rewire each of them to += 0.01 assortativity using the 
new method.
"""

import pandas as pd
import numpy as np
import networkx as nx
import concurrent.futures
from degree_preserving_rewiring import rewire, generate_graph
from itertools import repeat

def create_and_rewire(itr):
    distribution = itr[0]
    target = itr[1]
    sample_size = itr[2]
    if distribution in ['exponential', 'lognormal', 'weibull']:
        G = generate_graph(distribution, 5, 5000)
    if distribution == 'poisson':
        G = nx.erdos_renyi_graph(5000, 5/4999)
    if distribution == 'powerlaw':
        G = nx.barabasi_albert_graph(5000, m=3)

    if target < 0:
        current_r = nx.degree_assortativity_coefficient(G)
        target = current_r - 0.05

    G, results = rewire(G, target, distribution, sample_size, timed=True, time_limit = 3600, method='new', return_type='full')

    return results

if __name__ == "__main__":

    
    df = pd.DataFrame(columns = ['name', 'iteration', 'time', 'r', 'target_r',
                                 'sample_size', 'edges_rewired', 'duplicate_edges',
                                 'self_edges', 'existing_edges', 'preserved','method', 'summary'])

    minmaxes = {'exponential':[-.85, 0.9], 'lognormal':[-.7, .85], 
                'poisson':[-.9, .9], 'weibull':[-.7, .4], 'powerlaw':[-.25, .15]}
    itrs = []
    for dist in minmaxes:
        for sample_size in [2,5,10,20,50]:
            itrs.append([dist, minmaxes[dist][0], sample_size])
            itrs.append([dist, minmaxes[dist][1], sample_size])

    for i in range(5):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(create_and_rewire, itrs)
        
        for result in results:
            df = pd.concat([df, result], ignore_index=True)

    df.to_csv('../results/all_distributions_new_method.csv', index=False)
