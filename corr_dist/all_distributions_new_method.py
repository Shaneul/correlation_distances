"""
This script will create 10 instances of each of our 5 distributions.
It will attempt to rewire each of them to += 0.01 assortativity using the 
new method.
"""

import pandas as pd
import numpy as np
import networkx as nx
import sys
import concurrent.futures
from degree_preserving_rewiring import rewire, generate_graph
from assortativity import assortativity_d_max
from itertools import repeat

def load_and_rewire(itr):
    dist = itr[0]
    target = itr[1]
    indexes = itr[2]
    for i in indexes:
        if i < 10:
            G = nx.read_gml(f'../graphs/{dist}/{dist}_0{i}.gml')
        else:
            G = nx.read_gml(f'../graphs/{dist}/{dist}_{i}.gml')
        G, _ = rewire(G, target, f'{dist}', method='new', sample_size=50, return_type='full')
        s, r = assortativity_d_max(G, 100)

    return [r, s]


if __name__ == "__main__":

    targets = {'exponential': [-0.4, 0.4], 'poisson': [-0.4, 0.4], 'powerlaw':[-0.1, 0.2], 
           'powerlaw_4': [-0.1, 0.2], 'lognormal':[-0.4, 0.4], 'weibull':[-0.3, 0.3]}

    dist = sys.argv[1]
    itrs = []
    
    df_r_neg = pd.DataFrame()
    df_r_pos = pd.DataFrame()
    df_s_neg = pd.DataFrame()
    df_s_pos = pd.DataFrame()

    starts = []
    i = 0;
    while i < 901:
        starts.append(i)
        i += 100
    for start in starts:
        indexes = []
        for k in range(start, start + 100):
            indexes.append(k)

        itrs.append([dist, targets[dist][0], indexes])
        itrs.append([dist, targets[dist][1], indexes])

    for i in range(10):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(load_and_rewire, itrs)
        
        for result in results:
            if(result[0][0] < 0):
                df_r_neg = pd.concat([df_r_neg, pd.Series(result[0])], ignore_index=True)
                df_s_neg = pd.concat([df_s_neg, pd.Series(result[1])], ignore_index=True)
            else:
                df_r_pos = pd.concat([df_r_pos, pd.Series(result[0])], ignore_index=True)
                df_s_pos = pd.concat([df_s_pos, pd.Series(result[1])], ignore_index=True)

    df_r_neg.to_csv('../results/{dist}_r_neg.csv', index=False)
    df_r_pos.to_csv('../results/{dist}_r_pos.csv', index=False)
    df_s_neg.to_csv('../results/{dist}_s_neg.csv', index=False)
    df_s_pos.to_csv('../results/{dist}_s_pos.csv', index=False)
