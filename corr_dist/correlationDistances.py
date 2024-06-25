"""
This script will create 10 instances of each of our 5 distributions.
It will attempt to rewire each of them to += 0.01 assortativity using the 
new method.
"""

import pandas as pd
import numpy as np
import sys
import networkx as nx
import concurrent.futures
import Functions.graph_tool_functions as gtf
from degree_preserving_rewiring import rewire, generate_graph

from itertools import repeat
from assortativity import assortativity_d_max

if __name__ == "__main__":
    targets = {'exponential': [-0.4, 0.4], 'poisson': [-0.4, 0.4], 'powerlaw':[-0.1, 0.2], 
               'powerlaw_4': [-0.1, 0.2], 'lognormal':[-0.4, 0.4], 'weibull':[-0.3, 0.3]}

    columns = []
    for i in range(1,1001):
        columns.append(f'r{i}')
    df_r = pd.DataFrame(columns = columns)
    print(f'the number of columns is {len(columns)}')
    columns = []
    for i in range(1,1001):
        columns.append(f's{i}')
    df_s = pd.DataFrame(columns = columns)
    dist = sys.argv[1]
    for i in range(1000):
        print(f'Negative number {i}')
        if i < 10:
            G = nx.read_gml(f'../graphs/{dist}/{dist}_0{i}.gml')
        else:
            G = nx.read_gml(f'../graphs/{dist}/{dist}_{i}.gml')
        G, results = rewire(G, targets[dist][0], f'{dist}_{i}', method='new', sample_size=50, return_type='full')
        g = gtf.nx2gt(G)
        r, s = gtf.gt_assortativity_all(g)
        while len(r) < 1000:
            r.append(0)
            s.append(0)

        try:
            df_r.loc[len(df_r)] = r
            df_s.loc[len(df_s)] = s
        except ValueError:
            pass

    df_r.to_csv(f'../results/{dist}/{dist}_neg_r.csv')
    df_s.to_csv(f'../results/{dist}/{dist}_neg_s.csv')

    columns = []
    for i in range(1,1001):
        columns.append(f'r{i}')
    df_r = pd.DataFrame(columns = columns)
    columns = []
    for i in range(1,1001):
        columns.append(f's{i}')
    df_s = pd.DataFrame(columns = columns)
    dist = sys.argv[1]
    for i in range(1000):
        print(f'Positive number {i}')
        if i < 10:
            G = nx.read_gml(f'../graphs/{dist}/{dist}_0{i}.gml')
        else:
            G = nx.read_gml(f'../graphs/{dist}/{dist}_{i}.gml')
        G, results = rewire(G, targets[dist][1], f'{dist}_{i}', method='new', sample_size=50, return_type='full')
        g = gtf.nx2gt(G)
        r, s = gtf.gt_assortativity_all(g)
        while len(r) < 1000:
            r.append(0)
            s.append(0)
        try:
            df_r.loc[len(df_r)] = r
            df_s.loc[len(df_s)] = s
        except ValueError:
            pass

    df_r.to_csv(f'../results/{dist}/{dist}_pos_r.csv')
    df_s.to_csv(f'../results/{dist}/{dist}_pos_s.csv')
