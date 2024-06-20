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
from degree_preserving_rewiring import rewire, generate_graph
from itertools import repeat
from assortativity import assortativity_d_max

if __name__ == "__main__":
    columns = []
    for i in range(1,101):
        columns.append(f'r{i}')
    df_r = pd.DataFrame(columns = columns)
    print(f'the number of columns is {len(columns)}')
    columns = []
    for i in range(1,101):
        columns.append(f's{i}')
    df_s = pd.DataFrame(columns = columns)
    dist = sys.argv[1]
    for i in range(1000):
        if i < 10:
            G = nx.read_gml(f'../graphs/{dist}/{dist}_0{i}.gml')
        else:
            G = nx.read_gml(f'../graphs/{dist}/{dist}_{i}.gml')
        G, results = rewire(G, -1, f'{dist}_{i}', method='max', return_type='full')
        s, r = assortativity_d_max(G)
        while len(r) < 100:
            r.append(0)
            s.append(0)

        print(f'columns have length {len(columns)}, r and s have {len(r)} and {len(s)}')
        df_r.loc[len(df_r)] = r
        df_s.loc[len(df_s)] = s

    df_r.to_csv(f'../results/{dist}/{dist}_neg_r.csv')
    df_s.to_csv(f'../results/{dist}/{dist}_neg_s.csv')
    
    columns = []
    for i in range(1,101):
        columns.append(f'r{i}')
    df_r = pd.DataFrame(columns = columns)
    columns = []
    for i in range(1,101):
        columns.append(f's{i}')
    df_s = pd.DataFrame(columns = columns)
    dist = sys.argv[1]
    for i in range(1000):
        G = nx.read_gml(f'../graphs/{dist}/{dist}_{i}.gml')
        G, results = rewire(G, 1, f'{dist}_{i}', method='max', return_type='full')
        s, r = assortativity_d_max(G)
        while len(r) < 100:
            r.append(0)
            s.append(0)
        df_r.loc[len(df_r)] = r
        df_s.loc[len(df_s)] = s

    df_r.to_csv(f'../results/{dist}/{dist}_pos_r.csv')
    df_s.to_csv(f'../results/{dist}/{dist}_pos_s.csv')
