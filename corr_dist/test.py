"""
This script will create 10 instances of each of our 5 distributions.
It will attempt to rewire each of them to += 0.01 assortativity using the 
new method.
"""

import pandas as pd
import numpy as np
import time
import sys
import networkx as nx
import concurrent.futures
import graph_tool.all as gt
import Functions.graph_tool_functions as gtf
from degree_preserving_rewiring import rewire, generate_graph, degree_list
from itertools import repeat
from assortativity import assortativity_d_max

if __name__ == "__main__":
    G = nx.read_gml('../graphs/weibull/weibull_23.gml')
    print(f' G has {len(G.nodes)} nodes and assortativity {nx.degree_assortativity_coefficient(G)}')
    G = nx.read_gml('../graphs/poisson/poisson_23.gml')
    print(f' G has {len(G.nodes)} nodes and assortativity {nx.degree_assortativity_coefficient(G)}')
    G = nx.read_gml('../graphs/powerlaw/powerlaw_23.gml')
    print(f' G has {len(G.nodes)} nodes and assortativity {nx.degree_assortativity_coefficient(G)}')
    G = nx.read_gml('../graphs/lognormal/lognormal_23.gml')
    print(f' G has {len(G.nodes)} nodes and assortativity {nx.degree_assortativity_coefficient(G)}')
    G = nx.read_gml('../graphs/exponential/exponential_23.gml')
    print(f' G has {len(G.nodes)} nodes and assortativity {nx.degree_assortativity_coefficient(G)}')
    start = time.time()
    G, results = rewire(G, -0.1, 'test', sample_size=8, method='new', return_type='full', time_limit = 120)
    end = time.time()
    after = degree_list(G)
    print(f'Rewiring took {end - start} seconds')
    start = time.time()
    g = gtf.nx2gt(G)
    end = time.time()
    print(f'Converting took {end - start} seconds')
    diam, _ = gt.pseudo_diameter(g)
    print(f'diameter is {diam}')
    start = time.time()
    r, s = gtf.gt_assortativity_all(g)
    end = time.time()
    print(f'Assortativity took {end - start} seconds')
    print(s)

