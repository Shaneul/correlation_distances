"""
A script to generate graphs from each of our 5 studied distributions and save them 
for later study
"""

import networkx as nx
from degree_preserving_rewiring import generate_graph

if __name__ == "__main__":
    for dist in ['exponential', 'poisson', 'powerlaw', 'lognormal', 'weibull']:
        for i in range(100):
            if dist in ['exponential', 'lognormal', 'weibull']:
                G = generate_graph(dist, 5, 10000)
            if dist == 'poisson':
                G = nx.erdos_renyi_graph(10000, 5/9999)
            if dist == 'powerlaw':
                G = nx.barabasi_albert_graph(10000, m=3)

            nx.write_gml(G, f'../graphs/{dist}/{dist}_{i}.gml')

            
