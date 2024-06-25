"""
A script to generate graphs from each of our 5 studied distributions and save them 
for later study
"""

import networkx as nx
from degree_preserving_rewiring import generate_graph

if __name__ == "__main__":
    for dist in ['weibull', 'exponential', 'lognormal']:
        for i in range(1000):
            if dist in ['exponential', 'lognormal', 'weibull']:
                G = generate_graph(dist, 5, 10000)
            #if dist == 'poisson':
            #    G = nx.erdos_renyi_graph(10000, 5/9999)
            if dist == 'powerlaw':
                G = nx.barabasi_albert_graph(10000, m=4)
            
            if i < 10:
              nx.write_gml(G, f'../graphs/{dist}/{dist}_0{i}.gml')
            else:
              nx.write_gml(G, f'../graphs/{dist}/{dist}_{i}.gml')

            
