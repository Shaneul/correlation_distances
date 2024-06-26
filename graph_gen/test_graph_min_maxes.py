import pandas as pd
import networkx as nx
import pickle
from degree_preserving_rewiring import rewire

if __name__ == "__main__":
    graph_dict = {}
    for dist in ['powerlaw', 'poisson', 'exponential', 'lognormal', 'weibull', 'powerlaw_4']:
        for i in range(1000):
            graph_dict[f'{dist}_{i}'] = {}
            if dist == 'powerlaw_4':
                if i < 10:
                    G = nx.read_gml(f'../graphs/{dist}/powerlaw_0{i}.gml')
                else:
                    G = nx.read_gml(f'../graphs/{dist}/powerlaw_{i}.gml')
            else:
                if i < 10:
                    G = nx.read_gml(f'../graphs/{dist}/{dist}_0{i}.gml')
                else:
                    G = nx.read_gml(f'../graphs/{dist}/{dist}_{i}.gml')
            results = rewire(G, -1, f'{dist}_{i}', method='max', return_type='summary')
            r_min = results.iloc[0]['r']
            r_min_preserved = results.iloc[0]['preserved']
            graph_dict[f'{dist}_{i}']['min'] = r_min
            graph_dict[f'{dist}_{i}']['min_preserved'] = r_min_preserved
            
            
            if dist == 'powerlaw_4':
                if i < 10:
                    G = nx.read_gml(f'../graphs/{dist}/powerlaw_0{i}.gml')
                else:
                    G = nx.read_gml(f'../graphs/{dist}/powerlaw_{i}.gml')
            else:
                if i < 10:
                    G = nx.read_gml(f'../graphs/{dist}/{dist}_0{i}.gml')
                else:
                    G = nx.read_gml(f'../graphs/{dist}/{dist}_{i}.gml')
            results = rewire(G, 1, f'{dist}_{i}', method='max', return_type='summary')
            r_max = results.iloc[0]['r']
            r_max_preserved = results.iloc[0]['preserved']
            graph_dict[f'{dist}_{i}']['max'] = r_max
            graph_dict[f'{dist}_{i}']['max_preserved'] = r_max_preserved

    with open(f'../graphs/graph_dictionary.pkl', 'wb') as f:
        pickle.dump(graph_dict, f, protocol= pickle.HIGHEST_PROTOCOL)
