import pandas as pd
import networkx as nx
import pickle
from degree_preserving_rewiring import rewire

if __name__ == "__main__":
    with open(f'../graphs/graph_dictionary.pkl', 'rb') as f:
        gd = pickle.load(f)

    mins = 0
    maxes = 0
    for graph in gd:
        if 'powerlaw_4_' in graph:
            mins += gd[graph]['min']
            maxes += gd[graph]['max']
    
    print(gd)
    print(f'for the powerlaw with m = 4, mins is {mins/1000}')
    print(f'for the powerlaw with m = 4, maxes is {maxes/1000}')
