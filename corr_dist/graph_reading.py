import pandas as pd
import networkx as nx
import pickle
from degree_preserving_rewiring import rewire

if __name__ == "__main__":
    with open(f'../graphs/graph_dictionary.pkl', 'rb') as f:
        gd = pickle.load(f)    
   
    
    for dist in ['powerlaw', 'poisson', 'exponential', 'lognormal', 'weibull', 'powerlaw_4']:
        df_min = pd.DataFrame()
        df_max = pd.DataFrame()
        rmin = []
        rmax = []
        for i in range(1000):
            rmin.append(gd[f'{dist}_{i}']['min'])
            rmax.append(gd[f'{dist}_{i}']['max'])
            
        df_min[f'{dist}'] = pd.Series(rmin)
        df_max[f'{dist}'] = pd.Series(rmax)

    df_min.to_csv('../results/graph_mins.csv')
    df_max.to_csv('../results/graph_max.csv')
