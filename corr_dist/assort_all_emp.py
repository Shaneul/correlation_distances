# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:46:06 2023

@author: shane
"""


import graph_tool.all as gt
import networkx as nx
import numpy as np
import Functions.graph_tool_functions as gtf
import pickle
from Functions.CreateNetworks import Create_Network1, Create_Network2

with open('data/data_dictionary', 'rb') as f:
    data_dict = pickle.load(f)
    
assortativity_result_dict = {}
for network in data_dict:
    if network not in ['ASoIaF', 'Laxdaela', 'RIP Ireland']:
        if data_dict[network][1] == 1:
            G = Create_Network1(data_dict[network][0])
        else:
            G = Create_Network2(data_dict[network][0])
            
            
        
        g = gtf.nx2gt(G)
        
        assortativity, npaths = gtf.gt_assortativity_all(g)
        assortativity_result_dict[network] = [assortativity, npaths]
        
        
        
with open('results/assortativity_all', 'wb') as f:
    pickle.dump(assortativity_result_dict, f)
