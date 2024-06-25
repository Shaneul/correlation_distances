# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:37:02 2021

@author: shane
"""

def Avg_Shortest_Path_Length(G):
    import networkx as nx
    apl = []
    diameters = []
    len_comp = []
    for g in (G.subgraph(c).copy() for c in nx.connected_components(G)):
        if len(g) > 1:
            apl += [nx.average_shortest_path_length(g)]
            diameters.append(nx.diameter(g))
            len_comp += [len(g)]
    d = 0
    for i in range(len(apl)):
        d += apl[i]*len_comp[i]*(len_comp[i]-1)
    n_m = 0
    for i in len_comp:
        n_m += i*(i-1.)
    path_length = d/n_m
    max_l = max(diameters)
    return max_l, path_length