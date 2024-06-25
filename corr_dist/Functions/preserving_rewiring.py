# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:25:24 2023

@author: shane
"""

import networkx as nx
import random
import Functions.MLE_functions as mle
import Functions.Assortativity as assort
from Functions.Rewire_New import Increase_Assortativity, Decrease_Assortativity

def degree_preserving_rewire(G: nx.Graph, target_assort:float, max_tries:int):
    tries = 0
    print(f'Initial Assortativity: {nx.degree_assortativity_coefficient(G)}')
    initial_assortativity = assort.Assortativity(G)
    while abs(assort.Assortativity(G) - target_assort) > 0.01:
        G_edges = list(G.edges())
        E_k = 0 #get avg degree at end of an edge
        
        
        for edge in G.edges():
            E_k += G.degree(edge[0]) + G.degree(edge[1])
        
        E_k /= 2*len(G.edges())
        
        #must ensure number of unique nodes is 4, to avoid self-edges
        n_unique_nodes = 3
        while n_unique_nodes < 4: 
            edge_sample = random.sample(G_edges, 2) # select two edges
            unique_nodes = set()
            for edge in edge_sample:
                unique_nodes.add(edge[0])
                unique_nodes.add(edge[1])
            n_unique_nodes = len(unique_nodes)
                
        #print(f'EK: {E_k}')
        #print(f'edges: {edge_sample}')
        initial_contribution = 0
        for edge in edge_sample:
            #print(f'edge degrees: {G.degree(edge[0])}, {G.degree(edge[0])}')
            initial_contribution += ((G.degree(edge[0]) - E_k)*(G.degree(edge[1]) - E_k))
            
        edge_dict = {'pair 1': [(edge_sample[0][0], edge_sample[1][0]), (edge_sample[0][1], edge_sample[1][1])],
                     'pair 2': [(edge_sample[0][0], edge_sample[1][1]), (edge_sample[0][1], edge_sample[1][0])]}
        
        contribution_dict = {'pair 1':0,
                             'pair 2':0}
        
        for edge_pair in edge_dict:
            for edge in edge_dict[edge_pair]:
                contribution_dict[edge_pair] += ((G.degree(edge[0]) - E_k)*(G.degree(edge[1]) - E_k))
        
        edges_to_add = None
        if target_assort < initial_assortativity:
            if min(contribution_dict.values()) < initial_contribution:
                edges_to_add = edge_dict[min(contribution_dict)]
            
        else:
            if max(contribution_dict.values()) > initial_contribution:
                #print(f'contribution of edge pair {initial_contribution}')
                #print(f'Contribution of rewired pair {max(contribution_dict.values())}')
                edges_to_add = edge_dict[max(contribution_dict)]
            
        if edges_to_add != None:
            for edge in edges_to_add:
                if G.has_edge(edge[0], edge[1]) == True:
                    edges_to_add.remove(edge)
                    
            if len(edges_to_add) == 2:

                G.remove_edges_from(edge_sample) 
                G.add_edges_from(edges_to_add)

                    
            
            new_assortativity = assort.Assortativity(G)
            delta_assort = new_assortativity - initial_assortativity
            #print(f'{delta_assort}')
            tries = 0
        else:
            tries += 1
            
        if tries == max_tries:
            print('max attempts reached')
            return G
    print(f'{abs(target_assort - assort.Assortativity(G))} from target')
    return G
 