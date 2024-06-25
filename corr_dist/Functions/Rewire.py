# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:37:10 2021

@author: shane
"""

def Increase_Assortativity(G, New_Assort):
    from Functions.Assortativity import Assortativity
    import random
    loops = 0
    while Assortativity(G) < New_Assort:
        L = len(G.edges())
        E_k_sum = 0
        Pos_nodes = []
        Neg_nodes = []
        Pos_edges = []
        Neg_edges = []        
        for e in G.edges:
            E_k_sum += G.degree(e[0])
            E_k_sum += G.degree(e[1])
        E_k = E_k_sum/(2*L)
        
        for n in G.nodes():
            if G.degree(n) > E_k:
                Pos_nodes.append(n)
            else:
                Neg_nodes.append(n)    

        for e in G.edges:
            if G.degree(e[0]) < E_k < G.degree(e[1]):
                Neg_edges.append(e)
            else:
                if G.degree(e[1]) < E_k < G.degree(e[0]):
                    Neg_edges.append(e) 
                else:    
                    Pos_edges.append(e)
                            
        e_neg = random.choice(Neg_edges)
        n = random.choice(e_neg)
        
        if G.degree(n) > E_k:
            new_node = random.choice(Pos_nodes)
        
        if G.degree(n) < E_k:
            new_node = random.choice(Neg_nodes)
        
        G.remove_edge(e_neg[0], e_neg[1])
        G.add_edge(n, new_node)
        Neg_edges.remove(e_neg)  
        loops += 1
    return loops, G, E_k, Assortativity(G)

def Decrease_Assortativity(G, New_Assort):
    from Functions.Assortativity import Assortativity
    import random
    while Assortativity(G) > New_Assort:
        L = len(G.edges())
        E_k_sum = 0
        Pos_nodes = []
        Neg_nodes = []
        Pos_edges = []
        Neg_edges = []        
        for e in G.edges:
            E_k_sum += G.degree(e[0])
            E_k_sum += G.degree(e[1])
        E_k = E_k_sum/(2*L)
        
        for n in G.nodes():
            if G.degree(n) > E_k:
                Pos_nodes.append(n)
            else:
                Neg_nodes.append(n)    

        for e in G.edges:
            if G.degree(e[0]) < E_k < G.degree(e[1]) or G.degree(e[1]) < E_k < G.degree(e[0]):
                Neg_edges.append(e)
            else:
                Pos_edges.append(e)
                            
        e_pos = random.choice(Pos_edges)
        n = random.choice(e_pos)
        
        if G.degree(n) > E_k:
            new_node = random.choice(Neg_nodes)
        
        if G.degree(n) < E_k:
            new_node = random.choice(Pos_nodes)
        
        G.remove_edge(e_pos[0], e_pos[1])
        G.add_edge(n, new_node)
        Pos_edges.remove(e_pos)  
        
    return G, E_k, Assortativity(G)

def Increase_Assortativity_N(G, New_Assort, N_loops):
    from Functions.Assortativity import Assortativity
    import random
    loops = 0
    R = 0
    while R < New_Assort:
        if loops < N_loops:
            R = 0
        else:
            R = Assortativity(G)
            N_loops += 100
        L = len(G.edges())
        E_k_sum = 0
        Pos_nodes = []
        Neg_nodes = []
        Pos_edges = []
        Neg_edges = []        
        for e in G.edges:
            E_k_sum += G.degree(e[0])
            E_k_sum += G.degree(e[1])
        E_k = E_k_sum/(2*L)
        
        for n in G.nodes():
            if G.degree(n) > E_k:
                Pos_nodes.append(n)
            else:
                Neg_nodes.append(n)    

        for e in G.edges:
            if G.degree(e[0]) < E_k < G.degree(e[1]):
                Neg_edges.append(e)
            else:
                if G.degree(e[1]) < E_k < G.degree(e[0]):
                    Neg_edges.append(e) 
                else:    
                    Pos_edges.append(e)
                            
        e_neg = random.choice(Neg_edges)
        n = random.choice(e_neg)
        
        if G.degree(n) > E_k:
            new_node = random.choice(Pos_nodes)
        
        if G.degree(n) < E_k:
            new_node = random.choice(Neg_nodes)
        
        G.remove_edge(e_neg[0], e_neg[1])
        G.add_edge(n, new_node)
        Neg_edges.remove(e_neg)  
        loops += 1
    return loops, G, E_k, Assortativity(G)
