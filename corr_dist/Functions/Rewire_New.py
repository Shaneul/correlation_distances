# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:37:10 2021

@author: shane
"""

from Functions.Assortativity import Assortativity
import random

def Increase_Assortativity(G, New_Assort):
    loops = 0
    edges_added = 0
    edges_removed = 0
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
            Pos_nodes.remove(n)
            try:
                new_node = random.choice(Pos_nodes)
            except IndexError:
                break
        
        if G.degree(n) < E_k:
            Neg_nodes.remove(n)
            try:
                new_node = random.choice(Neg_nodes)
            except IndexError:
                break
        
        if G.has_edge(n, new_node) == False and G.has_edge(e_neg[0], e_neg[1]) == True:
            G.remove_edge(e_neg[0], e_neg[1])
            edges_removed += 1
            G.add_edge(n, new_node)
            edges_added += 1
        
        Neg_edges.remove(e_neg)  
        loops += 1
    print(f'Edges added: {edges_added}. Edges removed: {edges_removed}')
    return loops, G, E_k, Assortativity(G)

def Decrease_Assortativity(G, New_Assort):
    loops = 0
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
        loops += 1
    return loops, G, E_k, Assortativity(G)

def Increase_Assortativity_N(G, New_Assort, N_loops, step):
    loops = 0
    R = 0
    while R < New_Assort:
        if loops < N_loops:
            R = 0
        else:
            R = Assortativity(G)
            N_loops += step
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
                            
        e_neg = random.choice(Neg_edges)
        n = random.choice(e_neg)
        
        if G.degree(n) > E_k:
            new_node = random.choice(Pos_nodes)
        
        if G.degree(n) < E_k:
            new_node = random.choice(Neg_nodes)
        
        G.remove_edge(e_neg[0], e_neg[1])
        G.add_edge(n, new_node)
        loops += 1
    return loops, G, E_k, Assortativity(G)

def Decrease_Assortativity_N(G, New_Assort, N_loops, step):
    loops = 0
    R = 0
    while R > New_Assort:
        if loops < N_loops:
            R = 0
        else:
            R = Assortativity(G)
            N_loops += step
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
        loops += 1
    return loops, G, E_k, Assortativity(G)
