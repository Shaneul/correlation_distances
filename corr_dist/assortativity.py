# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:51:13 2021

@author: shane
"""
import networkx as nx
import time 

def Assortativity(G, printing=False):
    E_k_sum = 0
    E_k_q_sum = 0
    E_k_Squared_sum = 0
    L = len(G.edges())
    for e in G.edges:
        E_k_sum += G.degree(e[0])
        E_k_sum += G.degree(e[1])
        E_k_Squared_sum += (G.degree(e[0]))**2
        E_k_Squared_sum += (G.degree(e[1]))**2
        E_k_q_sum += (G.degree(e[1]))*(G.degree(e[0]))
        E_k_q_sum += (G.degree(e[0]))*(G.degree(e[1]))
    E_k = E_k_sum/(2*L)
    
    E_k_q = E_k_q_sum/(2*L)
    
    E_k_Squared = E_k_Squared_sum/(2*L)
    
    TL = E_k_q - (E_k)**2
    
    BL = E_k_Squared - (E_k)**2
    if printing == True:
        print(f'ek : {E_k}')
        print(f'ekq : {E_k_q}')
        print(f'Ek sq: {E_k_Squared}')
        print(f'TL : {TL}')
        print(f'BL : {BL}')
    r = TL/BL
    return r

def Assortativity_D2(G):
    T_start = time.time()
    sp = nx.all_pairs_shortest_path_length(G, cutoff = 2)
    S_2 = 0
    K_2s = 0
    K_2s_Squared = 0
    for i in range(len(G)):
#        if i % 5000 == 0:
#            print('loop 1 iteration', i)
        pathlist = next(sp)[1]
        for j in pathlist:
            if pathlist[j] == 2:
                S_2 += 1
                K_2s += (G.degree(i))
                K_2s_Squared += (G.degree(i)**2)
        del pathlist        
    S2 = S_2/2
    E = K_2s/(S_2)
    E_Squared = K_2s_Squared/(S_2)
    TL = 0
    del sp
    sp2 = nx.all_pairs_shortest_path_length(G, cutoff = 2)    
    for i in range(len(G)):
#        if i % 5000 == 0:
#            print('loop 2 iteration', i)        
        pathlist = next(sp2)[1]
        for j in pathlist:
            if pathlist[j] == 2:
                TL += (G.degree(i) - E) * (G.degree(j) - E)
        del pathlist        
    BL = (S_2) * (E_Squared - (E)**2)            
    r2 = TL/BL
    T_end = time.time()
    T =  T_end - T_start
    return S2, r2

def Assortativity_D2_New(G):
    T_start = time.time()
    sp = nx.all_pairs_shortest_path_length(G, cutoff = 2)
    S_2 = 0
    K_2s = 0
    K_2s_Squared = 0
    T1 = 0
    T2 = 0
    for i in range(len(G)):
#        if i % 5000 == 0:
#            print('loop 1 iteration', i)
        pathlist = next(sp)[1]
        for j in pathlist:
            if pathlist[j] == 2:
                S_2 += 1
                K_2s += (G.degree(i))
                K_2s_Squared += (G.degree(i)**2)
                T1 += (G.degree(i))*(G.degree(j))
                T2 += (G.degree(i) + G.degree(j))
        del pathlist        
    S2 = S_2/2
    E = K_2s/(S_2)
    E_Squared = K_2s_Squared/(S_2)    
    TL = T1 - (T2*E) + ((E**2)*S_2)
    BL = (S_2) * (E_Squared - (E)**2)            
    r2 = TL/BL
    T_end = time.time()
    T = T_end-T_start
    return S2, r2

def Assortativity_D3(G):
    T_s = time.time()
    sp = nx.all_pairs_shortest_path_length(G, cutoff = 3)
    S_2 = 0
    K_2s = 0
    K_2s_Squared = 0
    for i in range(len(G)):
#        if i % 5000 == 0:
#            print('loop 1 iteration', i)        
        pathlist = next(sp)[1]
        for j in pathlist:
            if pathlist[j] == 3:
                S_2 += 1
                K_2s += (G.degree(i))
                K_2s_Squared += (G.degree(i))**2
        del pathlist    
    S3 = S_2/2
    E = K_2s/(S_2)
    E_Squared = K_2s_Squared/(S_2)
    TL = 0
    del sp
    sp2 = nx.all_pairs_shortest_path_length(G, cutoff = 3) 
    for i in range(len(G)):
#        if i % 5000 == 0:
#            print('loop 2 iteration', i)        
        pathlist = next(sp2)[1]
        for j in pathlist:
            if pathlist[j] == 3:
                TL += (G.degree(j) - E) * (G.degree(i) - E) 
        del pathlist        
    BL = (S_2) * (E_Squared - (E)**2)            
    r3 = TL/BL
    T_e = time.time()
    T = T_e - T_s
    return S3, r3

def Assortativity_D3_New(G):
    T_start = time.time()
    sp = nx.all_pairs_shortest_path_length(G, cutoff = 3)
    S_2 = 0
    K_2s = 0
    K_2s_Squared = 0
    T1 = 0
    T2 = 0 
    for i in range(len(G)):
#        if i % 5000 == 0:
#            print('loop 1 iteration', i)        
        pathlist = next(sp)[1]
        for j in pathlist:
            if pathlist[j] == 3:
                S_2 += 1
                K_2s += (G.degree(i))
                K_2s_Squared += (G.degree(i))**2
                T1 += (G.degree(i))*(G.degree(j))
                T2 += (G.degree(i) + G.degree(j))
        del pathlist    
    S3 = S_2/2
    E = K_2s/(S_2)
    E_Squared = K_2s_Squared/(S_2)
    TL = T1 - (T2*E) + ((E**2)*S_2)
    BL = (S_2) * (E_Squared - (E)**2)            
    r3 = TL/BL
    T_end = time.time()
    T = T_end - T_start
    return S3, r3

def assortativity_d_max(G, D=100):
    s = []
    r = []
    m = 1
    has_path = True
    nodelist = list(G.nodes)
    while has_path == True:
        sp = nx.all_pairs_shortest_path_length(G, cutoff = m)
        S_2 = 0
        K_2s = 0
        K_2s_Squared = 0
        for i in nodelist:
            pathlist = next(sp)[1]
            for j in pathlist:
                if pathlist[j] == m:
                    S_2 += 1
                    K_2s += (G.degree[i])
                    K_2s_Squared += (G.degree[i]**2)
        if S_2 != 0:
       
            S2 = S_2/2
            E = K_2s/(S_2)
            E_Squared = K_2s_Squared/(S_2)    
            TL = 0
            sp = nx.all_pairs_shortest_path_length(G, cutoff = m)
            for i in nodelist:
                pathlist = next(sp)[1]
                for j in pathlist:
                    if pathlist[j] == m:
                        TL += (G.degree[i] - E) * (G.degree[j] - E)
            BL = (S_2) * (E_Squared - (E)**2)  
            try:
                r2 = TL/BL
            except ZeroDivisionError:
                r2 = 0
            s.append(S2)
            r.append(r2)
            m += 1
            
        if S_2 == 0:
            has_path = False
        if m > D:
            has_path = False
    
    return s, r        
        
        
        
        
