# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:05:10 2023

@author: shane

Functions for getting converting a networkx graph categorical variable to a
numerical so that graph-tool handles it properly
Functions for converting a a networkx graph to a graph-tool one
Function to get the assortativity of any variable at all distances
Function to get the modularity at all distances
"""

import graph_tool.all as gt
import networkx as nx
import numpy as np


def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    if isinstance(key, bytes):
        # Encode the key as ASCII
        key = key.encode('ascii', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, str):
        tname = 'string'
        value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set() # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key  = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname) # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set() # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname) # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {} # vertex mapping for tracking edges later
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value # ep is short for edge_properties

    # Done, finally!
    return gtG


def gt_assortativity_all(g, variable = 'total'):
    """

    Parameters
    ----------
    g : gt.Graph()
        Graph on which to calculate the assortativity at a distance.
    variable:
        Can calculate the correlation for any numerical variable. The
        default, 'total', refers to the degree of each node.
    Returns
    -------
    assortativity : list
        list of assortativity values.
    n_paths : list
        list of number of pairs.

    """
    diameter, _ = gt.pseudo_diameter(g)
    
    N = g.num_vertices()                                  
    
    variable = np.array(g.degree_property_map(variable).a)
        
    shortest_dist = gt.shortest_distance(g)
    
    diameter = int(diameter)
        
    shortest_dist = shortest_dist.get_2d_array(range(N))
    
    assortativity = []
    n_paths = []
    
    for distance in range(1, diameter + 1):
        u,v = np.where(shortest_dist==distance)
    
        Ek = variable[u].mean() #average degree at the end of an edge
        Eksq = (variable[u]**2).mean() #average degree squared at end of an edge
        r = ((variable[u]-Ek)*(variable[v]-Ek)).sum()/(len(u)*(Eksq-Ek**2)) #assortativity
        assortativity.append(r)
        n_paths.append(len(u)/2)
    
    return assortativity, n_paths


def gt_modularity_all_d_new(g, atr='color', normalise=True):
    """
    
    Gets the modularity at all distances for a given graph and chosen attribute.
    ** Attribute must be numerical, e.g. for color, each color must be assigned
       a number.
    ----------
    g : gt.Graph()
        Graph for which to calculate the modularity.
    atr : String, optional
        Attribute for which to calculate the modularirt. The default is 'color'.
    normalise : Bool, optional
        Whether or not to normalise the modularity as in Yose 2017.
        The default is True.

    Returns
    -------
    list
        list of modularity values at each distance.

    """
    diameter, _ = gt.pseudo_diameter(g)

    N = g.num_vertices()
    
    shortest_dist = gt.shortest_distance(g)
    
    diameter = int(diameter)
    
    
    shortest_dist = shortest_dist.get_2d_array(range(N))
    categories = set()
    for vertex in g.vertices():
        categories.add(g.vp[atr][vertex])
    
    n_paths = []
    e_rr = [] # for each distance we have vector length of number of classes. 
                  # e_rr[i] = number of paths
                  # connecting nodes of type i to other nodes of type i
    a_r = []  # vector length of number of classes. a_r[i] = number of paths
                  # attached to nodes of type i at either end
                 
    for i in range(diameter):
        e_rr.append([0] * len(categories))
        a_r.append(0)
        n_paths.append(0)
    categories = np.array(list(categories))
    
    
    for distance in range(1, diameter + 1):

        u,v = np.where(shortest_dist==distance)
        n_paths[distance-1] = len(u)
        node_categories = g.vp['color'].a

        a_r[distance-1] = [len(np.where(node_categories[u] == category)[0]) for category in categories]
        matches = node_categories[u] == node_categories[v]
        for category in categories:
            category_match = node_categories[u] == category
            z = np.where(category_match==matches)
            cat_index = np.where(categories == category)[0][0]
            e_rr[distance-1][cat_index] = len(z[0]) 

    e_rr_normalised = []
    for sublist in e_rr:
        print(a_r, e_rr)
        e_rr_normalised.append([sublist[i]/(2*n_paths[e_rr.index(sublist)]) for i in range(len(sublist))])
    a_r_normalised = []
    for sublist in a_r:
        #a_r double counts all paths so divide by 2*n_paths
        #*** actually, e_rr, n_paths all double count, a_r quadruple counts so the
        #factor of two cancels in e_rr/n_paths, but need another factor of two for
        # a_r
        a_r_normalised.append([sublist[i]/(2*n_paths[a_r.index(sublist)]) for i in range(len(sublist))])
    Q = [0] * diameter
    for e_sublist, a_sublist in zip(e_rr_normalised, a_r_normalised):
        for i in range(len(e_sublist)):
            Q[e_rr_normalised.index(e_sublist)] += (e_sublist[i] - (a_sublist[i] ** 2))
    a_rsq = [sum([j**2 for j in i]) for i in a_r_normalised]
    
    if normalise == True:
        #Normalise as per 
        #https://pure.coventry.ac.uk/ws/portalfiles/portal/41005009/Yose2017.pdf
        Q_normalised = []
        for modularity_score, sum_squares in zip(Q, a_rsq):
            if sum_squares == 1:
                Q_normalised.append(1)
            else:    
                rho = modularity_score / (1 - sum_squares)
                if modularity_score >= 0:
                    Q_normalised.append(rho)
                if modularity_score < 0:
                    rho_min = ((-1 * sum_squares)/(1 - sum_squares))
                    Q_normalised.append(-1 * rho/rho_min)
        return Q_normalised
    else:
        return Q


def remap_attributes(G:nx.Graph(), variable:str):
    """
    Parameters
    ----------
    G : nx.Graph()
        Graph to remap.
    variable : str
        Name of categorical variable to remap.

    Returns
    -------
    G : nx.Graph()
        Graph with new node attribute corresponding to remapped variable.
    variable_map : dict
        Dictionary of categorical variable values:numeric values.

    """    
    attribute = nx.get_node_attributes(G, variable)
    values = list(set(attribute.values()))
    variable_map = {i :values.index(i) for i in values}
    new_attribute_dict = {i : variable_map[G.nodes()[i][variable]] for i in G.nodes()}
    
    nx.set_node_attributes(G, new_attribute_dict, name = variable + '_mapped')
    variable_map = {v: k for k, v in variable_map.items()}
    
    return G, variable_map
    
    

if __name__ == '__main__':

    # Create the networkx graph
    nxG = nx.Graph(name="Undirected Graph")
    nxG.add_node("v1", name="alpha", color="red")
    nxG.add_node("v2", name="bravo", color="blue")
    nxG.add_node("v3", name="charlie", color="blue")
    nxG.add_node("v4", name="hub", color="purple")
    nxG.add_node("v5", name="delta", color="red")
    nxG.add_node("v6", name="echo", color="red")

    nxG.add_edge("v1", "v2", weight=0.5, label="follows")
    nxG.add_edge("v1", "v3", weight=0.25, label="follows")
    nxG.add_edge("v2", "v4", weight=0.05, label="follows")
    nxG.add_edge("v3", "v4", weight=0.35, label="follows")
    nxG.add_edge("v5", "v4", weight=0.65, label="follows")
    nxG.add_edge("v6", "v4", weight=0.53, label="follows")
    nxG.add_edge("v5", "v6", weight=0.21, label="follows")

    for item in nxG.edges(data=True):
        print(item)

    # Convert to graph-tool graph
    gtG = nx2gt(nxG)
    gtG.list_properties()
