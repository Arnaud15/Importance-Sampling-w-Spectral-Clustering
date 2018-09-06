# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:17:57 2017

@author: ijmadrid
"""

from igraph import *
import numpy as np


def graph_from_A(A):
    """
    Input :     A : Matrice d'adjacence
    
    Output :    igraph.Graph object
    """
    nV = len(A)
    g = Graph()
    g.add_vertices(nV)
    g.vs["label"]=range(nV)
    
    for i in range(nV):
        for j in range(nV):
            if (A[i,j] == 1 and i <= j):
                g.add_edges([(i,j)])
    
    return g

def graph_plot(g,savefile_name):
    layout = g.layout("kk")
    plot(g, savefile_name+"_graph.png", layout = layout)
    

def visualisation(adjacence_loadfile, classes_loadfile):
    A = np.load(adjacence_loadfile)
    C = np.load(classes_loadfile)

    graph = graph_from_A(A)
    
    colors = []
    for i in range(0, max(C)+1):
        colors.append('%06X' % np.random.randint(0, 0xFFFFFF))
    for vertex in graph.vs():
        vertex["color"] = str('#') + colors[C[vertex.index]]
    
    graph_plot(graph,classes_loadfile+"_plot_")