# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:41:31 2017

@author: ijmadrid
"""

"""
Clustering spectral
"""

import numpy as np
import sklearn.cluster as skc
import Q1 as q1

def clustering_spectral(A, k, option):
    """
    Fait le clustering spectral sur le graph definit par la matrice d'adjacence A
    NB : A est symetrique (graph non orienté)
    
    Input :     np.array([[]])  A : matrice d'adjacence
                int             k : nombre de classes
                int        option : 1 ou 2 
    """
    if option == 1:
        
        eigval, eigvec = np.linalg.eigh(A)
        
        n = len(A)
        U = np.zeros((k,n))
        
        eigvec = np.transpose(eigvec)
        
        for i in range(k):
            U[i] = eigvec[-i-1]
        
        U = np.transpose(U)
        kmeans = skc.KMeans(n_clusters=k).fit(U)        
        C = kmeans.labels_
        graph = q1.graph_from_A(A)
    
        print A
        print C
        colors = []
        for i in range(0, max(C)+1):
            colors.append('%06X' % np.random.randint(0, 0xFFFFFF))
        for vertex in graph.vs():
            vertex["color"] = str('#') + colors[C[vertex.index]]          
        q1.graph_plot(graph,"Clustering_spectral option "+str(option))   

    if option == 2:
        print '2'
    
    return C
        
testA = np.array([[1,0,1,1],
                  [0,0,1,0],
                  [1,1,1,1],
                  [1,0,1,0]])