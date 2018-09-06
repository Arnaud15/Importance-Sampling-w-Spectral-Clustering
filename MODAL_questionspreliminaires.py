#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 14:10:26 2017

@author: ijmadrid
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from igraph import *
import igrap_plottools as q1

def partition_irreguliere(n,fraction_premiere_classe):
    """
    Input :     n : nombre de sommets
                fraction_premiere_classe : % d'indv. de la premiere classe
                (NB: k = 2)
                
    Ouput :     vecteur de classes
    """
    resultat=[]        
    for l in range(int(fraction_premiere_classe*n)):
        resultat.append(0)
    for l in range(int(fraction_premiere_classe*n),n):
        resultat.append(1)
    return(resultat)

def partition_reguliere(n,k):
    """
    Input :     n : nombre de sommets
                k : nombre de classes
                
    Ouput :     vecteur de classes
    """
    assert(n%k==0)
    resultat=[]        
    for l in range(k):
        for i in range(int(n/k)):
            resultat.append(l)
    return(resultat)

def stochastic_block_model(n,C,B):
    """
    Input :     n : nombre de sommets
                C : Vecteur classes (C[k] = classe du sommet k)
                B : Matrice de probas (inter et intra clusters)
                option : 1 ou 2
                
    Ouput :     A : Matrice d'adjacence du graphe géneré par B
    """
    X=np.random.rand(n,n)
    A = np.zeros((n,n))    
    for i in range(n):
        for j in range(i+1,n):
            if(X[i][j]<B[C[i]][C[j]]):
                A[i][j]=1
                A[j][i]=1
#    if(option==1):    
#        return(A)
#    else:
#        Diagonale=np.sum(A,axis=1)
#        with np.errstate(divide='print'):
#            preD = (np.sqrt(1./Diagonale))
#        preD[~np.isfinite(preD)] = 0.
#        D=np.diag(preD)
#        M=np.dot(np.dot(D,A),D)
#        return(M)
    return A


def clustering_spectral(matrice,k,option,plot=False,couleurs=None,savefilename="Clustering_spectral plot"):
    A = matrice
    if(option == 2):
        Diagonale=np.sum(matrice,axis=1)
        with np.errstate(divide='warn'):
            preD = (np.sqrt(1./Diagonale))
        preD[~np.isfinite(preD)] = 0.
        D=np.diag(preD)
        M=np.dot(np.dot(D,matrice),D)
        matrice = M
    valpropres,vect_propres=np.linalg.eigh(matrice)
    vects = np.transpose(vect_propres)
    k_selectionnes=[]
    for i in range(k):
        k_selectionnes.append(vects[-1-i])
    k_selectionnes=np.transpose(k_selectionnes)
    clustering=KMeans(n_clusters=k).fit(k_selectionnes)
    etiquettes=clustering.labels_

    if(plot):
        if(couleurs is None):
            couleurs = etiquettes
        graph = q1.graph_from_A(A)
        colors = []
        for i in range(0, max(couleurs)+1):
            colors.append('%06X' % np.random.randint(0, 0xFFFFFF))
        for vertex in graph.vs():
            vertex["color"] = str('#') + colors[couleurs[vertex.index]]     
        q1.graph_plot(graph,savefilename)  
    
    return(etiquettes)




