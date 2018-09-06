# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:05:25 2017

@author: Arnaud
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#####FONCTIONS TECHNIQUES ET SECONDAIRES#####
#Partitionne les individus en deux classes, en fixant la proportion d'individus dans la première classe
def partition_irreguliere(fraction_premiere_classe):
    resultat=[]        
    for l in range(int(fraction_premiere_classe*n)):
        resultat.append(0)
    for l in range(int(fraction_premiere_classe*n),n):
        resultat.append(1)
    return(resultat)
#Partitionne les individus en deux classes de tailles égales
def partition_reguliere(nindiv,nclasses):
    assert(n%k==0)
    resultat=[]        
    for l in range(k):
        for i in range(int(n/k)):
            resultat.append(l)
    return(resultat)

#fonction de clustering spectral
def clustering_spectral(Matrice):
    valpropres,vect_propres=np.linalg.eigh(Matrice)
    vects = np.transpose(vect_propres)
    k_selectionnes=[]
    for i in range(k):
        k_selectionnes.append(vects[-1-i])
    k_selectionnes=np.transpose(k_selectionnes)
    clustering=KMeans(n_clusters=k).fit(k_selectionnes)
    etiquettes=clustering.labels_
    return(etiquettes)
#calcul de l'indice de Rand d'un clustering
def Rand(classes,clustering):
    somme=0
    for a in range(n_):
        for b in range(a+1,n_):
            x,y = classes[a],classes[b]
            if (x==y and clustering[a]==clustering[b]):
                somme+=1
            if (x!=y and clustering[a]!=clustering[b]):
                somme+=1
    return(somme*2/(n_*(n_-1)))
#calcul du critere de clustering réussi
def Critere(cin, cout, n):
    return (cin-cout-np.sqrt(np.log(n)*(cin+cout)))

def Criteremieux():
    plt.close('all')
    plt.figure(1)
    t=np.linspace(0,n/30)
    ci=n/10
    co=n/30
    vect=[]
    for a in range(50):
        vect.append(Critere(ci-t[a],co+t[a],n))
        if(abs(vect[a]-10)<0.5):
            print(ci-t[a],co+t[a])
    plt.plot(t,vect)
    return;
#Calcul des puissances à appliquer à notre facteur correctif lié au changement de probabilités
def coefficients_normalisation(N,partition):
    somme1=0
    somme2=0
    for i in range(N): #N individus dont le comportement d'amitié va être modifié
        for j in range(i+1,n):
            if (partition[i]==partition[j]):
                somme1+=1
            else:
                somme2+=1
    return(somme1,somme2)


#####PARAMETRES GENERAUX#####
k = 2
n = 200
fraction_classe_1 = 0.25
type_partition = 2
cin = 34
cout = 16
pin=cin/n
pout=cout/n
nb_simul=1000
seuil=0.54
if (type_partition==1):
    n_=int(n/2)
    Indices=partition_reguliere(n,k)
else:
    n_=int(n*fraction_classe_1)
    Indices=partition_irreguliere(fraction_classe_1)





#####SIMULATION DU MODELE#####
B=np.ones((k,k))*pout-np.diag([pout]*k)+np.diag([pin]*k)



#Génération d'une observation du modèle
# le paramètre option correspond à la méthode de clustering choisie
def Simulation_modele(option):
    X=np.random.rand(n,n)
    A=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if(X[i][j]<B[Indices[i]][Indices[j]]):
                A[i][j]=1
                A[j][i]=1
    if(option==1):    
        return(A)
    else:
        Diagonale=np.sum(A,axis=1)
        D=np.diag(np.sqrt(1/Diagonale))
        M=np.dot(np.dot(D,A),D)
        return(M)

#Génération d'une observation du modèle avec changement de proba
#les nbindivides premiers individus du graphe voient leurs probabilités changées en npin, npout

def Simulation_modele_partiel(option,matrice_stochastique,nbindividus,npin,npout):
    X=np.random.rand(n,n)
    A=np.zeros((n,n))
    facteur=1
    for i in range(n):
        for j in range(i+1,n):
            if (i<nbindividus):
                if(X[i][j]<matrice_stochastique[Indices[i]][Indices[j]]):
                    A[i][j]=1
                    A[j][i]=1
                    if (Indices[i]!=Indices[j]):
                        facteur=facteur*pout*(1-npout)/(npout*(1-pout))
                    else:
                        facteur=facteur*pin*(1-npin)/(npin*(1-pin))
            elif(X[i][j]<B[Indices[i]][Indices[j]]):
                A[i][j]=1
                A[j][i]=1
    if(option==1):    
        return(A,facteur)
    else:
        Diagonale=np.sum(A,axis=1)
        D=np.diag(np.sqrt(1/Diagonale))
        M=np.dot(np.dot(D,A),D)
        return(M,facteur)

#####METHODE DE MONTE CARLO NAIVE#####
#N simulations du modele selon la méthode de clustering donnée par option
def sommets_mal_clusterises_naive(option,N,seuil):
    X=[0]*N
    for s in range(N):
        V=clustering_spectral(Simulation_modele(option))
        if (Rand(Indices,V)<seuil):
            X[s]=1
    print(n_)
    print('Probabilite méthode naive : ', np.mean(X))
    print('critere de clustering reussi :', Critere(cin,cout,n))
    print('Intervalle de confiance de demi-largeur : ', 1.96*np.std(X)/np.sqrt(N))
    return(np.mean(X),1.96*np.std(X)/np.sqrt(N));

#sommets_mal_clusterises_naive(2,nb_simul,seuil)


######METHODE AVEC CHANGEMENT DE PROBABILITES#####
#nbindividus voient leurs probabilites modifiees
#N simulations sont effectuees
def sommets_mal_clusterises_partiel(option,ncin,ncout,nbindividus,N,seuil):
    npin=ncin/n
    npout=ncout/n
    C=np.ones((k,k))*npout-np.diag([npout]*k)+np.diag([npin]*k)
    coef1,coef2=coefficients_normalisation(nbindividus,Indices)
    Facteur_normalisation=np.power((1-pin)/(1-npin),coef1)*np.power((1-pout)/(1-npout),coef2)
    X=[0]*N
    somme_globale=0
    for s in range(N):
        tableau,x=Simulation_modele_partiel(option,C,nbindividus,npin,npout)
        V=clustering_spectral(tableau)
        if (Rand(Indices,V)<seuil):
            X[s]=x
            somme_globale+=1
    X=np.array(X)
    X=Facteur_normalisation*X
    print('Nombre de fois que notre simulation biaisee a mal clusterise un nombre \' important \' de sommets : ',somme_globale)
    print('Nouveau parametre intraclasse choisi : ', ncin)
    print('Nouveau parametre hors classe choisi : ', ncout)
    print('critere de clustering reussi nouveaux parametres :', (ncin-ncout)-np.sqrt(np.log(n)*(ncin+ncout)))
    print('Probabilite estimée par méthode de fixation de nouvelles probas : ', np.mean(X))
    print('Intervalle de confiance de demi-largeur : ', 1.96*np.std(X)/np.sqrt(N))
    return (np.mean(X),(ncin-ncout)-np.sqrt(np.log(n)*(ncin+ncout)),1.96*np.std(X)/np.sqrt(N));

#sommets_mal_clusterises_partiel(2,34,16,5,nb_simul,seuil)

#print(Critere(34,16,n))  #critère presque pas vérifié

#####Démarche#####      
#1 seuil ?
def testRand(option,N):
    plt.close(2)
    plt.figure(2)
    valeurs_Rand=[]
    for s in range(N):
        tableau=Simulation_modele(option)
        V=clustering_spectral(tableau)
        valeurs_Rand.append(Rand(Indices,V))
    abscisses=range(N)
    plt.plot(abscisses,valeurs_Rand)

    return (valeurs_Rand);

#testRand(2,1000)

#2 Recherche changement
def Question21(option):
    plt.close('all')
    plt.figure(1)
    proba_estimees=[]
    ci=34
    co=16
    abscisse=range(10)
    for i in range(10):
        ci=ci-0.1
        co=co+0.1
        p,cr,var=sommets_mal_clusterises_partiel(option,ci,co,5,nb_simul,seuil)
        proba_estimees.append(p/var)        
    plt.plot(abscisse,proba_estimees)
    return(0);

#Question21(2)

def Question22(option):
    plt.close('all')
    plt.figure(1)
    proba_estimees=[]
    ci=33.82
    co=16.18
    abscisse=range(5)
    for i in range(5):
        ci=ci-0.02
        co=co+0.02
        p,cr,var=sommets_mal_clusterises_partiel(option,ci,co,5,nb_simul,seuil)
        proba_estimees.append(p/var)        
    plt.plot(abscisse,proba_estimees)
    return(0);

#Question22(2)

#3 Conclusion
def Question23(option):
    proba_estimees_classique=[]
    demi_largeur_classique=[]
    proba_estimees_is=[]
    demi_largeur_is=[]
    for i in range(10):
        p,var=sommets_mal_clusterises_naive(option,nb_simul, seuil)
        proba_estimees_classique.append(p)
        demi_largeur_classique.append(var)
    ci=33.76
    co=16.24
    for i in range(10):
        p,cr,var=sommets_mal_clusterises_partiel(option,ci,co,5,nb_simul, seuil)
        proba_estimees_is.append(p)
        demi_largeur_is.append(var)      
    print('classique proba', np.mean(proba_estimees_classique), ' ', np.std(proba_estimees_classique))
    print('classique demi-largeur',np.mean(demi_largeur_classique), ' ', np.std(demi_largeur_classique))
    print('IS proba', np.mean(proba_estimees_is), ' ', np.std(proba_estimees_is))
    print('IS demi-largeur',np.mean(demi_largeur_is), ' ', np.std(demi_largeur_is))
    return(0);
    
#Question23(2)    



