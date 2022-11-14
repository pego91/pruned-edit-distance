import numpy as np
from copy import deepcopy
import time

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core.util import sum_product,quicksum

import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
# logging.getLogger('pyomo.core').setLevel(logging.WARNING)

from Trees_OPT import Tree
from itertools import product
import multiprocessing as mp

from sympy import *


T_build=0
T_solve=0


def calculate_Wxy_pool(LIST):
    
    [T_0,T_1,W,x,y,SUB_0,SUB_1,W_0,W_1,V_0,V_1,return_map]=LIST
    
    T_x=SUB_0[x]
    T_y=SUB_1[y]
    
    W_aux=np.zeros((T_x.dim-1,T_y.dim-1)).astype(np.float)
    
#     available_x=set(T_x.vertices).copy()-set(np.where(T_x.name_vertices==x)[0])
#     available_y=set(T_y.vertices).copy()-set(np.where(T_y.name_vertices==y)[0])

#     available_x = np.sort(list(available_x))
#     available_y = np.sort(list(available_y))
    
#     available_x_aux=set(T_x.name_vertices).copy()-set([x])
#     available_y_aux=set(T_y.name_vertices).copy()-set([y])

#     available_x_aux = np.sort(list(available_x_aux))
#     available_y_aux = np.sort(list(available_y_aux))

    available_x = T_x.vertices[:-1]
    available_x_aux = T_x.name_vertices[:-1]

    available_y = T_y.vertices[:-1]
    available_y_aux = T_y.name_vertices[:-1]

    for i,v0 in enumerate(available_x_aux):
        for j,v1 in enumerate(available_y_aux):
            W_aux[i,j]=W[v0,v1]
    
#     if x==20 and y==21:
#         print('\nI nomi sono: ',T_x.name_vertices,T_y.name_vertices, available_x, available_y)
#         print('\nQueste sono le due matrici ad ora: ', W[15,19],'\n',W_aux)
        
    if len(available_x)*len(available_y)>0:
        new_W,M=make_model(x,y,T_x,T_y,W_aux,available_x,available_y,T_x.dim-1,T_y.dim-1,
                         W_0,W_1,V_0,V_1,True,False,return_map)
        return new_W,M
    elif len(available_x)>0:
        M={}
        return W_0[x],M
    elif len(available_y)>0:
        M={}
        return W_1[y],M
    else:
        M={}
        return 0,M
    
def calculate_Wxy(T_0,T_1,W,x,y,SUB_0,SUB_1,W_0,W_1,V_0,V_1,return_map):
    
    T_x=SUB_0[x]
    T_y=SUB_1[y]
    
    W_aux=np.zeros((T_x.dim-1,T_y.dim-1)).astype(np.float)
    
    avaiable_x=set(T_x.vertices).copy()-set(np.where(T_x.name_vertices==x)[0])
    avaiable_y=set(T_y.vertices).copy()-set(np.where(T_y.name_vertices==y)[0])

    available_x = T_x.vertices[:-1]
    available_x_aux = T_x.name_vertices[:-1]

    available_y = T_y.vertices[:-1]
    available_y_aux = T_y.name_vertices[:-1]
    
    for i,v0 in enumerate(available_x_aux):
        for j,v1 in enumerate(available_y_aux):
            W_aux[i,j]=W[v0,v1]
 
    if len(available_x)*len(available_y)>0:
        new_W,M=make_model(x,y,T_x,T_y,W_aux,available_x,available_y,T_x.dim-1,T_y.dim-1,
                         W_0,W_1,V_0,V_1,True,False,return_map)
        return new_W,M
    elif len(available_x)>0:
        M={}
        return W_0[x],M
    elif len(available_y)>0:
        M={}
        return W_1[y],M
    else:
        M={}
        return 0,M


def sym_delta(n,W,paths,len_paths,wmax,delta):     
    """
    W sono le molteplicità dell'albero: un dizionario con chiavi [v,father(v)]
    """ 
    w=np.vstack([W[(paths[n,k],paths[n,k+1])] for k in range(len_paths[n]-1)])   

    """
    w quindi è un listone di funzioni; le faccio mettere una sotto l'altra in una matrice.
    Ora per ogni riga della matrice devo affiancarle
    """
    N = w.shape[1]
    
    for i in np.arange(1,w.shape[0]):
        a = w[i-1,:]
        b = w[i,:]
        aux = a+b
    
        w[i,:] = aux
            
    delta=[w[i,:] for i in np.arange(0,w.shape[0])]          
    return delta
        
def sym_objective_fun(x,y,n,m,W,L,wmax,delta,
                      mult_x,W_x,paths_x,len_paths_x,
                      mult_y,W_y,paths_y,len_paths_y,
                      W_0,W_1,V_0,V_1,root=False): 
    
    p = 1
    
    """
    Memorizza per un albero, per ogni vertice, la lunghezza del ramo ottenuta 
    ghostando i vertici sopra di lui    
    """
    D={}
    D_N_num={}
    D_M_num={}    
    
    """
    Somma per ogni vertice di un albero, tutte le variabili sue e
    degli altri vertici che passano per lui. Facendo attenzione
    che se ghosto uno sopra di lui, deve essere ghostato anche lui. 
    Se alla fine è 1 viene ghostato    
    """
    G_x={}
    G_y={}
    """
    Somma per ogni vertice dell'albero, tutte le variabili di quell'albero 
    che riguardano il suo assegnamento: 
    alla fine sarà zero se il punto non è assegnato, oppure 1    
    """
    S_x={}
    S_y={}
    
    for i in range(n-int(root)):  
        S_x[i] = sum([L[i,j,aux_x,aux_y]  \
                     for j in range(m-int(root))\
                     for aux_x in range(len_paths_x[i]-1)\
                     for aux_y in range(len_paths_y[j]-1)\
                     ])
    
        d = sym_delta(i,mult_x,paths_x,len_paths_x,wmax,delta)        
        D_N_num[i]=d

    for j in range(m-int(root)):  
        S_y[j] = sum([L[i,j,aux_x,aux_y]  \
                     for i in range(n-int(root))\
                     for aux_x in range(len_paths_x[i]-1)\
                     for aux_y in range(len_paths_y[j]-1)\
                     ])

        d = sym_delta(j,mult_y,paths_y,len_paths_y,wmax,delta)        
        D_M_num[j]=d


    for i in range(n-int(root)):          
        for j in range(m-int(root)):
            for aux_x in range(len_paths_x[i]-1):
                for aux_y in range(len_paths_y[j]-1):
                    D[i,j,aux_x,aux_y] = \
                        np.linalg.norm(D_N_num[i][aux_x]-D_M_num[j][aux_y], ord = p)*delta*\
                            L[i,j,aux_x,aux_y]
      
    out=0
    
    for i in range(n-int(root)):
        aux=np.where(paths_x==i)   
        
        aux_0=aux[0]
        aux_1=aux[1]
        AUX=0
        
        for t in range(len(aux_0)):
            AUX+=sum([L[aux_0[t],j,k,aux_y] \
                      for j in range(m-int(root))\
                      for aux_y in range(len_paths_y[j]-1)
                      for k in np.arange(aux_1[t],len_paths_x[aux_0[t]]-1)])
            
        G_x[i]=AUX
            
    for j in range(m-int(root)):
        aux=np.where(paths_y==j)        
        aux_0=aux[0]
        aux_1=aux[1]
        AUX=0
        
        for t in range(len(aux_0)):
            AUX+=sum([L[i,aux_0[t],aux_x,k] \
                      for i in range(n-int(root))\
                      for aux_x in range(len_paths_x[i]-1)
                      for k in np.arange(aux_1[t],len_paths_y[aux_0[t]]-1)])
         
        G_y[j]=AUX
        
    """
    Questo dovrebbere essere il costo di tutto quello che devo contrarre.
    Contraggo tutti quelli che non sono ghostati ne assegnati. 
    Però poi devo togliere la massa che c'è sotto i vertici che assegno
    perchè quella è già contenuta nelle iterazioni precedenti        
    """

    for i in range(n-int(root)):
        out+=(1-G_x[i])*np.linalg.norm(mult_x[(i,paths_x[i,1])], ord = p)*delta
        out+=-W_0[V_0[x][i]]*S_x[i]
        
    for j in range(m-int(root)):
        out+=(1-G_y[j])*np.linalg.norm(mult_y[(j,paths_y[j,1])], ord = p)*delta
        out+=-W_1[V_1[y][j]]*S_y[j]   
        
    SUM={}
    """    
    Questo è il costo di trasformare gli alberi sotto i punti assegnati!
    """
    for i in range(n):
        for j in range(m):
            SUM[i,j]=sum([L[i,j,aux_x,aux_y] \
                         for aux_x in range(len_paths_x[i]-1)\
                         for aux_y in range(len_paths_y[j]-1)])

            out+=SUM[i,j]*W[i,j]

    for i in range(n-int(root)):        
        for j in range(m-int(root)):
            for aux_x in range(len_paths_x[i]-1):            
                for aux_y in range(len_paths_y[j]-1):            
                    out+=(D[i,j,aux_x,aux_y]) 
       
    return out

def make_poly(x,y,n,m,W,L,wmax,delta,
              mult_x,W_x,paths_x,len_paths_x,
              mult_y,W_y,paths_y,len_paths_y,
              W_0,W_1,V_0,V_1,root=False):
    
    out=sym_objective_fun(x,y,n,m,W,L,wmax,delta,
                          mult_x,W_x,paths_x,len_paths_x,
                          mult_y,W_y,paths_y,len_paths_y,
                          W_0,W_1,V_0,V_1,root)    
    return out


def eval_objective_fun(x,y,n,m,W,L,wmax,delta,
                      mult_x,W_x,paths_x,len_paths_x,
                      mult_y,W_y,paths_y,len_paths_y,
                      W_0,W_1,V_0,V_1,root=False): 
    
    p = 1
    
    """
    Memorizza per un albero, per ogni vertice, la lunghezza del ramo ottenuta 
    ghostando i vertici sopra di lui    
    """
    D={}
    D_N_num={}
    D_M_num={}    
    
    """
    Somma per ogni vertice di un albero, tutte le variabili sue e
    degli altri vertici che passano per lui. Facendo attenzione
    che se ghosto uno sopra di lui, deve essere ghostato anche lui. 
    Se alla fine è 1 viene ghostato    
    """
    G_x={}
    G_y={}
    """
    Somma per ogni vertice dell'albero, tutte le variabili di quell'albero 
    che riguardano il suo assegnamento: 
    alla fine sarà zero se il punto non è assegnato, oppure 1    
    """
    S_x={}
    S_y={}
    
    for i in range(n-int(root)):  
        S_x[i] = sum([L[i,j,aux_x,aux_y]  \
                     for j in range(m-int(root))\
                     for aux_x in range(len_paths_x[i]-1)\
                     for aux_y in range(len_paths_y[j]-1)\
                     ])
    
        d = sym_delta(i,mult_x,paths_x,len_paths_x,wmax,delta)        
        D_N_num[i]=d

    for j in range(m-int(root)):  
        S_y[j] = sum([L[i,j,aux_x,aux_y]  \
                     for i in range(n-int(root))\
                     for aux_x in range(len_paths_x[i]-1)\
                     for aux_y in range(len_paths_y[j]-1)\
                     ])

        d = sym_delta(j,mult_y,paths_y,len_paths_y,wmax,delta)        
        D_M_num[j]=d


    for i in range(n-int(root)):          
        for j in range(m-int(root)):
            for aux_x in range(len_paths_x[i]-1):
                for aux_y in range(len_paths_y[j]-1):
                    D[i,j,aux_x,aux_y] = \
                        np.linalg.norm(D_N_num[i][aux_x]-D_M_num[j][aux_y], ord = p)*delta*\
                            L[i,j,aux_x,aux_y]
      
    out=0
    
    for i in range(n-int(root)):
        aux=np.where(paths_x==i)   
        
        aux_0=aux[0]
        aux_1=aux[1]
        AUX=0
        
        for t in range(len(aux_0)):
            AUX+=sum([L[aux_0[t],j,k,aux_y] \
                      for j in range(m-int(root))\
                      for aux_y in range(len_paths_y[j]-1)
                      for k in np.arange(aux_1[t],len_paths_x[aux_0[t]]-1)])
            
        G_x[i]=AUX
            
    for j in range(m-int(root)):
        aux=np.where(paths_y==j)
        
        aux_0=aux[0]
        aux_1=aux[1]
        AUX=0
        
        for t in range(len(aux_0)):
            AUX+=sum([L[i,aux_0[t],aux_x,k] \
                      for i in range(n-int(root))\
                      for aux_x in range(len_paths_x[i]-1)
                      for k in np.arange(aux_1[t],len_paths_y[aux_0[t]]-1)])
         
        G_y[j]=AUX
        
    """
    Questo dovrebbere essere il costo di tutto quello che devo contrarre.
    Contraggo tutti quelli che non sono ghostati ne assegnati. 
    Però poi devo togliere la massa che c'è sotto i vertici che assegno
    perchè quella è già contenuta nelle iterazioni precedenti        
    """

    out_del = 0
    out_sub = 0
    out_shrink = 0
    
    for i in range(n-int(root)):
        out+=(1-G_x[i])*np.linalg.norm(mult_x[(i,paths_x[i,1])], ord = p)*delta
        out+=-W_0[V_0[x][i]]*S_x[i]
        out_del+=(1-G_x[i])*np.linalg.norm(mult_x[(i,paths_x[i,1])], ord = p)*delta
        out_del+=-W_0[V_0[x][i]]*S_x[i]        
        
        
    for j in range(m-int(root)):
        out+=(1-G_y[j])*np.linalg.norm(mult_y[(j,paths_y[j,1])], ord = p)*delta
        out+=-W_1[V_1[y][j]]*S_y[j]   
        out_del+=(1-G_y[j])*np.linalg.norm(mult_y[(j,paths_y[j,1])], ord = p)*delta
        out_del+=-W_1[V_1[y][j]]*S_y[j]   
        
    SUM={}
    """    
    Questo è il costo di trasformare gli alberi sotto i punti assegnati!
    """
    for i in range(n):
        for j in range(m):
            SUM[i,j]=sum([L[i,j,aux_x,aux_y] \
                         for aux_x in range(len_paths_x[i]-1)\
                         for aux_y in range(len_paths_y[j]-1)])

            out+=SUM[i,j]*W[i,j]
            out_sub+=SUM[i,j]*W[i,j]

    for i in range(n-int(root)):        
        for j in range(m-int(root)):
            for aux_x in range(len_paths_x[i]-1):            
                for aux_y in range(len_paths_y[j]-1):            
                    out+=(D[i,j,aux_x,aux_y]) 
                    out_shrink+=(D[i,j,aux_x,aux_y]) 
                    
#     if x == 20 and y == 21:
#         print('\nDelta: ', delta)
#         print('\nCosti abbinamenti: ',D,'\nCosti alberi già fatti: ',W,'\nghost: ',G_x,'\n',G_y,
#              '\nAssociati: ',S_x,'\n',S_y,'\nCosti sub: ',W_0,'\n',W_1)
#         print('\nCosti calcolati: deletion ',out_del,'\nsub ', out_sub,'\nshrink ',out_shrink)
       
    return out


def eval_mapping(x,y,n,m,W,L,wmax,delta,
                      mult_x,W_x,paths_x,len_paths_x,
                      mult_y,W_y,paths_y,len_paths_y,
                      W_0,W_1,V_0,V_1,root=False): 
    
    p = 1
    
    """
    Memorizza per un albero, per ogni vertice, la lunghezza del ramo ottenuta 
    ghostando i vertici sopra di lui    
    """
    D={}
    D_N_num={}
    D_M_num={}    
    
    """
    Somma per ogni vertice di un albero, tutte le variabili sue e
    degli altri vertici che passano per lui. Facendo attenzione
    che se ghosto uno sopra di lui, deve essere ghostato anche lui. 
    Se alla fine è 1 viene ghostato    
    """
    G_x={}
    G_y={}
    """
    Somma per ogni vertice dell'albero, tutte le variabili di quell'albero 
    che riguardano il suo assegnamento: 
    alla fine sarà zero se il punto non è assegnato, oppure 1    
    """
    S_x={}
    S_y={}
    
    for i in range(n-int(root)):  
        S_x[i] = sum([L[i,j,aux_x,aux_y]  \
                     for j in range(m-int(root))\
                     for aux_x in range(len_paths_x[i]-1)\
                     for aux_y in range(len_paths_y[j]-1)\
                     ])
    
        d = sym_delta(i,mult_x,paths_x,len_paths_x,wmax,delta)        
        D_N_num[i]=d

    for j in range(m-int(root)):  
        S_y[j] = sum([L[i,j,aux_x,aux_y]  \
                     for i in range(n-int(root))\
                     for aux_x in range(len_paths_x[i]-1)\
                     for aux_y in range(len_paths_y[j]-1)\
                     ])

        d = sym_delta(j,mult_y,paths_y,len_paths_y,wmax,delta)        
        D_M_num[j]=d


    for i in range(n-int(root)):          
        for j in range(m-int(root)):
            for aux_x in range(len_paths_x[i]-1):
                for aux_y in range(len_paths_y[j]-1):
                    D[i,j,aux_x,aux_y] = \
                        np.linalg.norm(D_N_num[i][aux_x]-D_M_num[j][aux_y], ord = p)*delta*\
                            L[i,j,aux_x,aux_y]
      
    out=0
    
    for i in range(n-int(root)):
        aux=np.where(paths_x==i)   
        
        aux_0=aux[0]
        aux_1=aux[1]
        AUX=0
        
        for t in range(len(aux_0)):
            AUX+=sum([L[aux_0[t],j,k,aux_y] \
                      for j in range(m-int(root))\
                      for aux_y in range(len_paths_y[j]-1)
                      for k in np.arange(aux_1[t],len_paths_x[aux_0[t]]-1)])
            
        G_x[i]=AUX
            
    for j in range(m-int(root)):
        aux=np.where(paths_y==j)        
        aux_0=aux[0]
        aux_1=aux[1]
        AUX=0
        
        for t in range(len(aux_0)):
            AUX+=sum([L[i,aux_0[t],aux_x,k] \
                      for i in range(n-int(root))\
                      for aux_x in range(len_paths_x[i]-1)
                      for k in np.arange(aux_1[t],len_paths_y[aux_0[t]]-1)])
         
        G_y[j]=AUX
        
    """
    Questo dovrebbere essere il costo di tutto quello che devo contrarre.
    Contraggo tutti quelli che non sono ghostati ne assegnati. 
    Però poi devo togliere la massa che c'è sotto i vertici che assegno
    perchè quella è già contenuta nelle iterazioni precedenti        
    """

    out_del = 0
    out_sub = 0
    out_shrink = 0
    
    for i in range(n-int(root)):
        out+=(1-G_x[i])*np.linalg.norm(mult_x[(i,paths_x[i,1])], ord = p)*delta
        out+=-W_0[V_0[x][i]]*S_x[i]
        out_del+=(1-G_x[i])*np.linalg.norm(mult_x[(i,paths_x[i,1])], ord = p)*delta
        out_del+=-W_0[V_0[x][i]]*S_x[i]        
        
        
    for j in range(m-int(root)):
        out+=(1-G_y[j])*np.linalg.norm(mult_y[(j,paths_y[j,1])], ord = p)*delta
        out+=-W_1[V_1[y][j]]*S_y[j]   
        out_del+=(1-G_y[j])*np.linalg.norm(mult_y[(j,paths_y[j,1])], ord = p)*delta
        out_del+=-W_1[V_1[y][j]]*S_y[j]   
        
    SUM={}
    """    
    Questo è il costo di trasformare gli alberi sotto i punti assegnati!
    """
    for i in range(n):
        for j in range(m):
            SUM[i,j]=sum([L[i,j,aux_x,aux_y] \
                         for aux_x in range(len_paths_x[i]-1)\
                         for aux_y in range(len_paths_y[j]-1)])

            out+=SUM[i,j]*W[i,j]
            out_sub+=SUM[i,j]*W[i,j]

    for i in range(n-int(root)):        
        for j in range(m-int(root)):
            for aux_x in range(len_paths_x[i]-1):            
                for aux_y in range(len_paths_y[j]-1):            
                    out+=(D[i,j,aux_x,aux_y]) 
                    out_shrink+=(D[i,j,aux_x,aux_y]) 
                    
    """
    D -> segmenti accoppiati tra di loro
    S -> punti accoppiati in M^*
    G -> ghosted o meno
    """   
    return D, S_x, S_y, G_x, G_y


def make_model(x,y,T_x,T_y,W,avaiable_x,avaiable_y,n,m,
               W_0,W_1,V_0,V_1,no_r, root=False, return_map = False):
    
    t0 = time.time()
    
    cost = pyo.ConcreteModel()
    cost.n=n
    cost.m=m
    
    cost.L = pyo.Var(np.arange(cost.n),np.arange(cost.m),
                     np.arange(cost.n),np.arange(cost.m), 
                     domain=pyo.Binary,initialize=0)
    
#    for i in avaiable_x:
#        for j in avaiable_y:
#            for aux_x in np.arange(T_x.len_paths[i]-1,cost.n,1):
#                for aux_y in np.arange(T_y.len_paths[j]-1,cost.m,1):    
#                    if x==5 and y==5:                    
#                        print(i,j,aux_x,aux_y)
#                    cost.L[i,j,aux_x,aux_y].fixed=True

    for i in avaiable_x:
        for j in np.arange(cost.m):
            for aux_x in np.arange(T_x.len_paths[i]-1,cost.n,1):
                for aux_y in np.arange(cost.m):    
                    cost.L[i,j,aux_x,aux_y].fixed=True

    for i in np.arange(cost.n):
        for j in avaiable_y:
            for aux_x in np.arange(cost.n):
                for aux_y in np.arange(T_y.len_paths[j]-1,cost.m,1):    
                    cost.L[i,j,aux_x,aux_y].fixed=True



    def objective_poly(cost):        
        out=make_poly(x,y,n,m,W,cost.L,T_x.wmax,T_x.delta,
                       T_x.mult,T_x.weights,T_x.paths,T_x.len_paths,
                       T_y.mult,T_y.weights,T_y.paths,T_y.len_paths,
                       W_0,W_1,V_0,V_1,root)
        return out



    cost.obj=pyo.Objective(rule=objective_poly, sense=pyo.minimize)

    cost.costr=pyo.ConstraintList()
    make_costraints(cost,T_x,T_y,no_r)


##    options="threads=1,mip_strategy_file=3, mip_limits_treememory=2048,workmem=6000"
#    options="threads=7,mip_strategy_file=3,emphasis_memory=1"
##    options="threads=7,mip_strategy_file=3"

#    options="workdir=/mnt/d/Documents/TDA/Aneurisk/Trees_Library/"

        
#    solver = pyo.SolverFactory('gurobi',solver_io="python")
#    solver = pyo.SolverFactory('couenne')
#    S=solver.solve(cost)
#    solver = pyo.SolverFactory('gurobi',
#                               executable='/home/pego/gurobi900/bin/gurobi_cl')


#     solver = pyo.SolverFactory('glpk')
    solver = pyo.SolverFactory('cplex', 
                              executable='/mnt/d/Documents/CPLEX_students/cplex/bin/x86-64_linux/cplex')

    t1 = time.time()

    S=solver.solve(cost)
    
    
    t2 = time.time()
    
    global T_build
    global T_solve    
    
    T_build = T_build + t1-t0
    T_solve = T_solve + t2-t1

    
#    solver.options['workmem']= 6000
#    solver.options['mip strategy file']=3
#    solver.options['mip limits tree memory']=2048
#    solver.options['emphasis memory']=1
#    solver.options['threads']=1
#    solver.options['optimalitytarget']=1
#    S=solver.solve(cost,
#                   options_string = options,
#                   tee=False)        

#    try:
#        S=solver.solve(cost)
#    except:
#        cost.pprint()


#    print(x,y)
#    if x>=14 and y>=13:
#        
##        cost.pprint()
#        L_eval=np.ones((cost.n,cost.m,cost.n, cost.m))
#    
#        for i in avaiable_x:
#            for j in np.arange(cost.m):
#                for aux_x in np.arange(T_x.len_paths[i]-1,cost.n,1):
#                    for aux_y in np.arange(cost.m):    
#                        L_eval[i,j,aux_x,aux_y]=0
#    
#        for i in np.arange(cost.n):
#            for j in avaiable_y:
#                for aux_x in np.arange(cost.n):
#                    for aux_y in np.arange(T_y.len_paths[j]-1,cost.m,1):    
#                        L_eval[i,j,aux_x,aux_y]=0
#        
#        print(cost.obj(), np.sum(L_eval))
#        
    new_W = cost.obj()

#     L_eval=np.ones((cost.n,cost.m,cost.n, cost.m))

#     for i in avaiable_x:
#         for j in np.arange(cost.m):
#             for aux_x in np.arange(T_x.len_paths[i]-1,cost.n,1):
#                 for aux_y in np.arange(cost.m):    
#                     L_eval[i,j,aux_x,aux_y]=0

#     for i in np.arange(cost.n):
#         for j in avaiable_y:
#             for aux_x in np.arange(cost.n):
#                 for aux_y in np.arange(T_y.len_paths[j]-1,cost.m,1):    
#                     L_eval[i,j,aux_x,aux_y]=0    

    M = {}

    if return_map:
        L_eval=np.zeros((cost.n,cost.m,cost.n, cost.m))
        for key,val in cost.L.extract_values().items():
            L_eval[key]=val
        
        D,S_x,S_y,G_x,G_y = eval_mapping(x,y,n,m,W,L_eval,T_x.wmax,T_x.delta,
                       T_x.mult,T_x.weights,T_x.paths,T_x.len_paths,
                       T_y.mult,T_y.weights,T_y.paths,T_y.len_paths,
                       W_0,W_1,V_0,V_1,root)
        
        names_x = V_0[x] 
        names_y = V_1[y] 
        coupled = np.argwhere(L_eval>0)
        remaining_x = [np.max(T_x.vertices)] 
        remaining_y = [np.max(T_y.vertices)]
        
        T_x_tmp = deepcopy(T_x) 
        T_x_tmp.name_vertices = T_x.vertices 

        T_y_tmp = deepcopy(T_y) 
        T_y_tmp.name_vertices = T_y.vertices 

        for couple in coupled:
            
            x_ = couple[0]
            y_ = couple[1]           
            couple_aux = np.copy(couple)
            couple_aux[0] = names_x[x_]
            couple_aux[1] = names_y[y_]
            M[tuple(couple_aux)]=D[couple[0],couple[1],couple[2],couple[3]]
            
            remaining_x = remaining_x + list(T_x_tmp.sub_tree(x_).name_vertices)
            remaining_y = remaining_y + list(T_y_tmp.sub_tree(y_).name_vertices)
        
        
        for p in G_x.keys():
            if G_x[p]>0 and S_x[p]==0:
                M[(names_x[p],'G')] = 0
            
        for p in G_y.keys():
            if G_y[p]>0 and S_y[p]==0:
                M[('G',names_y[p])] = 0
        
        for p in T_x.vertices:
            if (p not in remaining_x and G_x[p]==0):
                M[(names_x[p],'D')]=T_x.norms_mult[p]
                
        for p in T_y.vertices:
            if (p not in remaining_y and G_y[p]==0):
                M[('D',names_y[p])]=T_y.norms_mult[p]

#     if x==15 and y==19:
        
#         print('Sotto alberi 15-19', new_W)

#     if (x==20 and y==21):
        
#         print('Sotto alberi 15-19', W.shape, W[3,3])
        
#         L_eval=np.zeros((cost.n,cost.m,cost.n, cost.m))
              
#         for key,val in cost.L.extract_values().items():
#             L_eval[key]=val
       
        
#         new_W_eval = eval_objective_fun(x,y,n,m,W,L_eval,T_x.wmax,T_x.delta,
#                        T_x.mult,T_x.weights,T_x.paths,T_x.len_paths,
#                        T_y.mult,T_y.weights,T_y.paths,T_y.len_paths,
#                        W_0,W_1,V_0,V_1,root)  
        
#         print(x,y,'\nSoluzioni: ', np.argwhere(L_eval>0),'\nCosto: ',new_W,'\nCosto Eval: ',new_W_eval)
        
#         L_eval=np.zeros((cost.n,cost.m,cost.n,cost.m))

#         L_eval[0,0,0,0]=1
#         L_eval[3,3,0,0]=1

#         new_W_0 = eval_objective_fun(x,y,n,m,W,L_eval,T_x.wmax,T_x.delta,
#                        T_x.mult,T_x.weights,T_x.paths,T_x.len_paths,
#                        T_y.mult,T_y.weights,T_y.paths,T_y.len_paths,
#                        W_0,W_1,V_0,V_1,root)

#         print(x,y,'\nSoluzioni: ', np.argwhere(L_eval>0),'\nCosto Eval Nuovo: ',new_W_0,'\nCosto Eval Vecchio: ',new_W_eval)

    return new_W,M

#@njit
# def calculate_opt(T_0,T_1,W,W_0,W_1,V_0,V_1):

#     avaiable_0=T_0.name_vertices
#     avaiable_1=T_1.name_vertices
    
#     new_W=make_model(max(avaiable_0),max(avaiable_1,),
#                      T_0,T_1,W,avaiable_0,avaiable_1,T_0.dim,T_1.dim,
#                      W_0,W_1,V_0,V_1,False,True)

    return new_W

def make_costraints(model,T_x,T_y,no_r,rows=True):
    """
    Devo mettere queste condizioni:
    1) per ogni punto, uno solo dei suoi l_x[i] può essere assegnato
    2) lungo un percorso da foglia a radice, posso avere al massimo un punto assegnato.
       Così metto anche che posso assegnarlo a massimo un punto!
    3) bisogna mettere qualcosa che regoli i ghost! se un punto ha 2 figli non ghostati
       non può essere ghost!
       Per ogni punto devo summare lungo il percorso da lui a root, tutti i
       cammini che passano per di lui!
       
       Voglio che se ho un ghost, allora sotto di lui gli altri sono morti.
       Voglio che in un punto 1-la somma degli l_x[i,j,k] con k>0 sia maggiore della somma
       dei l_x[i_,j_,k_] dei punti sotto. 
       Questo dovrebbe sistemare tutti i problemi dei ghost: infatti se un punto sotto
       ha un l_x[i,j,k'] con k'>k, allora in quel punto sta violando la regola perchè sotto di lui ha una cosa non zero!
       
    """

    """
    Lungo il percorso da ogni foglia alla root voglio massimo una assegnazione!
    """
           
    root=1-int(no_r)
            
    for v in T_x.leaves:
        AUX = 0
        for v_aux in T_x.paths[v,:T_x.len_paths[v]-int(no_r)]:
            aux = np.argwhere(T_x.paths == v_aux)
            for w in aux: 
                for j in range(model.m):
                    for aux_y in range(T_y.len_paths[j]-1):
                        AUX += model.L[w[0],j,w[1],aux_y]
        model.costr.add(AUX <= 1)
       
    for v in T_y.leaves:
        AUX = 0
        for v_aux in T_y.paths[v,:T_y.len_paths[v]-int(no_r)]:
            aux = np.argwhere(T_y.paths == v_aux)
            for w in aux: 
                for i in range(model.n):
                    for aux_x in range(T_x.len_paths[i]-1):
                        AUX += model.L[i,w[0],aux_x,w[1]]
        model.costr.add(AUX <= 1)

            
                
def make_W(T_0,T_1,SUB_0,SUB_1,W_0,W_1,V_0,V_1 ,MP,return_map):

    W=np.zeros((T_0.dim,T_1.dim)).astype(np.float)-1
    
    N_0=np.max(T_0.len_paths)
    N_1=np.max(T_1.len_paths)
    
    vertices_0=set(T_0.vertices.copy())
    vertices_1=set(T_1.vertices.copy())
    
    cnt_0=0
    cnt_1=0

    lvl_0=[0]
    lvl_1=[0]

    MAPPINGS = {}            
    
    while cnt_0<N_0 or cnt_1<N_1:

        for v in vertices_0:
            if T_0.len_paths[v]>=N_0-cnt_0:
                lvl_0.append(v)
                vertices_0=vertices_0-set([v])
                
        for v in vertices_1:
            if T_1.len_paths[v]>=N_1-cnt_1:
                lvl_1.append(v)
                vertices_1=vertices_1-set([v])

        if cnt_0==0 and cnt_1==0:        
            lvl_0=lvl_0[1:]
            lvl_1=lvl_1[1:]
        
        cnt_0+=1
        cnt_1+=1

        if MP: 
            
            couple_aux = [(v0,v1) for v0 in sorted(lvl_0) for v1 in sorted(lvl_1)
                                 if W[v0,v1]==-1]

            pool = mp.Pool(processes=7)

            RESULTS=pool.map(calculate_Wxy_pool,([T_0,T_1,W,v[0],v[1],
                                               SUB_0,SUB_1,W_0,W_1,V_0,V_1,return_map] 
                                     for v in couple_aux))
            pool.close()

            cnt_aux=0
            for v0 in sorted(lvl_0):
                for v1 in sorted(lvl_1):
                    if W[v0,v1]==-1:
                        W[v0,v1] = RESULTS[cnt_aux][0]
                        MAPPINGS[(v0,v1)] = RESULTS[cnt_aux][1]
                        cnt_aux+=1
        else:
            for v0 in sorted(lvl_0):
                for v1 in sorted(lvl_1):
                    if W[v0,v1]==-1:
                        results=calculate_Wxy(T_0,T_1,W,v0,v1,
                                               SUB_0,SUB_1,W_0,W_1,V_0,V_1,return_map) 
                        W[v0,v1] = results[0]
                        MAPPINGS[(v0,v1)] = results[1]
    return W, MAPPINGS

def make_sub_trees(T):
    
    SUB=[]  
    V=[]
    W=[]
    p = 1
    
    for v in T.vertices:
        father = T.find_father(v)[0]
        T_aux = T.sub_tree(v)
        mult_aux = {}
        
        for i in T_aux.vertices:
            j = T_aux.find_father(i)[0]
            w = T_aux.name_vertices[i]
            father_aux = T.find_father(w)[0]
            if j == -1:
                mult_aux[(i,j)] = [0]
            else:                
                mult_aux[(i,j)] = T.mult[(w,father_aux)]
                
        T_aux.mult = mult_aux
        T_aux.delta = T.delta
        T_aux.wmax = T.wmax
        T_aux.f = T.f
        T_aux.make_norms_mult()
        
        SUB.append(T_aux)   
        V.append(T.sub_tree(v).name_vertices)
        
        aux = 0
        for fn in mult_aux.values():
            aux += np.linalg.norm(fn, ord = p)*T.delta
            
        W.append(aux)
        
    return SUB,V,W


def top_TED_lineare(T_0,T_1, root = False, fun = False,
                    MP = False, 
                    normalize = False, 
                    verbose = False, return_map = False):

    global T_build
    global T_solve

    T_build = 0
    T_solve = 0
    
    if normalize:
        """
        Qua normalizzo nel senso che faccio diventare 1 l'area totale dell'oggetto
        su cui costruisco il merge tree
        """
        mult_0 = deepcopy(T_0.mult)
        mult_1 = deepcopy(T_1.mult)
        
#         K0 = np.sum(T_0.norms_mult)
#         K1 = np.sum(T_1.norms_mult)

#         KK0 = np.vstack([value for value in mult_0.values() if len(value)>1])
#         KK1 = np.vstack([value for value in mult_1.values() if len(value)>1])

#         K0 = np.max(KK0)
#         K1 = np.max(KK1)

        K0 = np.max([np.max(value) for value in mult_0.values()])
        K1 = np.max([np.max(value) for value in mult_1.values()])

        for key in T_0.mult.keys():
            T_0.mult[key] = T_0.mult[key]/K0
            
        for key in T_1.mult.keys():
            T_1.mult[key] = T_1.mult[key]/K1
        
    SUB_0,V_0,W_0=make_sub_trees(T_0)
    SUB_1,V_1,W_1=make_sub_trees(T_1)

    W, MAP = make_W(T_0,T_1,SUB_0,SUB_1,W_0,W_1,V_0,V_1, MP=MP, return_map = return_map)
        
#    opt=calculate_opt(T_0,T_1,W,W_0,W_1,V_0,V_1)

    if root:
        root_edit = np.abs(T_0.f_uniq[-1]-T_1.f_uniq[-1])
    else:
        root_edit = 0
           
    if MP == False and verbose == True:
        print('\nTempo speso per costruire il modello: ', T_build)
        print('Tempo speso per risolvere il modello: ', T_solve)

#     print(W.shape,root_edit,W)

    if normalize:
        T_0.mult = mult_0
        T_1.mult = mult_1
 
    if return_map:
        M={}
        couples = [(np.max(T_0.vertices),np.max(T_1.vertices))]
#         while len(assigned_0)<len(T_0.vertices)-1 or len(assigned_1)<len(T_1.vertices)-1:
        while len(couples)>0:
            couple = couples[-1]
            couples = couples[:-1]
            mapping =  MAP[couple]
            for edit in mapping.keys():
                M[edit]=mapping[edit]
                if len(edit)>2:
                    couples.append((edit[0],edit[1]))
                   
        return W[-1,-1] + root_edit, M
    else:
        return W[-1,-1] + root_edit



