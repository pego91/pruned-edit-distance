import numpy as np
from scipy.spatial.distance import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import fcluster
import io
import collections
import math

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False



class Graph:
    def __init__(self):
        self.vertices = set()
    
        # makes the default value for all vertices an empty list
        self.edges = collections.defaultdict(list)
        self.weights = {}

    def add_vertex(self, value):
        self.vertices.add(value)

    def add_edge(self, from_vertex, to_vertex, distance):
        if from_vertex == to_vertex: pass  # no cycles allowed
        self.edges[from_vertex].append(to_vertex)
        self.weights[(from_vertex, to_vertex)] = np.asarray(distance)

    def __str__(self):
        string = "Vertices: " + str(self.vertices) + "\n"
        string += "Edges: " + str(self.edges) + "\n"
        string += "Weights: " + str(self.weights)
        return string




def dijkstra(graph, start,vector_weights=True):
    # initializations
    S = set()

    # delta represents the length shortest distance paths from start -> v, for v in delta. 
    # We initialize it so that every vertex has a path of infinity
    delta = dict.fromkeys(list(graph.vertices), math.inf)
    previous = dict.fromkeys(list(graph.vertices), None)

    # then we set the path length of the start vertex to 0
    delta[start] = 0

    # while there exists a vertex v not in S
    while not S == set(graph.vertices):
        
        # let v be the closest vertex that has not been visited...it will begin at 'start'
        v = min((set(delta.keys()) - S), key=delta.get)

        # for each neighbor of v not in S
        for neighbor in set(graph.edges[v]) - S:
            if vector_weights==True:
                new_path = delta[v] + np.abs(graph.weights[v,neighbor][1]-graph.weights[v,neighbor][0])
            else:
                new_path = delta[v] + np.abs(graph.weights[v,neighbor][0])
                
            # is the new path from neighbor through 
            if new_path < delta[neighbor]:
                # since it's optimal, update the shortest path for neighbor
                delta[neighbor] = new_path
        
                # set the previous vertex of neighbor to v
                previous[neighbor] = v


        S.add(v)

    return (delta, previous)

def shortest_path(graph, start, end,tree_mode=True):
    
    if tree_mode==True:
        if start<end:
            graph_aux=inverse_undir_to_dir_graph(graph)
        else:
            graph_aux=undir_to_dir_graph(graph)
    else:
        graph_aux=graph
    
    delta, previous = dijkstra(graph_aux, start)
  
    path = []
    vertex = end

    while vertex is not None:
        path.append(vertex)
        vertex = previous[vertex]

    path.reverse()
    return path


def getEdges(VPos, ITris):
    """
    Given a list of triangles, return an array representing the edges
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.ones(M*6)
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    return L.nonzero()


def matrix_sub_lvl_filtration_mesh(VPos, ITris, fn,order=False):
    """
    
    """
    x = fn(VPos, ITris)

    if order==True:
        x=np.argsort(x)

    N = VPos.shape[0]
    # Add edges between adjacent points in the mesh    
    I, J = getEdges(VPos, ITris)
    V = np.maximum(x[I], x[J])
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    #Create the sparse distance matrix
#    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).toarray()

    return D

def rips_filtration(X,metric='euclidean',K=100):
    
    M=domain_to_matrix(X,metric=metric)
    
    m=np.min(M)    
    N_pt=M.shape[0]    
    eps=m/(N_pt*K)    
    diag=np.arange(0,N_pt)*eps    
    M[range(N_pt),range(N_pt)]=diag    
    f=np.unique(M.flatten())

    return M,f

def multipersistence(VPos, ITris, fn_1,fn_2):
    
    D_1=matrix_sub_lvl_filtration_mesh(VPos, ITris, fn_1,order=True)
    D_2=matrix_sub_lvl_filtration_mesh(VPos, ITris, fn_2,order=True)
    
    D=np.maximum(D_1,D_2)
    f=np.arange(np.min(D),np.max(D)+1)
    
    return D,f
    
##############PRIMO PASSAGGIO

#prendo la matrice delle distance pairwise del dominio#

def domain_to_matrix(D,metric='euclidean'):
    """
    Ottengo la matrice delle adiacenze del dominio
    """
    
    D=np.asarray(D)
    
    if len(D.shape)<2:
        D_aux=[]
        for value in D:
            D_aux.append([value])
    
    D=np.asarray(D_aux)
    
    aux=pdist(D,metric)
    
    return squareform(aux)


def cut_fn_mesh(f,VPos,ITris,I):

    MIN=I[0]
    MAX=I[1]
    
    aux=((MIN-1e15)<f)*(f<(MAX+1e15))
    
    VPos_aux=VPos[aux]
    f_aux=f[aux]
    
    N=np.arange(len(f))
    
    ITris_aux=[]
    
    for n in range(ITris.shape[0]):
        [i,j,k]=ITris[n,:]
        if (i in N) and (j in N) and (k in N):        
            ITris_aux.append([i,j,k])
            
    ITris_aux=np.asarray(ITris_aux)
    
    return f_aux,VPos_aux, ITris_aux


def cut_fn(f,D,I):

    MIN=I[0]
    MAX=I[1]
    
    aux=((MIN-1e15)<f)*(f<(MAX+1e15))
    
    D_aux=D[aux]
    f_aux=f[aux]
        
    return f_aux,D_aux



def sublvl_set_filtration_mesh(f,values,M,I,eps=1e-10, linkage = "single"):
    """
    f: funzione scritta come vettore di N_pts valori; NB non è f_uniq
    values: è tipo f_uniq
    M: matrice di ordine di nascita degli edges
    I: intervallo su cui guardo la persistence
    """

    MIN=I[0]
    MAX=I[1]
    
    N_pt=M.shape[0]
    
    if not len(f)==N_pt:
        print("la funzione ha un numero sbagliato di componenti")
   
    ADJ={}
    
    aux=np.arange(0,N_pt)
       
    for n in range(N_pt):
        point=M[n,:]
        adj=[]
        for vertex in range(N_pt):
            if M[n,vertex]>0 and not vertex==n:
                adj.append(vertex)
        ADJ[n]=adj
 
    FILTRATION={}    
    USED=[]
    FILTRATION[-1]=[]
    N_classes=0
    plt_tree=[]
    CLASSES={}
    
    for i,value in enumerate(values):
        
        sublvl=aux[f<(value+eps)]
        
        M_aux=np.zeros((len(sublvl),len(sublvl)))
        idx_aux=np.arange(0,len(sublvl))


        M_aux=M[np.ix_(sublvl,sublvl)]<value+eps

#
#        for u,pt in enumerate(sublvl):
#            for v,adj in enumerate(sorted(ADJ[pt])):
#                if adj in sublvl:
#                    if M[pt,adj]<(value+eps):
#                        M_aux[u,idx_aux[sublvl==adj]]=1
#                        M_aux[idx_aux[sublvl==adj],u]=1
        aux_graph=csr_matrix(M_aux)
        n_components, labels = connected_components(csgraph=aux_graph, directed=False, return_labels=True)


        if linkage == "average":
            for comp in labels:
                M_aux_aux = M_aux[comp,comp]
                    


        CONN_COMP=[]

        for r in range(n_components):
            aux_comp=sublvl[labels==r]
            CONN_COMP.append(aux_comp)

            UNITED_CLASSES=[]

            for component in FILTRATION[i-1]:
                if len(list(set(component) & set(aux_comp)))>0:
                    UNITED_CLASSES.append(CLASSES[str(component)])              
            if len(UNITED_CLASSES)==1:
                CLASSES[str(aux_comp)]=min(UNITED_CLASSES)
                if value==MAX:
                    plt_tree.append([i,int(max(UNITED_CLASSES)),int(min(UNITED_CLASSES))])
            elif len(UNITED_CLASSES)>1:
                CLASSES[str(aux_comp)]=min(UNITED_CLASSES)
                plt_tree.append([i,int(max(UNITED_CLASSES)),int(min(UNITED_CLASSES))])
            else:
                N_classes+=1
                CLASSES[str(aux_comp)]=N_classes
                plt_tree.append([i,int(N_classes),0])                      
                
        FILTRATION[i]=CONN_COMP

    return plt_tree       

def sublvl_set_filtration(f,values,D,I,epsilon,matrix=False,metric='euclidean',eps=1e-10):
    """
    D: matrice dei punti del dominio (N_pts,dimensione dominio)
    f: funzione scritta come vettore di N_pts valori
    epsilon: definisce un intorno di un punto; i punti nell'intorno gli sono adiacenti 
    I: intervallo su cui guardo la persistence
    """

    MIN=I[0]
    MAX=I[1]
    
    if matrix==False:
        M=domain_to_matrix(D,metric)
    else:
        M=D

    N_pt=M.shape[0]
    
    if not len(f)==N_pt:
        print("la funzione ha un numero sbagliato di componenti")
   
    ADJ={}
    
    aux=np.arange(0,N_pt)
       
    for n in range(N_pt):
        point=M[n,:]
        adj=aux[point<epsilon]
        ADJ[n]=adj
      

    FILTRATION={}    
    FILTRATION[-1]=[]
    N_classes=0
    plt_tree=[]
    CLASSES={}
    
    for i,value in enumerate(values):
        
        sublvl=aux[f<(value+eps)]
        
        M_aux=np.zeros((len(sublvl),len(sublvl)))
        idx_aux=np.arange(0,len(sublvl))

        for u,pt in enumerate(sublvl):
            for v,adj in enumerate(sorted(ADJ[pt])):
                if adj in sublvl:
                    if M[pt,adj]<epsilon:
                        M_aux[u,idx_aux[sublvl==adj]]=1
                        M_aux[idx_aux[sublvl==adj],u]=1

        aux_graph=csr_matrix(M_aux)
        n_components, labels = connected_components(csgraph=aux_graph, directed=False, return_labels=True)

        CONN_COMP=[]

        for r in range(n_components):
            aux_comp=sublvl[labels==r]
            CONN_COMP.append(aux_comp)

            UNITED_CLASSES=[]

            for component in FILTRATION[i-1]:
                if len(list(set(component) & set(aux_comp)))>0:
                    UNITED_CLASSES.append(CLASSES[str(component)])              
            if len(UNITED_CLASSES)==1:
                CLASSES[str(aux_comp)]=min(UNITED_CLASSES)
                if value==MAX:
                    plt_tree.append([i,str(max(UNITED_CLASSES)),str(min(UNITED_CLASSES)),'inf'])
            elif len(UNITED_CLASSES)>1:
                CLASSES[str(aux_comp)]=min(UNITED_CLASSES)
                plt_tree.append([i,str(max(UNITED_CLASSES)),str(min(UNITED_CLASSES)),'inf'])
            else:
                N_classes+=1
                CLASSES[str(aux_comp)]=N_classes
                plt_tree.append([i,str(N_classes),str(0),'inf'])                      
                
        FILTRATION[i]=CONN_COMP

    return plt_tree       


def sublvl_set_filtration_for_Trees(f,values,D,I,varepsilon,matrix=0,metric='euclidean',eps=1e-10):
    """
    D: matrice dei punti del dominio (N_pts,dimensione dominio)
    f: funzione scritta come vettore di N_pts valori
    epsilon: definisce un intorno di un punto; i punti nell'intorno gli sono adiacenti 
    I: intervallo su cui guardo la persistence
    """

    MIN=I[0]
    MAX=I[1]
    epsilon = varepsilon
    
    if np.max(matrix) ==0:
        M=domain_to_matrix(D,metric)
    else:
        M=matrix

    N_pt=M.shape[0]
    
    if not len(f)==N_pt:
        print("la funzione ha un numero sbagliato di componenti")
   


    FILTRATION={}    
    FILTRATION[-1]=[]
    N_classes=0
    plt_tree=[]
    CLASSES={}
    
    F=[]
    cnt=0
    
    """
    Check
    """
    n_components = 2
    i=0
    
    while n_components>1:
        M_aux=np.zeros((N_pt,N_pt))
        
        ADJ={}
    
        aux=np.arange(0,N_pt)

        for n in range(N_pt):
            point=M[n,:]
            adj=aux[point<epsilon]
            ADJ[n]=adj

#         print("le adiacenze sono al max: ", max([len(adj) for adj in ADJ.values()]))

        for u,pt in enumerate(aux):
            for v,adj in enumerate(sorted(ADJ[pt])):
                if M[pt,adj]<epsilon:
                    M_aux[u,adj]=1
                    M_aux[adj,u]=1

        aux_graph=csr_matrix(M_aux)
        n_components = connected_components(csgraph=aux_graph, directed=False, return_labels=False)

        epsilon = epsilon + epsilon*(i/10) 
        i=i+1

#         print("merdaaaaaaaaa", epsilon, np.max(M),n_components)
    
    
    
    for i,value in enumerate(values):        
        sublvl=aux[f<(value+eps)]
        
        M_aux=np.zeros((len(sublvl),len(sublvl)))
        idx_aux=np.arange(0,len(sublvl))
        
#         print("Sottolivello è %s: "%str(sublvl))


        for u,pt in enumerate(sublvl):
            for v,adj in enumerate(sorted(ADJ[pt])):
                if adj in sublvl:
                    if M[pt,adj]<epsilon:
                        M_aux[u,idx_aux[sublvl==adj]]=1
                        M_aux[idx_aux[sublvl==adj],u]=1

        aux_graph=csr_matrix(M_aux)
        n_components, labels = connected_components(csgraph=aux_graph, directed=False, return_labels=True)

        CONN_COMP=[]

#         print("Siamo al punto %i, su %i.    " %(i,len(values)), end='\r' )

        for r in range(n_components):
            """
            per ognuna delle componenti connesse, estraggo i punti che le appartengono nel sottolivello;
            e la aggiungo alle CONN_COMP
            """
            aux_comp=sublvl[labels==r]
            CONN_COMP.append(aux_comp)

            UNITED_CLASSES=[]

            for component in FILTRATION[i-1]:
                """
                Per ognuna delle componenti connesse allo step precedente guardo se c'è intersezione tra quella componente vecchia,
                e la componente attuale che sto considerando nel sottolivello. Se trovo intersezione allora in UNITED_CLASSES aggiungo
                il numero associato alla componente (dello step precedente). Così raccolgo tutte le classi che si sono unite nella nuova
                componente.
                """
                if len(list(set(component) & set(aux_comp)))>0:
                    UNITED_CLASSES.append(CLASSES[str(component)])          
                    
            """
            Se questa nuova componente è semplicemente una componente dello step precedente che si è espansa,
            allora posso soprassedere.
            Se invece sono più classi che si uniscono allora assegno a questa componente connessa, il numero della classe pari al 
            minimo delle classi che si uniscono e aggiungo il punto al plt_tree.
            Se invece la classe è nuova aumento il numero delle classi in giorno e aggiorno tutto di conseguenza.
            """
            if len(UNITED_CLASSES)==1:
                pass
                CLASSES[str(aux_comp)]=min(UNITED_CLASSES)
#                 if value==MAX:
#                     plt_tree.append([cnt,int(max(UNITED_CLASSES))-1,int(min(UNITED_CLASSES))-1])
#                     F.append(value)
#                     cnt+=1
            elif len(UNITED_CLASSES)>1:
                CLASSES[str(aux_comp)]=min(UNITED_CLASSES)
                plt_tree.append([cnt,int(max(UNITED_CLASSES))-1,int(min(UNITED_CLASSES))-1])
                F.append(value)
                cnt+=1
            else:
                N_classes+=1
                CLASSES[str(aux_comp)]=N_classes
                plt_tree.append([cnt,int(N_classes)-1,-1])                      
                F.append(value)
                cnt+=1
                
        FILTRATION[i]=CONN_COMP

    plt_tree.append([cnt,0,0])
    F.append(MAX)
    cnt+=1
    return np.asarray(plt_tree),np.asarray(F)       

def plt_tree_to_plt_Tree(plt_tree,f_old):
    
    plt_Tree=np.zeros((len(plt_tree),3),dtype=np.int)
    f=np.zeros((len(plt_tree),))
    
    for i,pt in enumerate(plt_tree):
        plt_Tree[i,:]=[i,np.int(pt[1])-1,np.int(pt[2])-1]
        f[i]=f_old[pt[0]]

#    if plt_Tree[-1][1]==plt_Tree[-1][2]:
#        aux=np.array([[plt_Tree[-1][0],np.max(plt_Tree[:,1])+1,-1],
#                      [plt_Tree[-1][0]+1,np.max(plt_Tree[:,1])+1,plt_Tree[-1][2]]])
#        plt_Tree=np.r_[plt_Tree[:-1,:],aux]
#        
#        f=np.r_[f[:-1],[f[-1]-1e-10,f[-1]]]


#    return np.int(plt_Tree),f
    return plt_Tree,f
    

def plt_Tree_to_plt_tree(plt_Tree,f_old):
    
    plt_tree=[]
    f=np.zeros((len(plt_Tree),))
    
    for i,pt in enumerate(plt_Tree):
        plt_tree.append([np.int(i),str(pt[1]+1),str(pt[2]+1),'inf'])
        f[i]=f_old[pt[0]]

    return plt_tree,f    


def to_newick(plt_tree,f_uniq,PERSISTENCE_MODE=True,d_1=False,MAP=[],ZERO_MODE=False):
    """
    plt_tree: punti chiave dell'albero
    f_uniq: vettore coi valori unici della funzione in ordine, MAX in coda
    """
    
    last_point={}
    blocks={}

    for n,point in enumerate(plt_tree):#prendo i punti uno alla volta con ascissa crescente 
        
        if point[2]=='0': #punto di nascita
            
            try:
                last_point[point[1]].append(n) #n-esimo punto: punto + alto della componente che nasce
            except:
                last_point[point[1]]=[]
                last_point[point[1]].append(n) #n-esimo punto: punto + alto della componente che nasce
            
            
            if len(MAP)==0:            
                blocks[point[1]]=str(n) 
            else:
                blocks[point[1]]=str(MAP[n])
        else:#punto di morte: si uniscono due componenti
            if not point[1]=='1'or not point[2]=='1': 
                                
                pt_1=last_point[point[1]][-1] #punto prec della compo morente  
                pt_2=last_point[point[2]][-1] #punto prec della compo sopravvivente
               
                last_point[point[2]].append(n) #n-esimo punto: il punto + alto della componente che vince                       

                if PERSISTENCE_MODE==False:
                    weight_1=np.asarray([f_uniq[plt_tree[pt_1][0]],f_uniq[point[0]]])
                    weight_2=np.asarray([f_uniq[plt_tree[pt_2][0]],f_uniq[point[0]]])
                elif d_1==True:
                    weight_1=np.asarray([np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt_1][0]])])
                    weight_2=np.asarray([np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt_2][0]])])
                else:
                    weight_1=np.asarray([0,np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt_1][0]])])
                    weight_2=np.asarray([0,np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt_2][0]])])

                if d_1==True and not ZERO_MODE:
                    blocks[point[1]]= blocks[point[1]]+':'+str(weight_1)[1:-1] #aggiorno la parente che muore con la lunghezza
                    blocks[point[2]]= blocks[point[2]]+':'+str(weight_2)[1:-1] #aggiorno la parentesi che vive con la lunghezza
                elif ZERO_MODE:
                    blocks[point[1]]= blocks[point[1]]+':'+str(0) #aggiorno la parente che muore con la lunghezza
                    blocks[point[2]]= blocks[point[2]]+':'+str(0) #aggiorno la parentesi che vive con la lunghezza
                else:
                    blocks[point[1]]= blocks[point[1]]+':'+np.array2string(weight_1,separator=',') #aggiorno la parente che muore con la lunghezza
                    blocks[point[2]]= blocks[point[2]]+':'+np.array2string(weight_1,separator=',') #aggiorno la parentesi che vive con la lunghezza
                    
                    
                blocks[point[2]]='('+blocks[point[2]]+','+blocks[point[1]]+')' 
            else:                

                pt=last_point[point[1]][-1] #punto prec della compo morente  

                if PERSISTENCE_MODE==False:
                    weight=np.asarray([f_uniq[plt_tree[pt][0]],f_uniq[point[0]]])
                elif d_1==True:
                    weight=np.asarray([np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt][0]])])
                else:
                    weight=np.asarray([0,np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt][0]])])

                if d_1==True and not ZERO_MODE:
                    blocks[point[1]]='('+blocks[point[1]]+':'+str(np.asarray(weight))[1:-1]+',fake:0.0)' 
                elif ZERO_MODE:
                    blocks[point[1]]='('+blocks[point[1]]+':'+str(0)+',fake:0)' 

                else:
                    blocks[point[1]]='('+blocks[point[1]]+':'+np.array2string(weight,separator=',')+',fake:[0,0])' 
            
    return blocks[point[2]]


def make_order(lista,data_paz):
    """
    prende una lista e la mette in cordine con le etichette in data_paz
    """
    U=[]
    N=[]
    L=[]
    
    for patient,obj in enumerate(lista):
        if data_paz["type"][patient]=='L':
            L.append(obj)
        elif data_paz["type"][patient]=='U':
            U.append(obj)
        else:
            N.append(obj)
            
    return L+N+U

def unify(f,eps,prec, mode):
    """
    perturbo la funzione nella speranza che f diventi iniettiva
    """
    for value in f:
        if mode == 0:
            aux=np.abs(f-value)<prec  
        else:
            aux=f==value        
        if sum(aux)>1:
            f=f-aux*eps*np.random.uniform(-1,1,(len(f),))
    return f


def preprocess_f(f,I,K=1000, prec = 0.000000001):
    """
    continuo a perturbare f finchè non diventa iniettiva, aggiungo il MAX alla fine
    """
    
    f=np.asarray(f)
    
    delta=0
    MAX=I[1]
    
    delta = np.max(np.diff(f))
         
#     eps=max([delta/K,prec*10])
    eps = prec*20
    
    f_aux=np.copy(f)
    cnt = 0
    
    if np.min(np.abs(np.diff(np.unique(f_aux))))<prec:
        mode = 0
    elif not len(np.unique(f_aux))==len(f_aux):
        mode = 1
    else:
        mode = -1
    
    if len(f)>1 and mode>-1 :
        print("Non iniettiva: ",len(np.unique(f_aux)),len(f_aux) )
        f_=np.array([0,prec])    
        while not mode==-1:
#         while not len(np.unique(f_))==len(f_aux):
            if cnt >1000:
                f_aux = f_
    
            f_=unify(f_aux,eps,prec,mode)
            
            if np.min(np.abs(np.diff(np.unique(f_))))<prec:
                mode = 0
            elif not len(np.unique(f_))==len(f_aux):
                mode = 1
            else:
                mode = -1
            cnt+=1
        
        f_aux=f_
        print("Ora è iniettiva: ",len(np.unique(f_aux)),len(f_aux) )
        
    f_uniq=np.unique(f_aux) #lo ordina

    if MAX >max(f_uniq):
        f_uniq=np.r_[f_uniq,[MAX]]

    return f_uniq,f_aux


def plot_newick(newick,axes=None):
    
    handle = io.StringIO(newick)
    tree = Phylo.read(handle, "newick")
    Phylo.draw(tree,axes=axes)


#####################################################################

def prune_graph(tree,thresh,vector_weights=True, verbose = False):
    
    aux_tree=Graph()
    while not len(aux_tree.vertices)==len(tree.vertices):

        aux_tree=tree
        tree=prune_leaves_graph(tree,thresh,vector_weights)
       
    if len(tree.vertices)==1:
        print('albero banale!')
    if verbose == True:
        print('Numero vertici: ',len(tree.vertices), '; Numero foglie:',len(find_leaves(tree)))

    return tree


def prune_leaves_graph(graph,thresh,vector_weights=True):
    """
    poto un albero. tolgo le foglie che hanno una persistence troppo bassa.
    - mi metto su un ramo. prendo la foglia più giovane e se ha una persistence bassa la poto!
    """
    
    aux_graph=Graph()

    """
    faccio un giro per trovare le foglie da togliere
    nello stesso giro capisco quelli che posso togliere
    """
    
    pruned=[]
    not_needed=[]
    root=max(graph.vertices)
    aux=0
    mapping={} #va da aux a graph
    inv_mapping={} # va da graph ad aux
    
    
    for vertex in sorted(graph.vertices):
        """
        sto prendendo i punti in ordine crescente di nascita
        """
        if vertex in pruned:
            pass
        else:
            FLAG=1

            if len(children_undir(graph,vertex))==0:
                """
                ho trovato una foglia, prendo suo padre e i suoi figli 
                i.e. i fratelli della foglia.
                """
                father=max(graph.edges[vertex])
                
                children=np.asarray(children_undir(graph,father))

                try:
                    """
                    prendo un fratello del mio vertice che so essere una foglia.
                    se questo brother è una foglia allora prendo il più giovine 
                    dei due (l'albero è binario!) ed è lui il prescelto per 
                    l'eventuale sacrificio. altrimenti è la foglia "vertex".
                    """
                    brother=children[children!=vertex ][0]
                    if len(children_undir(graph,brother))==0:
                        isacco=max(children)
                    else:
                        isacco=vertex
                                   
                    weight=np.asarray(graph.weights[(father,isacco)])
    
                    """
                    ora controllo se il ramo sacrificabile è effettivamente
                    più corto della mia threshold.
                    """
    
                    if np.abs(weight[1]-weight[0])<thresh and isacco not in pruned:
                        pruned.append(isacco)
    
                        if (not father == root) and (not father in not_needed):
                            not_needed.append(father)
                        if vertex == isacco:
                            FLAG=0
                except:                    
                    FLAG=1 #se fallisce brother!

            elif (len(children_undir(graph,vertex))==1) and (not vertex in not_needed) and (not vertex==root):
                not_needed.append(vertex)
                
            if vertex in not_needed:
                FLAG=0
                
            if FLAG==1:
                aux_graph.add_vertex(aux)
                mapping[aux]=vertex
                inv_mapping[vertex]=aux
                aux=aux+1                

    if root not in inv_mapping.keys():
        aux_graph.add_vertex(aux)
        mapping[aux]=root
        inv_mapping[root]=aux
        aux=aux+1  

    """
    ora sono pronto per aggiungere gli edge al grafo aux, aggiustando i pesi!
    """    
        
    dir_graph=undir_to_dir_graph(graph)
    inv_dir_graph=inverse_undir_to_dir_graph(graph)
     
    aux_root=max(aux_graph.vertices)
    
    if len(aux_graph.vertices)==1:
        print("l'albero è banale!")        
        return aux_graph
    else:
        for vertex in sorted(aux_graph.vertices):
            if vertex==aux_root:
                pass
            else:    
                father=max(inv_dir_graph.edges[mapping[vertex]]) #questo sta in graph
                
                if father not in not_needed:

                    children=children_undir(graph,mapping[vertex])
                   
                    aux_graph.add_edge(vertex,
                                        inv_mapping[father],graph.weights[(mapping[vertex],father)])
                    aux_graph.add_edge(inv_mapping[father],
                                        vertex,graph.weights[(mapping[vertex],father)])

                else:
                    
                    generations=[mapping[vertex],father]
                    
                    while max(inv_dir_graph.edges[father]) not in inv_mapping.keys():
                        father=max(inv_dir_graph.edges[father])
                        generations.append(father)
                    
                    granpa=inv_mapping[max(inv_dir_graph.edges[father])]
                    generations.append(max(inv_dir_graph.edges[father]))
                    
                    tot_weights=np.asarray([0.0,0.0])
                    
                    for i in range(len(generations)-1):
                        tot_weights+=np.asarray(graph.weights[(generations[i],generations[i+1])])
     
                    aux_graph.add_edge(granpa,vertex,tot_weights)
                    aux_graph.add_edge(vertex,granpa,tot_weights) 

        sorted_graph=sort_undir_graph(aux_graph,-1,vector_weights=vector_weights)    
        return sorted_graph
    
def dir_to_undir_graph(dir_graph):
    """
    prendo un grafo diretto e aggiungo tutti segmenti anche nell'altro verso
    """
    
    undir_graph=Graph()

    for v in dir_graph.vertices:
            undir_graph.add_vertex(v)
            for end_v in dir_graph.edges[v]:
                undir_graph.add_edge(v,end_v,dir_graph.weights[(v,end_v)])
                undir_graph.add_edge(end_v,v,dir_graph.weights[(v,end_v)])

    return undir_graph


def undir_to_dir_graph(undir_graph):
    """
    prendo un grafo non diretto e lo rendo diretto dal vertice verso le foglie
    """
    
    dir_graph=Graph()
    
    for v in undir_graph.vertices:
        dir_graph.add_vertex(v)
        for end_v in undir_graph.edges[v]:
                if end_v<v:
                    dir_graph.add_edge(v,end_v,undir_graph.weights[(v,end_v)])
    return dir_graph

def inverse_undir_to_dir_graph(undir_graph):
    """
    prendo un grafo non diretto e lo rendo diretto dalle foglie verso il vertice
    """

    dir_graph=Graph()
    
    for v in undir_graph.vertices:
        dir_graph.add_vertex(v)
        for end_v in undir_graph.edges[v]:
            if end_v>v:
                dir_graph.add_edge(v,end_v,undir_graph.weights[(v,end_v)])
    return dir_graph

def sort_undir_graph(graph,index_root,vector_weights=True,mapping_order=False):
    """
    prendo un grafo e rinomino i vertici in ordine dal più lontano al più vicino alla radice;
    NB la radice deve però essere data come posizione nell'ordine (crescente) dei vertici!! 
    """

    OLD_VERT=sorted(list(graph.vertices))

    aux_graph=Graph()
    aux_graph.vertices=set(np.arange(0,len(OLD_VERT.copy())))
    
    matrix=graph_to_metric(graph,vector_weights=vector_weights)
    M=matrix[index_root,:]
    
    mapping=np.argsort(-M)
    
    inv_map={}
    
    for n,key in enumerate(mapping):
        inv_map[OLD_VERT[key]]=n
        
    for edge in graph.weights.keys():
        aux_graph.add_edge(inv_map[edge[0]],inv_map[edge[1]],graph.weights[edge])

    if mapping_order:
        return aux_graph, inv_map
    else:
        return aux_graph


def children_ordered_graph(T,vertex):
    """
    Dato un albero binario ordinato left right dato un vertice prendo i suoi eventuali figli
    Se non ne ha restituisco lista vuota
    """
    
    M=max(T.vertices)
    aux=vertex*2
    if aux<M:
        children=[vertex*2,vertex*2+1]
    else:
        children=[]
    return children

def children_undir(undir_graph,vertex):
    """
    Dato un albero binario sotto forma di grafo non orientato, prendo gli eventuali figli di un vertice
    Se non ne ha resistuisco lista vuota
    """

    graph=undir_to_dir_graph(undir_graph)

    return sorted(graph.edges[vertex])


def find_leaves(graph):
    
    leaves=set()
    
    dir_graph=undir_to_dir_graph(graph)
    
    for vertex in dir_graph.vertices:
        if not vertex in dir_graph.edges.keys() and not vertex==max(dir_graph.vertices):
            leaves.add(vertex)
            
    return list(leaves)


def find_brother(graph,vertex):
    
    father_=find_father(graph,vertex)
    children=set(children_undir(graph,father_))        
    v=set()
    v.add(vertex)
    
    if len(children-v)==0:
        return None
    else:
        return list(children-v)[0]


def find_father(graph,vertex):
    inv_dir_graph=inverse_undir_to_dir_graph(graph)
    if len(inv_dir_graph.edges[vertex])>0:
        return inv_dir_graph.edges[vertex][0]
    else:
        return None
    
def parents(graph,vertex):
    
    if vertex==None:
        return set()
    elif vertex=='G':
        return graph.vertices
    else:
        inv_dir_graph=inverse_undir_to_dir_graph(graph)
        
        AUX,_=dijkstra(inv_dir_graph,vertex)        
        parents_set=[]
        
        for v in inv_dir_graph.vertices:
            if AUX[v]<np.inf:
                parents_set.append(v)
        
        return set(parents_set)
    
    
def to_graph(plt_tree,f_uniq,PERSISTENCE_MODE=True):
    """
    plt_tree: sono i punti chiave dell'albero
    f: è la funzione coi valori unici e in ordine, mi serve che abbia il MAX in coda
    se la PERSISTENCE MODE è attivata, la lunghezza dei rami non è (birth,death) ma è (0,persistence)
    """
#    come input prendo i punti
    
    tree=Graph()
    
    last_point=collections.defaultdict(list)

    for n,point in enumerate(plt_tree):#prendo i punti uno alla volta con ascissa crescente  

        tree.add_vertex(n)

        if int(point[2])==0: #punto di nascita            
#        if point[2]=='0': #punto di nascita            
            last_point[point[1]].append(n) #n-esimo punto: punto + alto della componente che nasce
        else:#punto di morte: si uniscono due componenti
            if not (point[1]==point[2]):                 
                
                pt_1=last_point[point[1]][-1] #punto prec della compo morente  
                pt_2=last_point[point[2]][-1] #punto prec della compo sopravvivente
    
                last_point[point[2]].append(n) #n-esimo punto: il punto + alto della componente che vince                                  

                if PERSISTENCE_MODE==False:
                    weight_1=np.asarray([f_uniq[plt_tree[pt_1][0]],f_uniq[point[0]]])
                    weight_2=np.asarray([f_uniq[plt_tree[pt_2][0]],f_uniq[point[0]]])
                else:
                    weight_1=np.asarray([0,np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt_1][0]])])
                    weight_2=np.asarray([0,np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt_2][0]])])

                tree.add_edge(n,pt_1,weight_1)
                tree.add_edge(pt_1,n,weight_1)
                tree.add_edge(n,pt_2,weight_2)
                tree.add_edge(pt_2,n,weight_2)
    
            else:
                pt=last_point[point[1]][-1] #punto prec della compo morente  

                if PERSISTENCE_MODE==False:
                    weight=np.asarray([f_uniq[plt_tree[pt][0]],f_uniq[point[0]]])
                else:
                    weight=np.asarray([0,np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt][0]])])
 
                tree.add_edge(n,n-1,weight)
                tree.add_edge(n-1,n,weight)
    
    return tree

def graph_to_metric(graph,vector_weights=True):
    """
    trasformo il grafo in uno sazio metrico usando dijkstra
    """
          
    D=[]
    
    for i,vertex_1 in enumerate(sorted(graph.vertices)):
        aux_1=[]
        
        aux_2,_=dijkstra(graph, vertex_1,vector_weights=vector_weights)
    
        for j,vertex_2 in enumerate(sorted(graph.vertices)):
            aux_1.append(aux_2[vertex_2])
        D.append(aux_1)
        
    D=np.asarray(D)
 
    return D


def graph_to_plt_tree(graph,I,PERSISTENCE_MODE=True,vector_weights=True):
    """
    prendo un grafo e lo trasformo in plt_tree, i.e. i punti salienti
    """
    plt_tree=[]
    
    matrix=graph_to_metric(graph,vector_weights=vector_weights)
    inv_dir_graph=inverse_undir_to_dir_graph(graph)
        
    children=collections.defaultdict(list) #ci metto le "classi vincenti" che si incontrano
    MIN=I[0]
    MAX=I[1]
    f_uniq=[]
    aux=-1
    
    classes={}
    n_class=0
    if len(graph.vertices)>2:
        for vertex in sorted(graph.vertices): 

            try:
                children[max(graph.edges[vertex])].append(min(children[vertex]))
            except:
                children[max(graph.edges[vertex])].append(vertex)

            if PERSISTENCE_MODE==False:
                try:
                    father=inv_dir_graph.edges[vertex][0]
                    fn=inv_dir_graph.weights[(vertex,father)][0]
                except:
                    fn=MAX
            else:
                fn=MAX-matrix[-1,vertex]
            
            if fn not in f_uniq:
                f_uniq.append(fn)
                aux=aux+1

            if not min(graph.edges[vertex])<vertex:   
                n_class+=1
                classes[vertex]=str(n_class)
                plt_tree.append([aux,classes[vertex],str(0),np.inf]) 
            else:
                plt_tree.append([aux,classes[max(children[vertex])],classes[min(children[vertex])],np.inf]) 

    elif len(graph.vertices)==1:
        print("L'albero è vuoto")
        return plt_tree, []
    elif len(graph.vertices)==2:
        return [[0,'1','0',np.inf],[1,'1','1',np.inf]],[MAX-matrix[0,1],MAX]
    

    return plt_tree, sorted(f_uniq)



def Z_to_plt_Tree(Z,M):
    
    n_pts = np.int(Z[-1,-1])
    eps = 0.0001
    
    births = np.cumsum(np.ones((n_pts,))*eps)
    
    f = Z[:,2]
    f = np.r_[births,f]
    
#     f_uniq = np.unique(f)

#     print(f.shape, f_uniq.shape)
    _, f = preprocess_f(f,[min(f),max(f)],K=10000, prec = eps*0.3)

    f_uniq = f
        
    plt_tree = np.zeros((Z.shape[0]+n_pts,3), dtype = np.int)

    clusters = {}
    M = M + np.identity(M.shape[0])*100000
    new_names = {}

    for i in range(plt_tree.shape[0]):
        
        if i < n_pts:
            plt_tree[i,:] = np.array([i,i,-1])
            clusters[i]=i    
            new_names[i]=i
        else:
            line = Z[i-n_pts,:]
            
            pt0 = line[0]
            pt1 = line[1]

            surv = np.min([clusters[pt0],clusters[pt1]])
            dies = np.max([clusters[pt0],clusters[pt1]])
            
            clusters[i] = clusters[surv]            

            plt_tree[i,:] = np.array([i,clusters[dies],clusters[surv]])        
            
#    for i in range(plt_tree.shape[0]):
#        
#        if i < n_pts:
#            plt_tree[i,:] = np.array([i,i,-1])
#            clusters[i]=i    
#            
#        else:
#            line = Z[i-n_pts,:]
#            
#            clusters[i] = clusters[line[0]]               
#            clus = fcluster(Z,f_uniq[i]-eps,criterion='distance')
#            
#            
#            if line[0]<n_pts and line[1]<n_pts:
#                pt_0 = line[0]
#                pt_1 = line[1]
#                
#            elif line[1]<n_pts:
#                pt_0 = np.int(Z[np.int(line[0]%n_pts)][0])
#                c_0 = clusters[line[0]]
#                clus_0 = np.argwhere(clus == clus[c_0])
#                pt_0 = np.min(np.array([pt_[0] for pt_ in clus_0]))
#                
#                pt_1 = line[1]
#                
#            elif line[0]<n_pts:
#                pt_0 = line[0]
#
#                pt_1 = np.int(Z[np.int(line[1]%n_pts)][0])
#                c_1 = clusters[line[1]]
#                clus_1 = np.argwhere(clus == clus[c_1])
#                pt_1 = np.min(np.array([pt_[0] for pt_ in clus_1]))
#                
#
#            else:
#                pt_0 = np.int(Z[np.int(line[0]%n_pts)][0])
#                c_0 = clusters[line[0]]
#                clus_0 = np.argwhere(clus == clus[c_0])
#                pt_0 = np.min(np.array([pt_[0] for pt_ in clus_0]))
#                
#                pt_1 = np.int(Z[np.int(line[1]%n_pts)][0])
#                c_1 = clusters[line[1]]
#                clus_1 = np.argwhere(clus == clus[c_1])
#                pt_1 = np.min(np.array([pt_[0] for pt_ in clus_1]))
#
#            pt = np.array([pt_0,pt_1])
#            aux_0 = np.min(pt)
#            aux_1 = np.max(pt)
#            plt_tree[i,:] = np.array([i,aux_1,aux_0])        

        
    return plt_tree,f_uniq
        

def sublvl_set_filtration_for_MergeDendros(f,values,D,I,varepsilon,matrix=0,metric='euclidean',eps=1e-15):
    """
    D: matrice dei punti del dominio (N_pts,dimensione dominio)
    f: funzione scritta come vettore di N_pts valori
    epsilon: definisce un intorno di un punto; i punti nell'intorno gli sono adiacenti 
    I: intervallo su cui guardo la persistence
    values: punti critici
    varepsilon: parametro che serve per costruire il grafo costruendo bolle intorno ai punti del dominio
    """
    
    epsilon = varepsilon
    delta = 99999999
    
    if np.max(matrix) ==0:
        M=domain_to_matrix(D,metric)
    else:
        M=matrix

    N_pt=M.shape[0]
    
    if not len(f)==N_pt:
        print("la funzione ha un numero sbagliato di componenti")
   


    FILTRATION={}    
    FILTRATION[-1]=[]
    N_classes=0
    plt_tree=[]
    CLASSES={}
    
    F=[]
    cnt=0
    
    """
    Check che all'ultimo sottolivello il grafo risulti connesso
    """
    n_components = 2
    i=0
    
    while n_components>1:
        M_aux=np.zeros((N_pt,N_pt))
        
        ADJ={}
    
        aux=np.arange(0,N_pt)

        for n in range(N_pt):
            point=M[n,:]
            adj=aux[point<epsilon]
            ADJ[n]=adj
            delta = np.min([np.min(np.diff(f[adj])),delta])    

#         print("le adiacenze sono: ", [adj for adj in ADJ.values()])
#         print("le adiacenze sono al max: ", max([len(adj) for adj in ADJ.values()]))

        for u,pt in enumerate(aux):
            for v,adj in enumerate(sorted(ADJ[pt])):
                if M[pt,adj]<epsilon:
                    M_aux[u,adj]=1
                    M_aux[adj,u]=1

        aux_graph=csr_matrix(M_aux)
        n_components = connected_components(csgraph=aux_graph, directed=False, return_labels=False)

        epsilon = epsilon + epsilon*(i/10) 
        i=i+1

    MIN=I[0]
    delta_ = np.min(np.diff(values))
    delta = np.min([0.000001,delta_*0.1])
    delta = 0.000001
    MAX=I[1]+delta
    
#     print('deltaaaaa ',delta)     
               
    for i,value in enumerate(values):        
        sublvl=aux[f<=value]
#         sublvl=aux[(f<value+eps)]
        
        M_aux=np.zeros((len(sublvl),len(sublvl)))
        idx_aux=np.arange(0,len(sublvl))
        
#         print("\nIl sottolivello ad altezza %f è %s: "%(value,str(sublvl)))


        for u,pt in enumerate(sublvl):
            for v,adj in enumerate(sorted(ADJ[pt])):
                if adj in sublvl:
                    if M[pt,adj]<epsilon:
                        M_aux[u,idx_aux[sublvl==adj]]=1
                        M_aux[idx_aux[sublvl==adj],u]=1

        aux_graph=csr_matrix(M_aux)
        n_components, labels = connected_components(csgraph=aux_graph, directed=False, return_labels=True)

        CONN_COMP=[]

#         print("Siamo al punto %i, su %i.    " %(i,len(values)), end='\r' )
        pts_added = 1
  
        for r in range(n_components):
            """
            per ognuna delle componenti connesse, estraggo i punti che le appartengono nel sottolivello;
            e la aggiungo alle CONN_COMP
            """
            aux_comp=sublvl[labels==r]
            CONN_COMP.append(aux_comp)

            UNITED_CLASSES=[]
            flag = 0      
            d = delta*pts_added
            
            for component in FILTRATION[i-1]:
                """
                Per ognuna delle componenti connesse allo step precedente guardo se c'è intersezione tra quella componente vecchia,
                e la componente attuale che sto considerando nel sottolivello. Se trovo intersezione allora in UNITED_CLASSES aggiungo
                il numero associato alla componente (dello step precedente). Così raccolgo tutte le classi che si sono unite nella nuova
                componente. La flag mi dice che la componente non è rimasta uguale al passo prima della filtrazione, ma effettivamente si 
                sono aggiunti dei punti.
                """
                if len(list(set(component) & set(aux_comp)))>0:                    
                    UNITED_CLASSES.append(CLASSES[str(component)])
                    if len(list(set(aux_comp) - set(component)))>0:
                        flag = 1
#                     print('\nLe componenti sono: ',component,aux_comp)
            
#             print('\nLe classi in ', CLASSES, '  che si uniscono ad altezza ',value,', sono: ', UNITED_CLASSES)
            """
            Se questa nuova componente è semplicemente una componente dello step precedente che si è espansa,
            allora posso soprassedere.
            Se invece sono più classi che si uniscono allora assegno a questa componente connessa, il numero della classe pari al 
            minimo delle classi che si uniscono e aggiungo il punto al plt_tree.
            Se invece la classe è nuova aumento il numero delle classi in giorno e aggiorno tutto di conseguenza.
            """
            if len(UNITED_CLASSES)==1 and flag ==1:
                
                CLASSES[str(aux[f==value])] = int(N_classes)
                CLASSES[str(aux_comp)]=min(UNITED_CLASSES)
                plt_tree.append([cnt,int(N_classes),-1])                      
                F.append(value)
                cnt+=1
                
#                 print('cheppalle ',d,cnt,F[-3:])
                
                plt_tree.append([cnt,int(N_classes),int(min(UNITED_CLASSES))])                      
                N_classes+=1
                F.append(value + d)
                cnt+=1                
                pts_added += 1

#                 print('cheppalle 2 ',d,cnt,F[-3:])

                
            elif len(UNITED_CLASSES)>1:
                """
                qua perdo dei punti: il punto che mi unisce le componenti me lo perdo. devo farlo nascere e poi morire.
                """
#                 N_classes+=1
                CLASSES[str(aux_comp)]=min(UNITED_CLASSES)
                plt_tree.append([cnt,int(N_classes),-1])                      
        
                F.append(value-d)
                cnt+=1

#                 print('cheppalle ',d,cnt,F[-3:])

                plt_tree.append([cnt,int(N_classes),int(min(UNITED_CLASSES))])                      
                F.append(value)
                
                N_classes+=1
                cnt+=1

#                 print('cheppalle 2',d,cnt,F[-3:])

                if len(UNITED_CLASSES)>2:
                    auxxx = [clasS for clasS in UNITED_CLASSES if clasS < max(UNITED_CLASSES)]
                    for s,clasS in enumerate(auxxx):
                        plt_tree.append([cnt,int(clasS),int(min(UNITED_CLASSES))])
                        F.append(value+d*(s/len(auxxx)))
                        cnt+=1
                else:
                    plt_tree.append([cnt,int(max(UNITED_CLASSES)),int(min(UNITED_CLASSES))])
                    F.append(value+d)
                    cnt+=1

#                 print('cheppalle 3',d,cnt,F[-3:])

                pts_added += 1
                    
            
            elif len(UNITED_CLASSES)==0:
                CLASSES[str(aux_comp)]=N_classes
                plt_tree.append([cnt,int(N_classes),-1])                      
                F.append(value)
                N_classes+=1
                cnt+=1
                pts_added += 1

#             print('\npoints added ', pts_added)

        FILTRATION[i]=CONN_COMP

    plt_tree.append([cnt,0,0])
    F.append(MAX+delta+delta)
    cnt+=1

#     print('Finally \n', np.array(plt_tree))

    return np.array(plt_tree),np.array(F) ,delta      

  


 













