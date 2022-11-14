from scipy.cluster.hierarchy import single, average, weighted, complete, centroid, median, ward, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import beta as measure
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from Trees_OPT import Tree
import Utils_OPT

import numpy as np

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
                  
    return plt_tree,f_uniq

def dendrolink(N, linkage = 'single', root = False, root_height = 1):
    
    if N.shape[0]!=N.shape[1]:
        M = squareform(pdist(N))
    else:
        M = np.copy(N)
        
    y = squareform((M+M.T)/2)
        
    if linkage == 'single':
        Z = single(y)
    elif linkage == 'average':
        Z = average(y)
    elif linkage == 'complete':
        Z = complete(y)
    elif linkage == 'weighted':
        Z = weighted(y)
    elif linkage == 'centroid':
        Z = centroid(y)
    elif linkage == 'median':
        Z = median(y)
    elif linkage == 'ward':
        Z = ward(y)
        
    plt_Tree, f_uniq = Utils_OPT.Z_to_plt_Tree(Z,M)
        
    if root:       
        plt_Tree = np.vstack([plt_Tree,np.array([[len(plt_Tree),0,0]])])                
        f_uniq   = np.hstack([f_uniq,[root_height]])
        
    T = Tree(plt_Tree, f_uniq, range(len(f_uniq)))    
    
    return T


def area_function(VPos, M, ITris = [], f = None):
    """
    Calcola l'area di una triangolazione (i.e. 2D) o la lunghezza di un segmento.
    Nel caso del segmento ITris è una lista di indici.
    """
    
#     print(ITris.shape,ITris)

    if ITris.shape[0]==0:
        Area = 0
    elif len(ITris.shape)==1:
#         Area = 0
#         segments = [[i,i+1] for i in range(len(VPos)-1)]
#         for seg in segments:
#             x = VPos[seg[0]]
#             y = VPos[seg[1]]
#             A_tr = M[x,y]
#             Area = Area+A_tr      
#         Area = np.sum(np.diag(M,1))
        Area = np.sum(M[ITris[:-1],ITris[1:]])
        
    else:
        if f is None:
            f = np.ones((VPos.shape[0],))
        
        Area = 0
        for tr in ITris:
            
            f_mean = np.mean(f[tr])
            
#             x = VPos[tr[0]]
#             y = VPos[tr[1]]
#             z = VPos[tr[2]]
            x = tr[0]
            y = tr[1]
            z = tr[2]
            
            A = M[x,y]
            B = M[y,z]
            C = M[x,z]

            p = (A+B+C)/2

            A_tr = f_mean*(p*(p-A)*(p-B)*(p-C))**0.5
            Area = Area+A_tr  

    return Area
        
        
        

def sublvl_set_filtration_multiplicity(f, values, D, varepsilon, matrix=0,
                                       root_is_max = False, root_height = None,
                                       metric='euclidean',eps=1e-15,ITris = []):
    """
    f: funzione scritta come vettore di N_pts valori
    values: valori assunti dalla funzione messi in ordine crescente
    D: matrice dei punti del dominio (N_pts,dimensione dominio)
    varepsilon: parametro che serve per costruire il grafo costruendo bolle intorno ai punti del dominio
    matrix: eventuale matrice delle distanze pairwise di D
    root_is_max: la radice dell'albero è messa in corrispondenza del max della funzione; utile se uso la misura delle compo conn
    root_height : se <root_is_max == True> permette di stabilire a mano l'altezza della root
    """
    
    epsilon = varepsilon
    
    if np.max(matrix) ==0:
        M = squareform(pdist(D))
#         M=domain_to_matrix(D,metric)
    else:
        M=matrix

    N_pt=M.shape[0]
    
    if not len(f)==N_pt:
        print("La funzione ha un numero sbagliato di componenti")
   
    FILTRATION={}    
    FILTRATION[-1]=[]
    N_classes=0
    plt_tree=[]
    CLASSES={}
    MULT_fn={}
    
    F=[]
    cnt=0
    
    """
    Check che all'ultimo sottolivello il grafo risulti connesso
    """
    if len(ITris)==0:
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
    else:
        ADJ={}
        aux=np.arange(0,N_pt)
        for n in range(N_pt):
            ADJ[n]=[]
        
        for tr in ITris:
            ADJ[tr[0]].append(tr[1])
            ADJ[tr[0]].append(tr[2])

            ADJ[tr[1]].append(tr[0])
            ADJ[tr[1]].append(tr[2])

            ADJ[tr[2]].append(tr[0])
            ADJ[tr[2]].append(tr[1])

    for i,value in enumerate(values):   
        sublvl=aux[f<=value]
#         sublvl=aux[(f<value+eps)]
        
        M_aux=np.zeros((len(sublvl),len(sublvl)))
        idx_aux=np.arange(0,len(sublvl))

        ITris_aux = np.array([tr for tr in ITris if max([f[tr[0]],f[tr[1]],f[tr[2]]])<=value])       
#         print("\nIl sottolivello ad altezza %f è %s: "%(value,str(sublvl)))


        for u,pt in enumerate(sublvl):
            for v,adj in enumerate(sorted(ADJ[pt])):
                if adj in sublvl:
                    M_aux[u,idx_aux[sublvl==adj]]=1
                    M_aux[idx_aux[sublvl==adj],u]=1

#                     if M[pt,adj]<epsilon:
#                         M_aux[u,idx_aux[sublvl==adj]]=1
#                         M_aux[idx_aux[sublvl==adj],u]=1

        aux_graph=csr_matrix(M_aux)
        n_components, labels = connected_components(csgraph=aux_graph, directed=False, return_labels=True)

        CONN_COMP=[]

        pts_added = 1
  
        for r in range(n_components):
            """
            per ognuna delle componenti connesse, estraggo i punti che le appartengono nel sottolivello;
            e la aggiungo alle CONN_COMP
            """
            aux_comp=sublvl[labels==r]
            CONN_COMP.append(aux_comp)

            UNITED_CLASSES = []
            UNITED_COMPONENTS = []
            flag = 0      
            
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
                    UNITED_COMPONENTS.append(str(component))
                    if len(list(set(aux_comp) - set(component)))>0:
                        flag = 1
            """
            Se questa nuova componente è semplicemente una componente dello step precedente che si è espansa,
            allora posso soprassedere.
            Se invece sono più classi che si uniscono allora assegno a questa componente connessa, il numero della classe pari al 
            minimo delle classi che si uniscono e aggiungo il punto al plt_tree.
            Se invece la classe è nuova aumento il numero delle classi in giorno e aggiorno tutto di conseguenza.
            """
            if len(ITris) == 0:
                ITris_ = np.sort(aux_comp)
            else:
                ITris_ = np.array([tr for tr in ITris_aux if tr[0] in aux_comp])
            
#             v_comp = np.sort(aux_comp)
#             VPos_ = D[v_comp]
#             M_ = M[v_comp,:]
#             M_ = M_[:,v_comp]           
            
            A = area_function(D, M, ITris = ITris_)
            
#             MULT_fn[cnt_aux].append([value,A])
            
            
            if len(UNITED_CLASSES)==1:
                """
                aux_comp esisteva già ma non è cresciuta
                """
                CLASSES[str(aux_comp)]=min(UNITED_CLASSES)
                class_ = min(UNITED_CLASSES)-1
                cnt_rewind=-1
                
                while plt_tree[cnt_rewind][-1]!= class_ and plt_tree[cnt_rewind][-2]!= class_:
                    cnt_rewind=cnt_rewind-1
                cnt_aux = plt_tree[cnt_rewind][0]
                    
                MULT_fn[cnt_aux].append([value,A])
                                
            elif len(UNITED_CLASSES)>1:
                delta_ = 0.0000001
                united_aux = np.sort(UNITED_CLASSES)[::-1]
                classes_aux = np.array(UNITED_COMPONENTS)[np.argsort(UNITED_CLASSES)[::-1]]
                for u in range(len(UNITED_CLASSES)-1):
                    death = united_aux[u]
                    alive = united_aux[u+1]
                    plt_tree.append([cnt,int(death)-1,int(alive)-1])
                    F.append(value+u*delta_)
#                     if u==len(UNITED_CLASSES)-2:
#                         MULT_fn[cnt]=[[value,A]]
#                     else:
#                         MULT_fn[cnt]=[[value,A]]
                    MULT_fn[cnt]=[[value,delta_]]
                    cnt=cnt+1
                CLASSES[str(aux_comp)]=min(UNITED_CLASSES)
#                 plt_tree.append([cnt,int(max(UNITED_CLASSES))-1,int(min(UNITED_CLASSES))-1])
#                 F.append(value)
#                 MULT_fn[cnt]=[[value,A]]
#                 cnt+=1
            else:
                N_classes+=1
                CLASSES[str(aux_comp)]=N_classes
                plt_tree.append([cnt,int(N_classes)-1,-1])                      
                F.append(value)
                MULT_fn[cnt]=[[value,A]]
                cnt+=1
                
        FILTRATION[i]=CONN_COMP

    if root_is_max and root_height is None:
        if F[-1]<value:
            plt_tree.append([cnt,0,0])
            F.append(value)
            cnt+=1
            
    elif root_is_max:
        if root_height< np.max(f):
            print('There is a problem with the chosen root height!')
        if F[-1]<root_height:
            plt_tree.append([cnt,0,0])
            F.append(root_height)
            cnt+=1            
            
    return np.array(plt_tree),np.array(F), MULT_fn  



def prune_vertices(T, thresh, ret_names = False, only_names = False, 
                   mult=True, keep_root=False, approx = False, verbose = False):
    """
    dato un albero rimuovo i punti con size più piccola di una threshold 
    e restituisco l'albero come rimane.
    """        
    
    if mult: 
        params = np.copy(T.norms_mult)
    else:
        params = np.array([T.weights[v,T.find_father(v)[0]] for v in T.vertices])
        params = params[:-1]
    
    if thresh < np.min(params):
            if only_names == True:
                return [T.name_vertices[l] for l in T.leaves]

            T = Tree(T.plt_tree,T.f_uniq,T.name_vertices)

            if ret_names:
                return T, [T.name_vertices[l] for l in T.leaves]
            else:
                return T  
    elif len(T.leaves)==1:
        plt_tree = np.array([[0,0,-1],[1,0,0]])
        f = np.array([T.f_uniq[0],T.f_uniq[0]+0.00001])
        name_vertices=[np.max(T.vertices)]

        TT = Tree(plt_tree,f,[])  
        
        if mult:
            TT.make_mult(f = T.f)
            TT.make_norms_mult()
                
        if ret_names:
            return TT, name_vertices
        else:
            return TT        
        

    new_names = {}
    new_names[-1]=-1
    plt_tree = []
    f = []
    aux = 0
    idx=0
    new_sizes={}
    aux_cut = 1
    """
    in questo qua sotto metto i nomi vecchi dei vertici nuovi
    """
    name_vertices = []
    deleted = []

    L = T.leaves
    V = [params[i] for i in L]
    idxs_ = np.array([L[i] for i in np.argsort(V)])
    padd = np.arange(np.max(L),np.max(T.vertices),1)
    idxs = np.concatenate([idxs_,padd])

    for i in idxs_:
        fam = T.find_children(T.find_father(i)[0])
        bro = [p for p in fam if p != i][0]   
                
        if (params[i]<thresh) and (bro not in deleted):
            deleted.append(i)
            if not approx:
                break
           
    for i in range(T.plt_tree.shape[0]):
        pt = T.plt_tree[i,:]
        
#         if not ((params[i]<thresh) and (i in T.leaves) and (i in deleted)):
        if not (i in deleted):
            """
            Prendo un punto. Può essere che le componenti che nascono/muoiono/si mergiano in quel punto io non le abbia mai incontrate,
            perchè magari sono state uccise perchè non superavano la threshold. 
            Devo tenermi un elenco dei punti del vecchio albero che danno origine a punti nel nuovo albero. E sono new_names.keys().
            Lo ottengo assegnando ad ognuna di queste componenti del vecchio albero, un nuovo nome, dato dal numero della componente
            nel nuovo albero.
            """          
            if pt[2]==-1:
                new_names[pt[1]]=aux
                plt_tree.append([idx,new_names[pt[1]],new_names[pt[2]]])
                name_vertices.append(pt[0])

                f.append(T.f_uniq[i])
                aux+=1
                idx+=1    
            elif pt[1] not in new_names.keys() and pt[2] not in new_names.keys():
                new_names[pt[2]]=aux
                plt_tree.append([idx,aux,-1])
                name_vertices.append(pt[0])

                f.append(T.f_uniq[i])
                aux+=1
                idx+=1
            elif pt[1] not in new_names.keys():
                """
                se non è una foglia nel vecchio albero e il punto che sopravvive è l'unico associato ad una componente
                nel nuovo albero, allora posso fare finta di niente.
                """
                pass
            elif pt[2] not in new_names.keys():
                """
                Se invece è il punto sopravvivente che non ho mai visto, ed il punto morente è associato ad una componente
                nel nuovo albero, questa componente devo  associarla al punto sopravvivente del vecchio albero.
                """
                new_names[pt[2]]=new_names[pt[1]]

            elif new_names[pt[2]]==new_names[pt[1]]:
                pass
            else:
                """
                se invece si mergiano due componenti, che hanno dato origine a componenti anche nel nuovo albero,
                può succedere che nel nuovo albero si scambi chi vive e chi muore rispetto al vecchio.
                Chiamo 'm1' il punto morente nel nuovo albero ed 'm2' il punto che sopravvive.
                Devo poi 
                """
                
                name_vertices.append(pt[0])

                m1 = np.max([new_names[pt[1]],new_names[pt[2]]])
                m2 = np.min([new_names[pt[1]],new_names[pt[2]]])

                plt_tree.append([idx,m1,m2])

                new_names[pt[1]] = m2 
                new_names[pt[2]] = m2 

                f.append(T.f_uniq[i])
                idx+=1

                
    if keep_root and f[-1] < T.f_uniq[-1] :
        f.append(T.f_uniq[-1])
        plt_tree.append([idx,0,0])
        name_vertices.append(max(T.vertices))
        idx+=1
        
    plt_tree = np.array(plt_tree)
    f = np.array(f)

    if not idx>1:
        
        plt_tree = np.array([[0,0,-1],[1,0,0]])
        if len(f)>0:
            f = np.r_[f,[T.f_uniq[0]+0.00001]]
        else:
            f = np.array([T.f_uniq[0],T.f_uniq[0]+0.00001])
            
        name_vertices.append(np.max(T.vertices))

    if verbose == True:
        print('\nAbbiamo un albero di dimensioni: ',idx, plt_tree,f)

    if only_names== True:
        return name_vertices

    TT = Tree(plt_tree,f,[])   

    if mult:
        TT.f = T.f
        mult_TT={}
        
        for i,v in enumerate(name_vertices):
            j = TT.find_father(i)[0]
            
            if j==-1:
                aux = [0]
            else:
                w = name_vertices[j]

                path = np.array([p for p in T.paths[v,:] if p < w+1])
                aux = []

                for r in range(len(path[path>-1])-1):

                    b = np.copy(T.mult[(path[r],path[r+1])])
                    a = np.copy(aux)

                    if len(a)>0:   
#                         c = np.hstack([a[a>0][:-1],b[b>0]])
#                         padd = np.zeros((len(b)-len(c),))
#                         aux = np.concatenate([np.array([c]),np.array([padd])], axis=1)[0]
                        aux = a+b
                    else:
                        aux = np.copy(b)



            mult_TT[(i,j)] = aux

        TT.mult = mult_TT
        TT.delta = T.delta
        TT.wmax = T.wmax
        TT.grid = T.grid
        TT.f = T.f
        
        TT.make_norms_mult()

    if ret_names:
        return TT, name_vertices
    else:
        return TT

    
def prune_dendro(T,thresh,ret_names = False, only_names = False, mult=True, 
                 keep_root=False, approx = False, verbose = False):  
    
    T_aux = prune_vertices(T, thresh, ret_names, only_names, mult, keep_root,approx, verbose)
    T_aux_aux = prune_vertices(T_aux,thresh, ret_names, only_names, mult,keep_root,approx,verbose)
    
    while len(T_aux.vertices)>len(T_aux_aux.vertices):
        T_aux = T_aux_aux
        T_aux_aux = prune_vertices(T_aux,thresh, ret_names, only_names, mult, keep_root,approx, verbose)
        
    return T_aux_aux

def prune_dendro_N(T,N,keep_root=False,return_eps = False, approx=False):  
    
    if len(T.leaves)>N:
        try:
            eps = np.min([T.norms_mult[v] for v in T.leaves if T.norms_mult[v]>0])
        except:
            eps = np.min([T.norms_mult[v] for v in T.vertices if T.norms_mult[v]>0])/5

        T_aux = prune_vertices(T,eps,keep_root=keep_root,approx=approx)
        T_aux_aux = prune_vertices(T_aux,eps,keep_root=keep_root,approx=approx)

        while len(T_aux.leaves)>N:
            eps = eps*1.05
            T_aux = T_aux_aux
            T_aux_aux = prune_vertices(T_aux,eps,keep_root=keep_root,approx=approx)
        
        if return_eps:
            return T_aux,eps/2
        else:
            return T_aux
    else:
        if return_eps:
            return T,0
        else:
            return T
    

def from_cloud_to_dendro_sublvl(X, f, radius, grid=None, fun=False, root_is_max = False, root_height = None,
                                prec = 0.0000001, prune_param = None, ITris = []): 
    
    """
    X : dominio della funzione (N_pts,dim)
    f : funzione data come array ordinato di dim (N_pts,)
    radius : raggio per costruire il proximity graph
    grid : griglia per i pesi funzionali
    fun : se <True> le multiplicities sono delle funzioni
    root_is_max : decide se la radice e' l'ultimo punto topologicamente rilevante oppure il max di f.
                  Se viene scelto il max, verrà preservato se si fa del pruning in questa funzione.
    root_height : se <root_is_max == True> permette di stabilire a mano l'altezza della root
    """
    
    if len(X.shape)==1:
        aux = np.zeros((len(X),2))
        aux[:,0] = X
        X = aux
    
    MAT = squareform(pdist(X)) 
    epsilon = radius  
    I = np.array([np.min(f),np.max(f)])
    values,f= Utils_OPT.preprocess_f(f,I, prec)
#     print('Preprocessing fatto')
    plt_tree,f, MULT_fn = sublvl_set_filtration_multiplicity(f,values,X,epsilon, matrix = MAT,
                                                             root_is_max = root_is_max, root_height = root_height,
                                                             ITris = ITris)
#     print('Filtrazione costruita ',plt_tree,f, MULT_fn )
    T=Tree(plt_tree,f,np.arange(len(f)))
    T.mult = {}
    
    T.f = fun

    if fun: 
        if grid is None:
    #         grid = np.linspace(0,np.max(f)-np.min(f)+0.01,300)
            grid = np.linspace(np.min(f),np.max(f)+0.01,300)

    #     scaling = np.array((X[-1][0]-X[0][0])/len(X))
        scaling = 1

        T.grid = grid
        T.wmax = grid[-1]
        T.delta = grid[1]-grid[0]

        for key in MULT_fn.keys():
            father = T.find_father(key)[0]
            fn = np.array(MULT_fn[key])
    #         x = -fn[:,0][::-1]
    #         y = fn[:,1][::-1]
            x = fn[:,0]
            y = fn[:,1]

            if len(x)>1:
    #             f = interp1d(x-np.min(x), y, kind='linear', bounds_error = False, fill_value = 0)
                f = interp1d(x, y, kind='linear', bounds_error = False, fill_value = 0)
                mult = f(grid)
            else:
                mult = np.zeros_like(grid)
                mult[0] = y[0]

            mult[mult>0] = mult[mult>0] 
    #         mult[mult>0] = mult[mult>0][::-1] 
            T.mult[(key,father)] = mult*scaling
    #     print('Molteplicità fatta')
    #     print('Scaling: ', scaling)

        try:
            T.mult[(max(T.vertices),-1)]
        except:
            T.mult[(max(T.vertices),-1)]=0
    else:
        T.delta = 1
        T.wmax = np.sum(T.weights)/2
        T.grid = np.array([0])

        for key in MULT_fn.keys():
            father = T.find_father(key)[0]
            
            if father==-1:
                w = 0             
            else:
                w = T.weights[key,father]

            T.mult[(key,father)] = np.array([w])    
        
        
    T.make_norms_mult()
    
    if prune_param is None:
        return T 
    else:
        return prune_dendro(T,prune_param, keep_root=root_is_max)
    
    

