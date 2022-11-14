from collections import deque
from itertools import chain,permutations
import numpy as np
import Utils_OPT as Utils_OPT
#import sys
#sys.path.append('/mnt/d/Documents/TDA/Aneurisk/Trees/New_Folder/Pyomo')


class Tree(object):
    
    """Represents a Tree"""
    
    """
    Devo passargli plot_tree tutto di interi, e con -1 per le classi che nascono.
    La numerazione delle classi parte da 0
    
    """

    def __init__(self, plt_tree,f_uniq,name_vertices):
                
        self.f_uniq=f_uniq
        self.plt_tree=plt_tree
        self.dim=plt_tree.shape[0]
        self.vertices=np.arange(self.dim)
        self.weights=np.zeros((self.dim,self.dim))        


        self.make_tree(plt_tree,f_uniq) 

        if len(name_vertices)>0:
            self.name_vertices=name_vertices
        else:
            self.name_vertices=self.vertices
        
        self.edges=np.zeros((self.dim,self.dim)).astype(np.int)-1 
        self.n_edges=np.zeros((self.dim,)).astype(np.int) 

        self.make_edges()
        self.make_leaves()
        self.make_paths()
        

    def make_tree(self,plt_tree,f_uniq):    
        
        last_point=[]
    
        for n in range(self.dim):#prendo i punti uno alla volta con ascissa crescente  

            point=plt_tree[n,:]
            
            if point[2]==-1: #punto di nascita            
                last_point.append([n]) #n-esimo punto: punto + alto della componente che nasce
            else:#punto di morte: si uniscono due componenti
                if not (point[1]==point[2]): 
                    
                    pt_1=last_point[point[1]][-1] #punto prec della compo morente  
                    pt_2=last_point[point[2]][-1] #punto prec della compo sopravvivente
        
                    last_point[point[2]].append(n) #n-esimo punto: il punto + alto della componente che vince                                  
    
                    weight_1=np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt_1][0]])
                    weight_2=np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt_2][0]])
    
                    self.weights[n,pt_1]=weight_1
                    self.weights[pt_1,n]=weight_1

                    self.weights[n,pt_2]=weight_2
                    self.weights[pt_2,n]=weight_2
                    
                else:
                    pt=last_point[point[1]][-1] #punto prec della compo morente  
    
                    weight=np.abs(f_uniq[point[0]]-f_uniq[plt_tree[pt][0]])
                    
                    self.weights[n,n-1]=weight     
                    self.weights[n-1,n]=weight
    
    def find_edges(self,vertex):
        return np.where(self.weights[vertex,:]>0)[0].astype(np.int)        
            
    def make_edges(self):
        
        for v in self.vertices:
            aux=self.find_edges(v)
            self.n_edges[v]=aux.shape[0]
            self.edges[v,:self.n_edges[v]]=aux

    def find_father(self,vertex):
        aux=np.where(self.edges[vertex,
            :self.n_edges[vertex]]>vertex)[0].astype(np.int)

        if len(aux)>0:
            return np.asarray([self.vertices[self.edges[vertex,aux[0]]]])
        else:
            return np.asarray([-1]).astype(np.int)
        
    def path_to_root(self,vertex):
        
        aux=0        
        self.paths[vertex,aux]=vertex  
        father=self.find_father(vertex)
        aux=1
        
        while father[0]>-1:
            self.paths[vertex,aux]=father[0]  
            father=self.find_father(father[0])
            aux+=1
        
        self.len_paths[vertex]=aux
        
    def make_paths(self):

        self.paths=np.zeros((self.dim,self.dim)).astype(np.int)-1 
        self.len_paths=np.zeros((self.dim,)).astype(np.int) 
        
        for v in self.vertices:
            self.path_to_root(v)
    
    def find_children(self,vertex):
        aux=np.where(self.edges[vertex,
            :self.n_edges[vertex]]<vertex)[0].astype(np.int)
        if len(aux)>0:
            return np.asarray([self.vertices[self.edges[vertex,j]] for j in aux])
        else:
            return np.asarray([-1]).astype(np.int)
    
    def make_leaves(self):       
    
        leaves=[0]
        
        for v in self.vertices:
            aux=np.where(self.edges[v,:]>=0)[0]
            cnt=0
            for vert in aux:
                cnt+=np.int(self.edges[v,vert]<v)
            if cnt==0:
                leaves.append(v)                
        
        self.leaves=np.asarray(leaves[1:])
        
    def rename_plt_tree(self,plt_aux):
        
        N=len(plt_aux)
        names=np.zeros((self.dim,))
        name_cnt=0
        plot_t=np.zeros((N,3)).astype(np.int)
               
        for i in range(N):
            pt=plt_aux[i,:]
            if pt[-1]==-1:
                names[pt[1]]=name_cnt
                plot_t[i,:]=np.asarray([i,names[pt[1]],-1])
                name_cnt+=1
            else:
                plot_t[i,:]=np.asarray([i,names[pt[1]],names[pt[2]]])
                
        return plot_t                
        
    def sub_tree(self,vertex, keep_name_vertices=True):
                
        plt_aux=np.zeros((self.dim,3)).astype(np.int)
        plt_aux[0,:]=self.plt_tree[vertex,:]

        f_aux=np.zeros((self.dim,))
        f_aux[0]=self.f_uniq[vertex]
        
        name_vert=np.zeros((self.dim,)).astype(np.int)
        name_vert[0]=self.name_vertices[vertex]
        
        children=self.find_children(vertex)
        cnt=0
                
        while np.sum(children)>=0:
                        
            aux=0
            for child in np.sort(children)[::-1]:
                cnt+=1
                plt_aux[cnt]=self.plt_tree[child,:]
                f_aux[cnt]=self.f_uniq[child]
                name_vert[cnt]=self.name_vertices[child]
                
                children_aux_=self.find_children(child)

                if np.sum(children_aux_>=0):
                    if aux==0:
                        children_aux=children_aux_
                        aux+=1
                    else:
                        children_aux=np.concatenate((children_aux,children_aux_))
                        aux+=1
            
            if aux>0:
                children=np.asarray(children_aux).astype(np.int)
            else:
                children=np.asarray([-1]).astype(np.int)
            
        plt_tree=self.rename_plt_tree(plt_aux[:cnt+1][plt_aux[:cnt+1,0].argsort()])
        f_uniq=np.sort(f_aux[:cnt+1])
        
        if keep_name_vertices:
            name_vert=np.sort(name_vert[:cnt+1])
        else:
            name_vert = np.arange(0,cnt+1)
            
        T_aux = Tree(plt_tree,f_uniq.astype(np.float),name_vert.astype(np.int))
        mult_aux = {}
        
        for i in T_aux.vertices:
            j = T_aux.find_father(i)[0]
            w = T_aux.name_vertices[i]
            father_aux = self.find_father(w)[0]
            if j == -1:
                mult_aux[(i,j)] = [0]
            else:                
                mult_aux[(i,j)] = self.mult[(w,father_aux)]
                
        T_aux.mult = mult_aux
        T_aux.delta = self.delta
        T_aux.wmax = self.wmax
        T_aux.f = self.f
        T_aux.make_norms_mult()
       
        return T_aux
                                                
    def copy(self):
        return Tree(self.plt_tree,self.f_uniq*5,self.name_vertices)        
        
    def norm(self):
        return np.sum(self.weights)/2

    def make_newick(self,):
        aux_plt, aux_f = Utils_OPT.plt_Tree_to_plt_tree(self.plt_tree,self.f_uniq)
                
        self.newick = Utils_OPT.to_newick(aux_plt, aux_f ,d_1=True)

    def plot_newick(self, axes=None):
        
        self.make_newick()
        
        from Bio import Phylo    
        import io

        handle = io.StringIO(self.newick)
        tree = Phylo.read(handle, "newick")
        Phylo.draw(tree,axes=axes)
        
        
    def make_mult(self, mult=None, f=False, grid=None, delta = None, wmax = None, normalize=False):
        """
        mult: un dizionario con dentro le molteplicità; una per ogni edge
        grid: la griglia su cui valutare le funzioni di molteplicità
        delta: il passo della griglia
        wmax: il max della griglia (massimo del peso)
        """
        self.mult = {}
        N = 300
        
        self.f = f
        
        if self.f == True:
            if normalize == False:
                self.tot_mass = 1
            else:
                self.tot_mass = len(self.leaves)

            if grid is None:
                if wmax is None:
                    self.wmax = np.max(np.sum(self.weights, axis=1))*1.2
                else:
                    self.wmax = wmax            
                if delta is None:
                    self.delta = self.wmax/N 
                else:
                    self.delta = delta
                self.grid = np.arange(0, self.wmax, self.delta)
            else:
                self.grid = grid
                self.wmax = grid[-1]          
                self.delta = grid[1]-grid[0] 

            for i in self.vertices:
                father=self.find_father(i)[0]

                if father==-1:
                    w = np.array([0])
                elif mult is None:
                    grid = np.arange(0,self.weights[i,father],self.delta)
                    padd = np.arange(self.weights[i,father],self.wmax,self.delta)
                    w = np.hstack([(len(self.sub_tree(i).leaves)/self.tot_mass)*\
                                   np.ones_like(grid),np.zeros_like(padd)])                
                else:
                    w = mult[[i,father]]

                self.mult[(i,father)]=w
    #             self.mult[[father,i]]=w
        else:
            
            self.delta = 1
            self.wmax = np.sum(self.weights)/2
            self.grid = np.array([0])
            
            for i in self.vertices:
                father=self.find_father(i)[0]

                if father==-1:
                    w = 0             
                else:
                    w = self.weights[i,father]

                self.mult[(i,father)]=np.array([w])
                

    def make_norms_mult(self,):
        
        self.norms_mult = np.zeros_like(self.vertices, dtype = np.float)
        
        if self.f:
            for v in self.vertices:
                if v==max(self.vertices):
                    try:
                        fn = self.mult[(v,self.find_father(v)[0])]
                        self.norms_mult[v]=np.linalg.norm(fn, ord = 1)*self.delta
                    except:
                        self.norms_mult[v]=0
                else:
                    fn = self.mult[(v,self.find_father(v)[0])]
                    self.norms_mult[v]=np.linalg.norm(fn, ord = 1)*self.delta
        else:
            for v in self.vertices:
                self.norms_mult[v] = self.weights[(v,self.find_father(v)[0])]



    def cut_tree(self,h,return_clusters = True, return_subtrees = False, sort = True):
    
        if h<self.f_uniq[-1]:  
            children = [np.max(self.vertices)]
            c = children[0]

            cluster_nodes = [] # sono le roots dei subtrees associate ai clusters

            while len(children)>0:

                c = children[0]
                children = children[1:]

                c_tmp = self.find_children(c)

                for tmp in c_tmp:
                    if tmp>-1:
                        if self.f_uniq[tmp]>h:
                            children.append(tmp)
                        else:
                            cluster_nodes.append(tmp)
        else:
            cluster_nodes = [np.max(self.vertices)]
 
        clusters = []

        for c in cluster_nodes:
            T_aux = self.sub_tree(c)
            clus = [T_aux.name_vertices[i] for i in T_aux.leaves]
            clusters.append(clus)

        if sort:
            clusters = np.array(clusters,dtype=object)
            cluster_nodes = np.array(cluster_nodes)
            n = [len(c) for c in clusters]
            idxs = np.argsort(n)[::-1]

            clusters = clusters[idxs]
            cluster_nodes = cluster_nodes[idxs]    

        if return_subtrees:
            trees = []

            for c in cluster_nodes:
                T_aux = self.sub_tree(c)
                trees.append(T_aux)

            return clusters, trees

        elif return_clusters:
            return clusters

        else:
            return cluster_nodes



