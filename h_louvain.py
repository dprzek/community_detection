from distutils.util import change_root
import pandas as pd
import numpy as np
import igraph as ig      ## pip install python-igraph
import hypernetx as hnx
import hypernetx.algorithms.hypergraph_modularity as hmod
import hypernetx.algorithms.generative_models as gm
import pickle
from collections import defaultdict
import copy
import csv


def linear(d, c):
    return c / d if c > d / 2 else 0

def majority(d, c):
    return 1 if c > d / 2 else 0

def strict(d, c):
    return 1 if c == d else 0

# calculation of exponential part of binomial
def bin_ppmf(d, c, p):
    return p**c * (1 - p)**(d - c)

# degree tax for node subset of given total volume (sum of nodes' weights)


class hLouvain:
    def __init__(
        self,
        HG,
        hmod_type = linear,
        delta_it = 0.00001, 
        delta_phase = 0.00001, 
        random_seed = 123
    ) -> None:
        self.HG = HG
        self.hmod_type = hmod_type
        self.delta_it = delta_it
        self.delta_phase = delta_phase
        self.random_seed = random_seed
        self.G = hmod.two_section(HG)
        self.startHGdict, self.startA = self.build_HG_dict_from_HG()
        self.HGdict = {},
        self.dts = defaultdict(float) #dictionary containing degree taxes for given volumes
        self.changes = 0
        self.communities = len(self.startA)
        self.phase = 1
        self.iteration = 1
        self.phase_history = []
        self.community_history = []
        self.changes_history = []
        self.iteration_history = []
        self.onchange_phase_history = []
        self.onchange_community_history = []
        self.onchange_changes_history = []
        self.onchange_iteration_history = []
        self.change_mode = "iter"



    def get_phase_history(self):
        return self.phase_history

    def get_changes_history(self):
        return self.changes_history

    def get_communities_history(self):
        return self.community_history

    def get_iteration_history(self):
        return self.iteration_history

    def get_oc_phase_history(self):
        return self.onchange_phase_history

    def get_oc_changes_history(self):
        return self.onchange_changes_history

    def get_oc_communities_history(self):
        return self.onchange_community_history

    def get_oc_iteration_history(self):
        return self.onchange_iteration_history

    def _setHGDictAndA(self):
        self.HGdict = copy.deepcopy(self.startHGdict)
        A = copy.deepcopy(self.startA)
        return A

    def _hg_neigh_dict(self):
        """
        Optimizing the access to nodes neighbours by additional dictionary based on hnx neighbours() function

        Parameters
        HG : an HNX hypergraph

        Returns
        -------
        : dict
        a dictionary with {node: list of neighboring nodes}
        """

        result = {}
        for v in self.HG.nodes:
            result[v] = self.HG.neighbors(v)
        return result


    def _neighboring_clusters(self, S, P): 
        """
        Given a list of sets (supernodes), a dictionary (partition), and given a hypergraph returns 
        a dictionary with lists of neighbouring clusters indexes for each supernode

        Parameters
        ----------
        S : a list of sets
        P : a partition of the vertices
        HG : an HNX hypergraph

        Returns
        -------
        : dict
        a dictionary with {supernode: list of neighboring cluster indices}
        """
        result = {}
        for i in range(len(S)):
            result[i]=[]
            for v in S[i]:
                for w in self.HG.neighbors_dict[v]:
                    result[i].append(P[w])
            result[i] = list(set(result[i]))
        return result


    #Naive procedure based on full modularity calculation (based on standard functions of hnx and igraph libraries)

    def combined_modularity(self,A, hmod_type=linear, alpha=0.5):
        
        # Exclude empty clusters if exist
        A = [a for a in A if len(a) > 0]
        
        # Calculation of hypergraph modularity (based on hypernetx)
        hyper = hmod.modularity(self.HG,A, wdc=hmod_type) # we use linear (default)  - other predifined possibilities are called majority/strict)
        
        # Calculation of modularity of 2-section graph (based on iGraph)
        d = hmod.part2dict(A)
        partition  = [d[i] for i in list(self.HG.nodes)]
        twosect = self.G.modularity(partition,weights='weight')  # weighting is enabled
        
        return (1-alpha)*twosect+alpha*hyper



    # calculating edge contribution loss when taking supernode s from source community c



    # wdc weighs calculation for "linear" hypergraph modularity 
    #def wdc(d, c):
    #    return c / d if c > d / 2 else 0




#JESTEM TUTAJ


    def _ec_loss(self, c, s, wdc=linear):
        
        ec_h = 0  # ec for hypergraph
        ec_2s = 0  #ec for 2section graph
        
        #Edge-by-edge accumulation of edge contribution loss (for hypergraph and 2section)
        for e in self.HGdict["v"][s]["e"].keys():
            d = self.HGdict["e"][e]["size"]
            if d > 1:  #we ignore hyperedges with one vertex
                w = self.HGdict["e"][e]["weight"]
                old_c = self.HGdict["c"][c]["e"][e] # counting vertices of edge e in community c
                new_c = old_c - self.HGdict["v"][s]["e"][e] #updating after taking the supernode     
                ec_h += w * (wdc(d, old_c)- wdc(d, new_c))
                ec_2s += w * self.HGdict["v"][s]["e"][e] * new_c/(d-1)
        return ec_h / self.HG.total_weight, 2*ec_2s / self.HG.total_volume 

    # calculating edge contribution gain when joinig supernode s to destination community c

    def _ec_gain(self, c, s, wdc=linear):
        

        ec_h = 0
        ec_2s = 0
        
        #Edge-by-edge accumulation of edge contribution change 
        for e in self.HGdict["v"][s]["e"].keys():
            d = self.HGdict["e"][e]["size"]
            if d > 1:  #we ignore hyperedges with one vertex
                w = self.HGdict["e"][e]["weight"]
                old_c = self.HGdict["c"][c]["e"][e]
                new_c = old_c + self.HGdict["v"][s]["e"][e] 
                ec_h += w * (wdc(d, new_c) - wdc(d, old_c))
                ec_2s += w * self.HGdict["v"][s]["e"][e] * old_c/(d-1)
        
        return ec_h / self.HG.total_weight, 2*ec_2s / self.HG.total_volume 


    def _degree_tax_hyper(self, volume, wdc=linear):
        
        volume = volume/self.HG.total_volume
        
        DT = 0
        for d in self.HG.d_weights.keys():
            x = 0
            for c in np.arange(int(np.floor(d / 2)) + 1, d + 1):
                x += self.HG.bin_coef[(d, c)] * wdc(d, c) * bin_ppmf(d, c, volume)
            DT += x * self.HG.d_weights[d]
        return DT / self.HG.total_weight

    def _combined_delta(self,delta_h, delta_2s,alpha=0.5):
        return (1-alpha)*delta_2s+alpha*delta_h

    #procedure for building dictionary-based data structures for hypergraph with supernodes
    #procedure produce the initial structure based on the input hypergraph with all supernodes with just only node

    def build_HG_dict_from_HG(self):
        
        HGdict = {}
        HGdict["v"]={}  #subdictionary storing information on vertices
        HGdict["c"]={}  #subdictionary storing information on current communities
        A = []          #current partition as a list of sets
        name2index = {} 
        
        # preparing structures for vertices and initial communities
        
        for i in range(len(self.HG.nodes)):
            HGdict["v"][i]={}
            HGdict["v"][i]["name"] = [list(self.HG.nodes)[i]]
            HGdict["v"][i]["e"] = defaultdict(int) #dictionary  edge: edge_dependent_vertex_weight   
            HGdict["v"][i]["strength"]=0   #strength = weigthed degree'
            HGdict["c"][i]={}
            HGdict["c"][i]["e"] = defaultdict(int) #dictionary  edge: edge_dependent_community_weight
            HGdict["c"][i]["strength"]=0
            A.append({list(self.HG.nodes)[i]})
            name2index[list(self.HG.nodes)[i]] = i

        # preparing structures for edges 
            
        HGdict["e"] = {}
        edge_dict = self.HG.incidence_dict
        
        HGdict["total_volume"] = 0    # total volume of all vertices             
        HGdict["total_weight"] = sum(self.HG.edges[j].weight for j in self.HG.edges) #sum of edges weights
        for j in range(len(self.HG.edges)):
            HGdict["e"][j] = {}
            HGdict["e"][j]["v"] = defaultdict(int)  #dictionary  vertex: edge_dependent_vertex_weight
            HGdict["e"][j]["weight"] = self.HG.edges[j].weight
            HGdict["e"][j]["size"] = self.HG.size(j)
            HGdict["e"][j]["volume"] = HGdict["e"][j]["weight"]*HGdict["e"][j]["size"]    
            for t in self.HG.incidence_dict[j]:
                HGdict["e"][j]["v"][name2index[t]] += 1 
                HGdict["v"][name2index[t]]["e"][j] += 1
                HGdict["c"][name2index[t]]["e"][j] += 1
                HGdict["v"][name2index[t]]["strength"] += HGdict["e"][j]["weight"]
                HGdict["c"][name2index[t]]["strength"] += HGdict["e"][j]["weight"]  
                HGdict["total_volume"] += HGdict["e"][j]["weight"]
                
        return HGdict, A


    # procedure for node_collapsing
    # procedure produce supervertices based on communities from the previous phase

    def node_collapsing(self,HGd,A):
        
        # reindexing the partition from previous phase (skipping empty indexes)
        newA = [a for a in A if len(a) > 0]
        
        # building new vertices and communities (by collapsing nodes to supernodes)
        
        HGdict = {}
        HGdict["v"]={}
        HGdict["c"]={}
        i = 0
        for j in range(len(A)):
            if len(A[j])>0:
                HGdict["v"][i]={}
                HGdict["c"][i]={}
                HGdict["v"][i]["e"]={}
                HGdict["c"][i]["e"]=defaultdict(int)
                for e in HGd["c"][j]["e"].keys():
                    if HGd["c"][j]["e"][e] > 0:
                        HGdict["v"][i]["e"][e] = HGd["c"][j]["e"][e]
                        HGdict["c"][i]["e"][e] = HGd["c"][j]["e"][e]
                HGdict["v"][i]["strength"] = HGd["c"][j]["strength"]
                HGdict["c"][i]["strength"] = HGd["c"][j]["strength"]
                i+=1        

        #updating edges based on new supernodes indexes and weights
        
        HGdict["e"] = HGd["e"]
        HGdict["total_volume"] = HGd["total_volume"]
        HGdict["total_weight"] = HGd["total_weight"]
        for j in range(len(HGdict["e"])):
            HGdict["e"][j]["v"] = {}
            HGdict["e"][j]["v"] = defaultdict(int)  
        for v in HGdict["v"].keys():
            for e in HGdict["v"][v]["e"].keys():
                HGdict["e"][e]["v"][v] = HGdict["v"][v]["e"][e] 
                
        return HGdict, newA


    def next_maximization_iteration(self, L, DL, A1, D, 
                alpha = 0.5):
        
        current_alpha = alpha
        A = copy.deepcopy(A1)
        superneighbors = self._neighboring_clusters(L,D)
        #print("Supernodes permuted!")
        for sn in list(np.random.RandomState(seed=self.random_seed).permutation(L)): #permute supernodes (L contains initial partition, so supernodes)
            #check the current cluster number (using the first node in supernode)
            c = D[list(sn)[0]] 
            #check the supernode index
            si = DL[list(sn)[0]] 
            
            # matrix with modularity deltas
            
            M = []       
            
            if len(superneighbors[si]) > 1 or c not in superneighbors[si]: #checking only clusters with some neighbors of current supernodes
                
                
                # calculating loss in degree tax for hypergraph when taking vertex si from community c
                vols = self.HGdict["v"][si]["strength"]
                volc = self.HGdict["c"][c]["strength"]

            #    try:
            #        dt_h_loss = self.dts[volc]
            #    except KeyError:
            #        self.dts[volc] = self._degree_tax_hyper(volc, wdc=self.hmod_type)
            #        dt_h_loss = self.dts[volc]

            #    try:
            #        dt_h_loss -= self.dts[volc-vols]
            #    except KeyError:
            #        self.dts[volc-vols] = self._degree_tax_hyper(volc - vols, wdc=self.hmod_type)
            #        dt_h_loss -= self.dts[volc-vols]
                

                if not self.dts[volc]:
                    self.dts[volc] = self._degree_tax_hyper(volc, wdc=self.hmod_type)
                if volc > vols and not self.dts[volc - vols]:
                    self.dts[volc-vols] = self._degree_tax_hyper(volc - vols, wdc=self.hmod_type)
                
                dt_h_loss = self.dts[volc] - self.dts[volc-vols] 
                
                # calculating loss in edge contribution for hypergraph and 2section when taking vertex si from community c
                
                ec_h_loss, ec_2s_loss = self._ec_loss( c, si, wdc=self.hmod_type)
                                
                for i in superneighbors[si]:   
                    if c == i:
                        M.append(0) 
                    else:
                        # gain in degree tax for hypergraph when joining vertex si to community i
                
                        voli = self.HGdict["c"][i]["strength"]
                   
                    #    try:
                    #        dt_h_gain = self.dts[voli + vols]
                    #    except KeyError:
                    #        self.dts[voli + vols] = self._degree_tax_hyper(voli + vols, wdc=self.hmod_type)
                    #        dt_h_gain = self.dts[voli + vols]

                    #    try:
                    #        dt_h_gain-= self.dts[voli]
                    #    except KeyError:
                    #        self.dts[voli] = self._degree_tax_hyper(voli, wdc=self.hmod_type)
                    #        dt_h_gain -= self.dts[voli]
                        
                   
                        if not self.dts[voli]:
                            self.dts[voli] = self._degree_tax_hyper(voli, wdc=self.hmod_type)
                        if not self.dts[voli + vols]:
                            self.dts[voli + vols] = self._degree_tax_hyper(voli + vols, wdc=self.hmod_type)
                        dt_h_gain = self.dts[voli + vols] - self.dts[voli]
                        
                        # change in degree tax for 2section graph
                        delta_dt_2s =2*vols*(vols+voli-volc)/(self.HG.total_volume**2)
                        
                        # gain in edge contribution when joining si to community i
                        ec_h_gain, ec_2s_gain = self._ec_gain(i, si, wdc=self.hmod_type)
                        
                        #calulating deltas
                        delta_h = ec_h_gain - ec_h_loss - (dt_h_gain - dt_h_loss)
                        delta_2s = ec_2s_gain - ec_2s_loss - delta_dt_2s
                        
                        M.append(self._combined_delta(delta_h, delta_2s, current_alpha))
                # make a move maximizing the gain 
                if max(M) > 0:
                    i = superneighbors[si][np.argmax(M)]
                    for v in list(sn):
                        A[c] = A[c] - {v}
                        A[i] = A[i].union({v})
                        D[v] = i
                    #recalculate neighboring clustes
                    superneighbors = self._neighboring_clusters(L,D)

                    self.changes+=1
                    self.new_changes+=1
                    if len(A[c]) == 0:
                        self.communities-=1


                    if self.change_mode == "modifications":
                        if self.new_changes >= self.after_changes:
                            self.level+=1
                            self.new_changes = 0
                            if self.level > len(self.alphas) - 1:
                                self.alphas.append(self.alphas[-1])
                            current_alpha = self.alphas[self.level]
                            #print("change alpha from", self.alphas[self.level-1],"to", self.alphas[self.level], "changes", self.changes)

                            #print("communities", self.communities)
                            #print("changes", self.changes)
                            self.onchange_phase_history.append(self.phase)
                            self.onchange_changes_history.append(self.changes)
                            self.onchange_community_history.append(self.communities)
                            self.onchange_iteration_history.append(self.iteration)


                    if self.change_mode == "communities":
                        if self.communities <= self.community_threshold:
                            self.level+=1
                            self.community_threshold = self.community_threshold/self.community_factor
                            if self.level > len(self.alphas) - 1:
                                self.alphas.append(self.alphas[-1])
                            current_alpha = self.alphas[self.level]
                            #print("change alpha from", self.alphas[self.level-1],"to", self.alphas[self.level], "communities", self.communities)

                            #print("communities", self.communities)
                            #print("changes", self.changes)
                            self.onchange_phase_history.append(self.phase)
                            self.onchange_changes_history.append(self.changes)
                            self.onchange_community_history.append(self.communities)
                            self.onchange_iteration_history.append(self.iteration)

                    
                    for e in self.HGdict["v"][si]["e"].keys():
                        self.HGdict["c"][c]["e"][e] -= self.HGdict["v"][si]["e"][e]
                        self.HGdict["c"][i]["e"][e] += self.HGdict["v"][si]["e"][e]
                        self.HGdict["c"][c]["strength"] -= self.HGdict["v"][si]["e"][e]*self.HGdict["e"][e]["weight"]
                        self.HGdict["c"][i]["strength"] += self.HGdict["v"][si]["e"][e]*self.HGdict["e"][e]["weight"]            
        q2 = self.combined_modularity(A, self.hmod_type, current_alpha)

        return q2, A, D
        


    def next_maximization_phase_alphas(self, L):

        #print("")
        #print("START MAXIMIZATION PHASE")
        DL = hmod.part2dict(L)    
        # copy L to A (the partition to be modified)
        # L contains initial partition (supernodes)
        # A will change during the step
        A1 = L[:]  
        D = hmod.part2dict(A1) #current partition as a dictionary (D will change during the phase)
        
        self.HG.total_volume = self.HGdict["total_volume"]
        # compute neighboring clusters for supernodes
        superneighbors = self._neighboring_clusters(L,D)
        # calculate modularity at the phase start

        if self.change_mode == "iter":
            if self.level > len(self.alphas) - 1:
                self.alphas.append(self.alphas[-1])
        qC = self.combined_modularity(A1, self.hmod_type, self.alphas[self.level])
        #print("q at phase start:",qC)
        
        
        #self.HG.dts = defaultdict(float) #dictionary containing degree taxes for given volumes
        
        while True:
         #   print("A1",A1)
            q2, A2, D =  self.next_maximization_iteration(L, DL, A1, D, self.alphas[self.level])
        #   print("level",level, "alpha",alphas[level])
        #   print("after iteration q2", level, alphas[level], q2)
        #    print("A10",A1)
        #    print("A2",A2)
            qC = self.combined_modularity(A1, self.hmod_type, self.alphas[self.level]) 
        #    print(qC)
        #    print(q2)

            if (q2 - qC) < self.delta_it:
                #print("Phase stoped")
                #print("communities", self.communities)
                #print("changes", self.changes)
                #qC = q2
                self.HGdict, newA = self.node_collapsing(self.HGdict,A2)
        #       print(qC)
                break
            #print("Phase continued")
            #print("communities", self.communities)
            #print("changes", self.changes)
            self.phase_history.append(self.phase)
            self.changes_history.append(self.changes)
            self.community_history.append(self.communities)
            self.iteration_history.append(self.iteration)
            self.iteration += 1
            if self.change_mode == "iter":
                self.level+=1
                self.onchange_phase_history.append(self.phase)
                self.onchange_changes_history.append(self.changes)
                self.onchange_community_history.append(self.communities)
                self.onchange_iteration_history.append(self.iteration)

                if self.level > len(self.alphas) - 1:
                    self.alphas.append(self.alphas[-1])
                
        #   print("before iteration qC", level, alphas[level], qC)
            A1 = A2
        
        return newA, qC 




    def combined_louvain_alphas(self,alphas = [1], change_mode = "iter", after_changes = 300, community_factor = 2):

        A1 = self._setHGDictAndA()

        self.changes = 0
        self.new_changes = 0
        self.communities = len(A1)
        self.phase = 1
        self.phase_history = []
        self.community_history = []
        self.changes_history = []
        self.iteration_history = []
        self.onchange_phase_history = []
        self.onchange_community_history = []
        self.onchange_changes_history = []
        self.onchange_iteration_history = []
        self.change_mode = change_mode
        self.after_changes = after_changes
        self.alphas = alphas
        self.level = 0
        self.iteration = 1
        self.community_factor = community_factor
        self.community_threshold = self.communities/self.community_factor
        
        self.HG.neighbors_dict = self._hg_neigh_dict()
        q1 = 0
        while True:
            A2, qnew  = self.next_maximization_phase_alphas(A1)
            q1 = self.combined_modularity(A1,  self.hmod_type, self.alphas[self.level])
            #q2 = self.combined_modularity(A2,  self.hmod_type, self.alphas[newlevel]) 
            #print(len(self.dts))
            A1 = A2
            if (qnew-q1) < self.delta_phase:
                return A2, qnew, self.alphas[0:self.level+1]
            q1 = qnew
            self.phase+=1
            self.iteration = 1


def load_ABCDH_from_file(filename):
    with open(filename,"r") as f:
        rd = csv.reader(f)
        lines = list(rd)

    print("File loaded")

    Edges = []
    for line in lines:
        Edges.append(set(line))

    HG = hnx.Hypergraph(dict(enumerate(Edges)))

    print("HG created")

    print(len(Edges), "edges")
    ## pre-compute required quantities
    HG = hmod.precompute_attributes(HG)

    print("binomials precomputed")

    ## build 2-section
    #G = hmod.two_section(HG)


    #print("2-section created")

    return HG


def load_GoT():
    ## load the GoT dataset
    Edges, Names, Weights = pickle.load(open( "../../hypernetx/utils/toys/GoT.pkl", "rb" ))
    print(len(Names),'nodes and',len(Edges),'edges')

    ## Nodes are represented as strings from '0' to 'n-1'
    HG = hnx.Hypergraph(dict(enumerate(Edges)))
    ## add edge weights
    for e in HG.edges:
        HG.edges[e].weight = Weights[e]
    ## add full names
    for v in HG.nodes:
        HG.nodes[v].name = Names[v]

    ## compute node strength (add unit weight if unweighted), d-degrees, binomial coefficients
    HG = hmod.precompute_attributes(HG)
    ## build 2-section
    ## G = hmod.two_section(HG)
    return HG


def main():


    HG = load_ABCDH_from_file("results_3000_he.txt")
    #HG = load_GoT()

    hL = hLouvain(HG,hmod_type=hmod.linear)

    A, q2, alphas_out = hL.combined_louvain_alphas(alphas = [0.5,1,0.5,0.7,0.8,1], change_mode="communities", community_factor= 2)
        
    print("")
    print("FINAL ANSWER:")
    print("partition:",A)
    print("qC:", q2)
    print("alphas", alphas_out)



if __name__ == "__main__":
    main()
