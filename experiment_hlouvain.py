from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import csv
import random
import sys
from copy import deepcopy

import igraph as ig      ## pip install python-igraph
import hypernetx as hnx
import hypernetx.algorithms.hypergraph_modularity as hmod
import hypernetx.algorithms.generative_models as gm
import pickle
from collections import defaultdict
#from h_louvain_decomposed import combined_louvain_constant_alpha, combined_modularity, combined_louvain_alphas
from h_louvain import hLouvain


from sklearn.metrics import adjusted_rand_score as ARI #ARI(pred, ground) (symmetric true <->pred)
from sklearn.metrics import adjusted_mutual_info_score as AMI #AMI(labels_true, labels_pred, *, average_method='arithmetic') (symmetric true <->pred)

class base_experiment:
    def __init__(
        self,
        search_methods: dict,
        savefile: str,
        abcd_input: str,
        dataset_short_name: str,
    #    random_seed: int,
        delta_iteration: float,
        delta_phase: float,
        change_mode: str, 
        after_changes: int,
        community_factor: int,
        verbosity: bool = False,
    ) -> None:
        self.search_methods = search_methods
        self.savefile = savefile
        self.abcd_input = abcd_input
        self.dataset_short_name = dataset_short_name
        self.verbosity = verbosity
    #    self.random_seed = random_seed
        self.delta_iteration = delta_iteration
        self.delta_phase = delta_phase
        self.change_mode = change_mode 
        self.after_changes = after_changes
        self.community_factor = community_factor


class Grid_search_hclustering(base_experiment):
    def __init__(self, config_filepath: str) -> None:
        self._load_config(config_filepath)
        super().__init__(
            search_methods=self.config["search_methods"],
            savefile=Path(__file__).parent / "results_df" / self.config["savefile"],
        #    random_seed=self.config["random_seed"],
            delta_iteration=self.config["delta_iteration"],
            dataset_short_name=self.config["dataset_short_name"],
            abcd_input=self.config["abcd_input"],
            delta_phase=self.config["delta_phase"],
            verbosity=self.config["verbosity"],
            change_mode=self.config["change_mode"],
            after_changes=self.config["after_changes"],
            community_factor=self.config["community_factor"],
        )
        if self.config["hmod_type"] == "strict":
            self.hmod_type = hmod.strict
        elif self.config["hmod_type"] == "majority":
            self.hmod_type = hmod.majority
        elif self.config["hmod_type"] == "linear":
            self.hmod_type = hmod.linear
        else: self.hmod_type = hmod.linear  # works as default


    def _load_config(self, config_filepath: str):
        filepath = Path(__file__).parent / "configs" / config_filepath
        with open(filepath, "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)


    def _load_GoT(self):
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

    def _load_Chung_Lu(self,n=300):
        k1 = {i : random.randint(2, 10) for i in range(n)}  ## node degrees
        k2 = {i : sorted(k1.values())[i] for i in range(n)} ## edge sizes
        H = gm.chung_lu_hypergraph(k1, k2)

        ## keep edges of size 2+
        E = []
        for e in H.edges:
            if len(H.edges[e])>1:
                E.append([str(x) for x in H.edges[e]])
        print("number of edges:",len(E))
        HG = hnx.Hypergraph(dict(enumerate(E)))

        ## pre-compute required quantities
        HG = hmod.precompute_attributes(HG)

        ## build 2-section
        ## G = hmod.two_section(HG)
        return HG


    def _load_ABCDH_from_file(self,filename):
        with open(filename,"r") as f:
            rd = csv.reader(f)
            lines = list(rd)

        print("File loaded")

        Edges = []
        for line in lines:
            Edges.append(set(line))

        HG = hnx.Hypergraph(dict(enumerate(Edges)))

        print("HG created")
        print("edges:", len(Edges))


        ## pre-compute required quantities
        HG = hmod.precompute_attributes(HG)

        print("binomials precomputed")

        ## build 2-section
        #G = hmod.two_section(HG)


        #print("2-section created")

        return HG


    def run_experiment(self):

       # HG = self._load_GoT()
        HG = self._load_ABCDH_from_file(self.abcd_input)
        dataset_short_name = self.dataset_short_name
        # HG, G = self._load_Chung_Lu(n=750)
        # results structure
        df_out = pd.DataFrame(
            columns=[
                "dataset",
                "random_seed",
                "method",
                "alphas",
                "phases",
                "communties",
                "changes",
                "iterations",
                "oc-phases",
                "oc-communties",
                "oc-changes",
                "oc-iterations",
                "h-modularity-type-maximized",
                "combined-modularity-optimized",
                "2s-modularity",
                "strict-h-modularity",
                "majority-h-modularity",
                "linear-h-modularity"
            ]
        )
        seed_no = 0
        for seed in self.config["random_seeds"]:
            seed_no+=1
            print("Experiment", seed_no, "/",len(self.config["random_seeds"]))
            
            for methods in self.search_methods:
                
                if methods["method"] == 'constant_level':
                    number_of_levels = methods["number_of_levels"]
                    alphas = []
                    
                    for i in range(number_of_levels):
                        alphas.append(i/max(1,number_of_levels-1))
                    
                    hL = hLouvain(HG,hmod_type=self.hmod_type, 
                                delta_it = self.delta_iteration, 
                                delta_phase = self.delta_phase, 
                                random_seed = seed) 

                    for i in range(number_of_levels):
                        print("alpha = ", alphas[i])
                        A, q2, alphas_out = hL.combined_louvain_alphas(alphas = [alphas[i]],
                                     change_mode=self.change_mode, community_factor=self.community_factor, after_changes=self.after_changes)

                        alphas_show = [round(alpha,2) for alpha in alphas_out]
        
    
                        df_out = pd.concat(
                            [
                                df_out,
                                pd.DataFrame(
                                    [
                                        [   
                                            #"ABCD750",
                                            dataset_short_name,
                                            # "ChungLu750",
                                            seed,
                                            methods["method"],
                                            alphas_show,
                                            str(hL.get_phase_history()),
                                            str(hL.get_communities_history()),
                                            str(hL.get_changes_history()),
                                            str(hL.get_iteration_history()),
                                            str(hL.get_oc_phase_history()),
                                            str(hL.get_oc_communities_history()),
                                            str(hL.get_oc_changes_history()),
                                            str(hL.get_oc_iteration_history()),
                                            self.config["hmod_type"],
                                            q2,
                                            hL.combined_modularity(A, self.hmod_type, 0),
                                            hL.combined_modularity(A, hmod.strict, 1),
                                            hL.combined_modularity(A, hmod.majority, 1),
                                            hL.combined_modularity(A, hmod.linear, 1)
                                        ]
                                    ],
                                    columns=df_out.columns,
                                ),
                            ],
                            ignore_index=True,
                        )

                if methods["method"] == 'increasing_levels':
                    hL = hLouvain(HG,hmod_type=self.hmod_type, 
                                delta_it = self.delta_iteration, 
                                delta_phase = self.delta_phase, 
                                random_seed = seed) 

                    for expected_number_of_iterations in methods["expected_number_of_iterations"]:
                        number_of_levels = methods["number_of_levels"]
                        start_level = methods["start_level"]
                        end_level = methods["end_level"]

                        levels = []
                        for i in range(expected_number_of_iterations):
                            levels.append(round(number_of_levels*i/max(1,expected_number_of_iterations-1)))

                        alphas = []
                        for i in range(expected_number_of_iterations):
                            alphas.append(0.01*round(100*start_level+100*(end_level-start_level)*levels[i]/number_of_levels))
                        

                        
                        A, q2, alphas_out = hL.combined_louvain_alphas(alphas = alphas, 
                                     change_mode=self.change_mode, community_factor=self.community_factor, after_changes=self.after_changes)

                        alphas_show = [round(alpha,2) for alpha in alphas_out]
            
        
                        df_out = pd.concat(
                            [
                                df_out,
                                pd.DataFrame(
                                    [
                                        [   
                                                #"ABCD750",
                                                dataset_short_name,
                                                # "ChungLu750",
                                                seed,
                                                methods["method"],
                                                alphas_show,
                                                str(hL.get_phase_history()),
                                                str(hL.get_communities_history()),
                                                str(hL.get_changes_history()),
                                                str(hL.get_iteration_history()),
                                                str(hL.get_oc_phase_history()),
                                                str(hL.get_oc_communities_history()),
                                                str(hL.get_oc_changes_history()),
                                                str(hL.get_oc_iteration_history()),
                                                self.config["hmod_type"],
                                                q2,
                                                hL.combined_modularity(A, self.hmod_type, 0),
                                                hL.combined_modularity(A, hmod.strict, 1),
                                                hL.combined_modularity(A, hmod.majority, 1),
                                                hL.combined_modularity(A, hmod.linear, 1)
                                        ]
                                    ],
                                    columns=df_out.columns,
                                ),
                            ],
                            ignore_index=True,
                        )

                if methods["method"] == 'bestNcontinued':

                    number_of_levels = methods["number_of_levels"]
                    bestN = methods["number_of_best"]
                    to_continue = []
                    hL = hLouvain(HG,hmod_type=self.hmod_type, 
                                delta_it = self.delta_iteration, 
                                delta_phase = self.delta_phase, 
                                random_seed = seed) 
                    for j in range(10):
                        alphas = []
                        scores = []
                        if j == 0:
                            for i in range(number_of_levels):
                                alphas.append([i/max(1,number_of_levels-1)])
                        else:
                            for candidate in to_continue:                        
                                for i in range(number_of_levels):
                                    if candidate[-1] != i/max(1,number_of_levels-1):
                                        new_array = deepcopy(candidate)
                                        new_array.append(i/max(1,number_of_levels-1))
                                        #print(new_array)
                                        alphas.append(new_array)
                            
                        alphas_to_evaluate = []
                        for i in range(len(alphas)):
                            #print("Nowe alphas", alphas)
                            A, q2, alphas_out = hL.combined_louvain_alphas(alphas = alphas[i],
                                     change_mode=self.change_mode, community_factor=self.community_factor, after_changes=self.after_changes)

                                #print("Score", q2)
                            alphas_show = [round(alpha,2) for alpha in alphas_out]
                            if len(alphas_out) > j+1:
                                print(len(alphas_out),j+1)
                                scores.append(hL.combined_modularity(A, self.hmod_type, 1))
                                alphas_to_evaluate.append(alphas[i])
            
                            df_out = pd.concat(
                                [
                                    df_out,
                                    pd.DataFrame(
                                        [
                                            [   
                                                #"ABCD750",
                                                dataset_short_name,
                                                # "ChungLu750",
                                                seed,
                                                methods["method"],
                                                alphas_show,
                                                str(hL.get_phase_history()),
                                                str(hL.get_communities_history()),
                                                str(hL.get_changes_history()),
                                                str(hL.get_iteration_history()),
                                                str(hL.get_oc_phase_history()),
                                                str(hL.get_oc_communities_history()),
                                                str(hL.get_oc_changes_history()),
                                                str(hL.get_oc_iteration_history()),
                                                self.config["hmod_type"],
                                                q2,
                                                hL.combined_modularity(A, self.hmod_type, 0),
                                                hL.combined_modularity(A, hmod.strict, 1),
                                                hL.combined_modularity(A, hmod.majority, 1),
                                                hL.combined_modularity(A, hmod.linear, 1)
                                            ]
                                        ],
                                        columns=df_out.columns,
                                    ),
                                ],
                                ignore_index=True,
                            )
                        if j > 0 and len(alphas_to_evaluate) > 0: 
                            scores.extend(best_scores)
                            for candidate in to_continue:
                                new_array = deepcopy(candidate)
                                new_array.append(candidate[-1])
                                alphas_to_evaluate.append(new_array)
                       
                        maxind = np.argpartition(scores, -bestN)[-bestN:]
                        to_continue = []
                        best_scores = []
                        for i in maxind:
                            to_continue.append(alphas_to_evaluate[i][0:j+1])
                            best_scores.append(scores[i])
                        print("best",best_scores)
                        print("TO_CONTINUE", to_continue)
                #    print(combined_modularity(HG,G,A, hmod.linear, 1))
                #    print(hmod.modularity(HG, A, hmod.linear))
                #    print(combined_modularity(HG,G,A, hmod.strict, 1))
                #    print(hmod.modularity(HG, A, hmod.strict))
                #    print(combined_modularity(HG,G,A, hmod.majority, 1))
                #    print(hmod.modularity(HG, A, hmod.majority))
                #    d = hmod.part2dict(A)
                #    part  = [d[i] for i in list(HG.nodes)]
                #    print(combined_modularity(HG,G,A, self.hmod_type, 0))
                #    print(G.modularity(part,weights='weight'))

                if methods["method"] == 'exhaustive':
                    hL = hLouvain(HG,hmod_type=self.hmod_type, 
                                delta_it = self.delta_iteration, 
                                delta_phase = self.delta_phase, 
                                random_seed = seed) 
                    #print(hL.get_communities_history())
                    number_of_levels = methods["number_of_levels"]
                    number_of_changes = methods["number_of_changes"]
                    checked_alphas = []
                    k=0
                    for j in range(number_of_changes + 1):
                        
                        alphas = []
                        if j == 0:
                            for i in range(number_of_levels):
                                alphas.append([i/max(1,number_of_levels-1)])
                        else:
                            for checked_alpha in checked_alphas:                        
                                for i in range(number_of_levels):
                                    if checked_alpha[-1] != i/max(1,number_of_levels-1):
                                        new_array = deepcopy(checked_alpha)
                                        new_array.append(i/max(1,number_of_levels-1))
                                        print(new_array)
                                        alphas.append(new_array)
                            
                       # print("ALPHAS",alphas)
                        for i in range(len(alphas)):
                         #   print("Nowe alphas", alphas)
                            A, q2, alphas_out = hL.combined_louvain_alphas(alphas = alphas[i],
                                     change_mode=self.change_mode, community_factor=self.community_factor, after_changes=self.after_changes)
                            
                            k+=1

                            if k % 2 == 1:
                                print("Experiment", seed_no, "/",len(self.config["random_seeds"]),"iteration", k, "/", number_of_levels**(1+number_of_changes))


                                #print("Score", q2)
                            #scores.append(combined_modularity(HG,G,A, self.hmod_type, 1))

                            alphas_show = [round(alpha,2) for alpha in alphas_out]
                           # print(alphas_show)
            
                            df_out = pd.concat(
                                [
                                    df_out,
                                    pd.DataFrame(
                                        [
                                            [   
                                                dataset_short_name,
                                                seed,
                                                methods["method"],
                                                alphas_show,
                                                str(hL.get_phase_history()),
                                                str(hL.get_communities_history()),
                                                str(hL.get_changes_history()),
                                                str(hL.get_iteration_history()),
                                                str(hL.get_oc_phase_history()),
                                                str(hL.get_oc_communities_history()),
                                                str(hL.get_oc_changes_history()),
                                                str(hL.get_oc_iteration_history()),
                                                self.config["hmod_type"],
                                                q2,
                                                hL.combined_modularity(A, self.hmod_type, 0),
                                                hL.combined_modularity(A, hmod.strict, 1),
                                                hL.combined_modularity(A, hmod.majority, 1),
                                                hL.combined_modularity(A, hmod.linear, 1)
                                            ]
                                        ],
                                        columns=df_out.columns,
                                    ),
                                ],
                                ignore_index=True,
                            )
                        for i in range(len(checked_alphas)):
                            checked_alphas[i].append(checked_alphas[i][-1])
                        for alpha in alphas:
                            checked_alphas.append(alpha[0:j+1])

        self.save(df_out)

    def load_datasets(self):
        return [5]


    def save(self, df):
        Path(self.savefile).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(self.savefile))

    def _procedureA(
        self,
        df
    ):
         return df



def main():
    gsh = Grid_search_hclustering(config_filepath=sys.argv[1])
    gsh.run_experiment()


if __name__ == "__main__":
    main()
