##### Readme File #####


## Folders ##

Data -> Contains the datasets with the embedding coordinates after the two PCA pipelines. 
		"df_PCA.csv" and "df_PCAGroups.csv" contain the 48 patients for which we have the clinical variables.
		"principalDf_p.csv" and "principalDfgroups_p.csv" contain all the 55 patients.
		
Results ->  The folder in which the outputs of the notebooks are stored. A copy of the outputs is already stored (along with a backup in the backup folder)
			"pruning.npy" and "edit.npy" are the outputs (pairwise distance matrices) of the simulation study.
			"principalDf_p_pruning_2.5_15.npy" and "principalDfgroups_p_pruning_2.5_15.npy" are the 55x55 pairwise distance matrices of the case study.
			"clustering_tree_pruned_pca.csv" and "clustering_tree_pruned_group_pca.csv" are the labels of the clusters of the case study.
			
Trees -> Contains the files needed to work with merge trees/dendrograms. 
		 "Trees_OPT.py" contains the python class used to represent merge trees/dendrograms.
		 "top_TED_lineare_multiplicity.py" contains the function <top_TED_lineare> which computes d_E between two such objects.
		 "Utils_dendrograms_OPT.py" contains the functions needed to obtain dendrograms from point clouds or (possibly multivariate) functions.
		 "Utils_OPT.py" contains other auxiliary functions.

		 
## Files ##
		 
The three jupyter notebooks contained in the main folder are needed to run: 
1) the simulation study "Simulation_study.ipynb"
2) the 55x55 pairwise distance matrices "Make_Metric_Matrices.ipynb"
3) the cluster analysis on those 55x55 matrices "Cluster_Analysis.ipynb"

The file "req.txt" cointains the environment requirements needed to run the code.


## Actions Required ##

A linear integer solver is needed to run the code. The solver is specified at the lines 575-576 of "top_TED_lineare_multiplicity.py".
There are two ways to specify the solver:
1) choose a solver from the pyomo library e.g. "pyo.SolverFactory('glpk')"
2) insert the address of the binaries of a solver available on the machine e.g. "pyo.SolverFactory('cplex', executable='address')"

