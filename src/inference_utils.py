
### 1. import
import numpy as np 
from tqdm import tqdm 
from matplotlib import pyplot as plt
import pickle 
from random import shuffle 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tdc import Oracle
torch.manual_seed(1)
np.random.seed(2)
import random 
random.seed(1)
from chemutils import * 

from dpp import DPPModel



def gnn_prediction_of_single_smiles(smiles, gnn):
	if not is_valid(smiles):
		return 0
	return gnn.smiles2pred(smiles)



def oracle_screening(smiles_set, oracle):
	smiles_score_lst = []
	for smiles in smiles_set:
		score = oracle(smiles)
		smiles_score_lst.append((smiles, score))
	smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
	return smiles_score_lst 

def dpp(smiles_score_lst, num_return, lamb):
	smiles_lst = [i[0] for i in smiles_score_lst]
	if len(smiles_lst) <= num_return:
		return smiles_lst, None, None 
	score_arr = np.array([i[1] for i in smiles_score_lst])
	sim_mat = similarity_matrix(smiles_lst)
	dpp_model = DPPModel(smiles_lst = smiles_lst, sim_matrix = sim_mat, f_scores = score_arr, top_k = num_return, lamb = lamb)
	smiles_lst, log_det_V, log_det_S = dpp_model.dpp()
	return smiles_lst, log_det_V, log_det_S 


def gnn_screening(smiles_set, gnn):
	smiles_score_lst = []
	for smiles in smiles_set:
		score = gnn_prediction_of_single_smiles(smiles, gnn)
		smiles_score_lst.append((smiles, score))
	smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
	return smiles_score_lst
	# smiles_lst = [i[0] for i in smiles_score_lst]
	# return smiles_lst

def optimize_single_node(smiles):
	assert substr_num(smiles)==1 
	vocabulary = load_vocabulary()
	atoms = ['N', 'C']
	



def optimize_single_molecule_one_iterate(smiles, gnn):
	if not is_valid(smiles):
		return set()
	(is_nonleaf, is_leaf, is_extend), node_indicator, adjacency_mask, adjacency_weight, leaf_extend_idx_pair, leaf_nonleaf_lst = smiles2differentiable_graph_v2(smiles)
	node_mask = is_nonleaf + is_leaf
	differentiable_graph = gnn.update_molecule(node_mask, node_indicator, adjacency_mask, adjacency_weight)
	smiles_set = differentiable_graph2smiles_v0(origin_smiles = smiles, differentiable_graph = differentiable_graph, 
											 leaf_extend_idx_pair = leaf_extend_idx_pair, leaf_nonleaf_lst = leaf_nonleaf_lst)
	return smiles_set

def random_optimizing_single_moleccule_one_iteration(smiles, gnn, ):
	if not is_valid(smiles):
		return set()
	(is_nonleaf, is_leaf, is_extend), node_indicator, adjacency_mask, adjacency_weight, leaf_extend_idx_pair, leaf_nonleaf_lst = smiles2differentiable_graph_v2(smiles)
	node_mask = is_nonleaf + is_leaf
	differentiable_graph = gnn.update_molecule(node_mask, node_indicator, adjacency_mask, adjacency_weight)
	smiles_set = differentiable_graph_to_smiles_purely_randomwalk(origin_smiles = smiles, differentiable_graph = differentiable_graph, 
											 leaf_extend_idx_pair = leaf_extend_idx_pair, leaf_nonleaf_lst = leaf_nonleaf_lst,)
	return smiles_set




def optimize_single_molecule_all_generations(input_smiles, gnn, oracle, generations, population_size, lamb):
	smiles2f = dict() 
	traceback_dict = dict() 
	input_smiles = canonical(input_smiles)
	input_score = oracle(input_smiles)
	best_mol_score_list = []
	existing_set = set([input_smiles])
	current_mol_score_list = [(input_smiles, input_score)]
	for it in tqdm(range(generations)):
		new_smiles_set = set()
		for smiles,score in current_mol_score_list:
			proposal_smiles_set = optimize_single_molecule_one_iterate_v2(smiles, gnn)
			proposal_smiles_set = proposal_smiles_set.difference(set([input_smiles]))
			for new_smiles in proposal_smiles_set:
				if new_smiles not in traceback_dict:
					traceback_dict[new_smiles] = smiles 
			new_smiles_set = new_smiles_set.union(proposal_smiles_set)
		existing_set = existing_set.union(new_smiles_set)
		mol_score_list = oracle_screening(new_smiles_set, oracle)

		for smiles, score in mol_score_list:
			if score > 0.50:
				print('example', smiles, score)
		best_mol_score_list.extend(mol_score_list)
		smiles_lst = dpp(mol_score_list, num_return = population_size, lamb = lamb)
		current_mol_score_list = [(smiles,0.0) for smiles in smiles_lst]


	best_mol_score_list.sort(key=lambda x:x[1], reverse=True) 
	return best_mol_score_list, input_score, traceback_dict 



def calculate_results(input_smiles, input_score, best_mol_score_list):
	if best_mol_score_list == []:
		with open(result_file, 'a') as fout:
			fout.write("fail to optimize" + input_smiles + '\n')
		return None 
	output_scores = [i[1] for i in best_mol_score_list]
	smiles_lst = [i[0] for i in best_mol_score_list]
	with open(result_file, 'a') as fout:
		fout.write(str(input_score) + '\t' + str(output_scores[0]) + '\t' + str(np.mean(output_scores[:3]))
				 + '\t' + input_smiles + '\t' + ' '.join(smiles_lst[:3]) + '\n')
	return input_score, output_scores[0]

def inference_single_molecule(input_smiles, gnn, result_file, generations, population_size):
	best_mol_score_list, input_score, traceback_dict = optimize_single_molecule_all_generations(input_smiles, gnn, oracle, generations, population_size)
	return calculate_results(input_smiles, input_score, result_file, best_mol_score_list, oracle)




def inference_molecule_set(input_smiles_lst, gnn, result_file, generations, population_size):
	score_lst = []
	for input_smiles in tqdm(input_smiles_lst):
		if not is_valid(input_smiles):
			continue 
		result = inference_single_molecule(input_smiles, gnn, result_file, generations, population_size)
		if result is None:
			score_lst.append(None)
		else:
			input_score, output_score = result
			score_lst.append((input_score, output_score))
	return score_lst








