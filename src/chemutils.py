import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from functools import reduce 
from tqdm import tqdm 
from copy import deepcopy 
import numpy as np 
import torch 
from torch.autograd import Variable
torch.manual_seed(4) 
np.random.seed(1)
import random 
random.seed(1)


def sigmoid(float_x):
    return 1.0 / (1 + np.exp(-float_x))

from scipy.stats import gmean

def logp_modifier(logp_score):
    return max(0.0,min(1.0,1/14*(logp_score+10))) 


def docking_modifier(docking_score):

    docking_score = 1/(12-4)*(-docking_score - 4)
    docking_score = max(docking_score, 0.0)
    docking_score = min(docking_score, 1.0) 
    return docking_score 

def qed_logp_fusion(qed_score, logp_score, jnk_score, gsk_score):
    logp_score = logp_modifier(logp_score)
    gmean_score = gmean([qed_score, logp_score])
    modified_score = min(1.0,gmean_score)
    return modified_score

def logp_jnk_gsk_fusion(logp_score, jnk_score, gsk_score):
    logp_score = logp_modifier(logp_score)
    return np.mean([logp_score, jnk_score, gsk_score])


def qed_logp_jnk_gsk_fusion(qed_score, logp_score, jnk_score, gsk_score):
    logp_score = logp_modifier(logp_score)
    gmean_score = gmean([qed_score, logp_score, jnk_score, gsk_score])
    modified_score = min(1.0,gmean_score)
    return modified_score

def qed_logp_jnk_gsk_fusion2(qed_score, logp_score, jnk_score, gsk_score):
    logp_score = logp_modifier(logp_score)
    return  np.mean([qed_score, logp_score, jnk_score, gsk_score])

def qed_logp_fusion(qed_score, logp_score):
    logp_score = logp_modifier(logp_score)
    gmean_score = gmean([qed_score, logp_score])
    modified_score = min(1.0, gmean_score)
    return modified_score 

def jnk_gsk_fusion(jnk_score, gsk_score):
    gmean_score = gmean([jnk_score, gsk_score])
    modified_score = min(1.0,gmean_score)
    return modified_score


def load_vocabulary():
	datafile = "data/vocabulary.txt"
	with open(datafile, 'r') as fin:
		lines = fin.readlines()
	vocabulary = [line.split()[0] for line in lines]
	return vocabulary 

vocabulary = load_vocabulary()
bondtype_list = [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE]


def ith_substructure_is_atom(i):
    substructure = vocabulary[i]
    return True if len(substructure)==1 else False

def word2idx(word):
    return vocabulary.index(word)




def smiles2fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=False)
    return np.array(fp)


## similarity of two SMILES 
def similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2) 


def similarity_matrix(smiles_lst):
    n = len(smiles_lst)
    sim_matrix = np.eye(n)
    mol_lst = [Chem.MolFromSmiles(smiles) for smiles in smiles_lst]
    fingerprint_lst = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False) for mol in mol_lst]
    for i in range(n):
        fp1 = fingerprint_lst[i]
        for j in range(i+1,n):
            fp2 = fingerprint_lst[j]
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            sim_matrix[i,j] = sim_matrix[j,i] = sim
    return sim_matrix 


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True) ### todo double check
    else:
        return None


def smiles2mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol 

## input: smiles, output: word lst;  
def smiles2word(smiles):
    mol = smiles2mol(smiles)
    if mol is None:
        return None 
    word_lst = []

    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques_smiles = []
    for clique in cliques:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        cliques_smiles.append(clique_smiles)
    atom_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    return cliques_smiles + atom_not_in_rings_list 

## is_valid_smiles 
def is_valid(smiles):
    word_lst = smiles2word(smiles)
    word_set = set(word_lst)
    return word_set.issubset(vocabulary)     


def is_valid_mol(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
    except:
        return False 
    if smiles.strip() == '':
        return False 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return False 
    return True 

def substr_num(smiles):
    mol = smiles2mol(smiles)
    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    return len(clique_lst)


def smiles2substrs(smiles):
    if not is_valid(smiles):
        return None 
    mol = smiles2mol(smiles)
    if mol is None:
        return None
    idx_lst = []

    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    # print(clique_lst)  ## [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15]]
    for clique in clique_lst:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        # print("clique_smiles", clique_smiles)  ## C1=CC=CC=C1, C1=COCC1, C1=CC=CC=C1 
        idx_lst.append(word2idx(clique_smiles))
    atom_symbol_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    atom_idx_not_in_rings_list = [atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing()]
    # print(atom_idx_not_in_rings_list)  ## [0, 1, 2, 3, 11, 12, 13, 14, 21]  nonring atom's index in molecule
    for atom in atom_symbol_not_in_rings_list:
        idx_lst.append(word2idx(atom))

    return idx_lst 



def smiles2graph(smiles):
    '''     N is # of substructures in the molecule 

    Output:
        1.
            idx_lst                 [N]      list of substructure's index
            node_mat                [N,d]
        2. 
            substructure_lst 
            atomidx_2substridx     dict 
        3. 
            adjacency_matrix        [N,N]    0/1   np.zeros((4,4))  
        4. 
            leaf_extend_idx_pair    [(x1,y1), (x2,y2), ...]
    '''

    ### 0. smiles -> mol 
    if not is_valid(smiles):
        return None 
    mol = smiles2mol(smiles)
    if mol is None:
        return None

    ### 1. idx_lst & node_mat 
    idx_lst = []
    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    # print(clique_lst)  ## [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15]]
    for clique in clique_lst:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        # print("clique_smiles", clique_smiles)  ## C1=CC=CC=C1, C1=COCC1, C1=CC=CC=C1 
        idx_lst.append(word2idx(clique_smiles))

    atom_symbol_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    atom_idx_not_in_rings_list = [atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing()]
    # print(atom_idx_not_in_rings_list)  ## [0, 1, 2, 3, 11, 12, 13, 14, 21]  nonring atom's index in molecule
    for atom in atom_symbol_not_in_rings_list:
        idx_lst.append(word2idx(atom))
    # print(idx_lst) ## [3, 68, 3, 0, 0, 0, 0, 0, 0, 1, 2, 4]  
    d = len(vocabulary)
    N = len(idx_lst)
    node_mat = np.zeros((N, d))
    for i,v in enumerate(idx_lst):
        node_mat[i,v]=1


    ### 2. substructure_lst & atomidx_2substridx     
    ###    map from atom index to substructure index 
    atomidx_2substridx = dict()
    substructure_lst = clique_lst + atom_idx_not_in_rings_list   
    ### [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15], 0, 1, 2, 3, 11, 12, 13, 14, 21] 
    ### 4:0  23:0, 22:0, ...   8:1, 7:1, 22:1, ... 16:2, 17:2, 18:2, ... 0:3, 1:4, 
    for idx, substructure in enumerate(substructure_lst):
    	if type(substructure)==list:
    		for atom in substructure:
    			atomidx_2substridx[atom] = idx 
    	else:
    		atomidx_2substridx[substructure] = idx 


    ### 3. adjacency_matrix 
    adjacency_matrix = np.zeros((N,N),dtype=np.int32)

    ####### 3.1 atom-atom bonds and atom-ring bonds
    for bond in mol.GetBonds():
    	if not bond.IsInRing():
    		a1 = bond.GetBeginAtom().GetIdx()
    		a2 = bond.GetEndAtom().GetIdx()
    		idx1 = atomidx_2substridx[a1] 
    		idx2 = atomidx_2substridx[a2]
    		adjacency_matrix[idx1,idx2] = adjacency_matrix[idx2,idx1] = 1 
    ####### 3.2 ring-ring connection 
    for i1,c1 in enumerate(clique_lst):
    	for i2,c2 in enumerate(clique_lst):
    		if i1>=i2:
    			continue 
    		if len(set(c1).intersection(set(c2))) > 0:
    			adjacency_matrix[i1,i2] = adjacency_matrix[i2,i1] = 1
    assert np.sum(adjacency_matrix)>=2*(N-1)

    leaf_idx_lst = list(np.where(np.sum(adjacency_matrix,1)==1)[0])
    M = len(leaf_idx_lst)
    extend_idx_lst = list(range(N,N+M))
    leaf_extend_idx_pair = list(zip(leaf_idx_lst, extend_idx_lst))
    ####### [(3, 12), (5, 13), (6, 14), (9, 15), (11, 16)]

    return idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair 






inf_value = 10



def smiles2differentiable_graph_v2(smiles):
    mol = smiles2mol(smiles)
    if mol is None:
        return None 
    if not is_valid(smiles):
        return None

    idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = smiles2graph(smiles)
    N = len(idx_lst) # of nodes in current molecule   
    d = len(vocabulary) 
    M = int(np.sum(np.sum(adjacency_matrix,1)==1)) # of leaf node

    is_nonleaf = np.sum(adjacency_matrix,1)>1
    is_nonleaf = np.concatenate([is_nonleaf, np.zeros((M), dtype=np.bool)])
    is_leaf = np.sum(adjacency_matrix,1)==1 
    is_leaf = np.concatenate([is_leaf, np.zeros((M), dtype=np.bool)])
    is_extend = np.concatenate([np.zeros((N), dtype=np.bool), np.ones((M), dtype=np.bool)])
    leaf_idx_lst = list(np.where(is_leaf==True)[0])
    leaf_nonleaf_lst = []  #### with edge to connect 
    for leaf_idx in leaf_idx_lst:
        for i in range(N):
            if adjacency_matrix[i,leaf_idx]==1:
                leaf_nonleaf_lst.append((leaf_idx, i))
                break 


    node_indicator_1 = np.zeros((N,d))
    node_indicator_1[node_mat==1] = inf_value 
    node_indicator_1[node_mat==0] = - inf_value
    node_indicator_2 = np.random.random((M,d))
    node_indicator = np.concatenate([node_indicator_1, node_indicator_2], 0)

    adjacency_mask = np.ones((N+M,N+M), dtype = np.bool)
    for leaf_idx,extend_idx in leaf_extend_idx_pair:
        adjacency_mask[leaf_idx, extend_idx] = False
        adjacency_mask[extend_idx, leaf_idx] = False 


    adjacency_weight = np.zeros((N+M,N+M))
    adjacency_weight.fill(-inf_value) 
    for i in range(N):
        for j in range(N):
            if adjacency_matrix[i,j]==1:
                adjacency_weight[i,j] = inf_value  
    for leaf_idx,nonleaf_idx in leaf_nonleaf_lst:
        adjacency_weight[leaf_idx, nonleaf_idx] = 0 
        adjacency_weight[nonleaf_idx, leaf_idx] = 0
    for leaf_idx,extend_idx in leaf_extend_idx_pair:
        adjacency_weight[leaf_idx, extend_idx] = 0
        adjacency_weight[extend_idx, leaf_idx] = 0  ### sigmoid(0) = 0.5  

    return (is_nonleaf, is_leaf, is_extend), node_indicator, adjacency_mask, adjacency_weight, leaf_extend_idx_pair, leaf_nonleaf_lst 



def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def add_atom_at_position(editmol, position_idx, new_atom, new_bond):
    '''
        position_idx:   index of edited atom in editmol
        new_atom: 'C', 'N', 'O', ... 
        new_bond: SINGLE, DOUBLE  
    '''
    ######  1 edit mol 
    new_atom = Chem.rdchem.Atom(new_atom)
    rwmol = deepcopy(editmol)
    new_atom_idx = rwmol.AddAtom(new_atom)
    rwmol.AddBond(position_idx, new_atom_idx, order = new_bond)
    ######  2 check valid of new mol 
    if not is_valid_mol(rwmol):
        return None  
    try:
        rwmol.UpdatePropertyCache()
    except:
        return None
    smiles = Chem.MolToSmiles(rwmol)
    assert '.' not in smiles
    return canonical(smiles)


def add_fragment_at_position(editmol, position_idx, fragment, new_bond):
    '''
        position_idx:  index of edited atom in editmol
        fragment: e.g., "C1=CC=CC=C1", "C1=CC=NC=C1", ... 
        new_bond: {SINGLE, DOUBLE}  

        Return:  
            list of SMILES
    '''  
    new_smiles_set = set()
    fragment_mol = Chem.MolFromSmiles(fragment)
    current_atom = editmol.GetAtomWithIdx(position_idx)
    neighbor_atom_set = set()  ## index of neighbor of current atom in new_mol  


    ## (A) add a bond between atom and ring 
    #### 1. initialize empty new_mol
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    #### 2. add editmol into new_mol
    old_idx2new_idx = dict()
    for atom in editmol.GetAtoms():
        old_idx = atom.GetIdx()
        new_atom = copy_atom(atom)
        new_idx = new_mol.AddAtom(new_atom)
        old_idx2new_idx[old_idx] = new_idx 
        assert old_idx == new_idx
    for bond in editmol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        i1 = a1.GetIdx()
        i2 = a2.GetIdx()
        i1_new = old_idx2new_idx[i1]
        i2_new = old_idx2new_idx[i2]
        bt = bond.GetBondType()
        new_mol.AddBond(i1_new, i2_new, bt)
        ### collect the neighbor atoms of current atom, both are in ring. 
        if (i1==position_idx or i2==position_idx) and (a1.IsInRing() and a2.IsInRing()):
            neighbor_atom_set.add(i1_new)
            neighbor_atom_set.add(i2_new)
    if neighbor_atom_set != set():
        neighbor_atom_set.remove(old_idx2new_idx[position_idx])

    #### 3. combine two components 
    #### 3.1 add fragment into new_mol
    new_atom_idx_lst = []
    old_idx2new_idx2 = dict()  ### fragment idx -> new mol idx 
    for atom in fragment_mol.GetAtoms():
        old_atom_idx = atom.GetIdx()
        new_atom = copy_atom(atom)
        new_atom_idx = new_mol.AddAtom(new_atom)
        new_atom_idx_lst.append(new_atom_idx)
        old_idx2new_idx2[old_atom_idx] = new_atom_idx 
    for bond in fragment_mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        i1 = old_idx2new_idx2[a1]
        i2 = old_idx2new_idx2[a2]
        bt = bond.GetBondType()
        new_mol.AddBond(i1, i2, bt)

    #### 3.2 enumerate possible binding atoms and generate new smiles 
    for i in new_atom_idx_lst:  ### enumeration 
        copy_mol = deepcopy(new_mol)
        copy_mol.AddBond(old_idx2new_idx[position_idx], i, new_bond)
        if is_valid_mol(copy_mol):
            try:
                copy_mol.UpdatePropertyCache()
                new_smiles = Chem.MolToSmiles(copy_mol)
                new_smiles = canonical(new_smiles)
                if new_smiles is not None:
                    assert '.' not in new_smiles
                    new_smiles_set.add(new_smiles) 
            except:
                pass  


    # if not current_atom.IsInRing() or new_bond != rdkit.Chem.rdchem.BondType.SINGLE:
    if not current_atom.IsInRing():
        return new_smiles_set


    # print(new_smiles_set)
    ## (B) share bond between rings 
    #### 1. initialize empty new_mol
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    #### 2. add editmol into new_mol
    old_idx2new_idx = dict()
    for atom in editmol.GetAtoms():
        old_idx = atom.GetIdx() 
        new_atom = copy_atom(atom)
        new_idx = new_mol.AddAtom(new_atom)
        old_idx2new_idx[old_idx] = new_idx 
        assert old_idx == new_idx 
    for bond in editmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        i1 = old_idx2new_idx[a1]
        i2 = old_idx2new_idx[a2]
        bt = bond.GetBondType()
        new_mol.AddBond(i1, i2, bt) 

    # print(Chem.MolToSmiles(new_mol))
    #### 3. fragment mol  
    ####### 3.1 find 2 common atoms and 1 bond  
    current_atom = editmol.GetAtomWithIdx(old_idx2new_idx[position_idx])
    current_atom_symbol = current_atom.GetSymbol()

    atom_lst = list(fragment_mol.GetAtoms())
    for neighbor_atom in neighbor_atom_set:
        neighbor_atom_symbol = editmol.GetAtomWithIdx(neighbor_atom).GetSymbol()
        bondtype_edit = new_mol.GetBondBetweenAtoms(neighbor_atom, old_idx2new_idx[position_idx]).GetBondType()
        for i,v in enumerate(atom_lst):
            v_idx = v.GetIdx()
            ### v1 is neighbor of v 
            for v1 in [atom_lst[i-1], atom_lst[i+1-len(atom_lst)]]: 
                v1_idx = v1.GetIdx()
                bondtype_frag = fragment_mol.GetBondBetweenAtoms(v_idx, v1_idx).GetBondType()
                # print("current:", current_atom_symbol, "neighbor:", neighbor_atom_symbol, bondtype_edit)
                # print(v.GetSymbol(), v1.GetSymbol(), bondtype_frag)
                if v.GetSymbol()==current_atom_symbol and v1.GetSymbol()==neighbor_atom_symbol and bondtype_edit==bondtype_frag: 
                    ####### 3.1 find 2 common atoms and 1 bond  
                    # print("2 common atoms and 1 bond ")
                    ############################################
                    ####### 3.2 add other atoms and bonds 
                    new_mol2 = deepcopy(new_mol)
                    old_idx2new_idx2 = dict()
                    old_idx2new_idx2[v_idx] = current_atom.GetIdx()
                    old_idx2new_idx2[v1_idx] = neighbor_atom
                    for atom in fragment_mol.GetAtoms():
                        old_idx = atom.GetIdx()
                        if not (old_idx==v_idx or old_idx==v1_idx):
                            new_atom = copy_atom(atom)
                            new_idx = new_mol2.AddAtom(new_atom)
                            old_idx2new_idx2[old_idx] = new_idx 
                    for bond in fragment_mol.GetBonds():
                        a1 = bond.GetBeginAtom()
                        a2 = bond.GetEndAtom()
                        i1 = a1.GetIdx()
                        i2 = a2.GetIdx()
                        i1_new = old_idx2new_idx2[i1]
                        i2_new = old_idx2new_idx2[i2]
                        bt = bond.GetBondType()
                        if not (set([i1,i2]) == set([v1.GetIdx(), v.GetIdx()])):
                            new_mol2.AddBond(i1_new, i2_new, bt)
                    ####### 3.2 add other atoms and bonds 
                    ####### 3.3 check validity and canonicalize
                    if not is_valid_mol(new_mol2):
                        continue 
                    try:
                        new_mol2.UpdatePropertyCache()
                        # print("success")
                    except:
                        continue 
                    new_smiles = Chem.MolToSmiles(new_mol2)
                    new_smiles = canonical(new_smiles)
                    if new_smiles is not None:
                        assert '.' not in new_smiles
                        new_smiles_set.add(new_smiles)
                    print(new_smiles)
    print(new_smiles_set)
    return new_smiles_set



def delete_substructure_at_idx(editmol, atom_idx_lst):
    edit_smiles = Chem.MolToSmiles(editmol)
    #### 1. initialize with empty mol
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    #### 2. add editmol into new_mol
    old_idx2new_idx = dict()
    for atom in editmol.GetAtoms():
        old_idx = atom.GetIdx()
        if old_idx in atom_idx_lst: 
            continue 
        new_atom = copy_atom(atom)
        new_idx = new_mol.AddAtom(new_atom)
        old_idx2new_idx[old_idx] = new_idx 
    for bond in editmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if a1 in atom_idx_lst or a2 in atom_idx_lst:
            continue 
        a1_new = old_idx2new_idx[a1]
        a2_new = old_idx2new_idx[a2]
        bt = bond.GetBondType()
        new_mol.AddBond(a1_new, a2_new, bt) 

    if not is_valid_mol(new_mol):
        return None
    try:
        new_mol.UpdatePropertyCache()
    except:
        return None 
    return new_mol, old_idx2new_idx 


def differentiable_graph2smiles_v0(origin_smiles, differentiable_graph, 
                                leaf_extend_idx_pair, leaf_nonleaf_lst, 
                                max_num_offspring = 100, topk = 3):

    new_smiles_set = set()
    #### 1. data preparation 
    origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(origin_smiles))
    origin_idx_lst, origin_node_mat, origin_substructure_lst, \
    origin_atomidx_2substridx, origin_adjacency_matrix, leaf_extend_idx_pair = smiles2graph(origin_smiles)
    node_indicator, adjacency_weight = differentiable_graph 
    N = len(origin_idx_lst)
    M = len(leaf_extend_idx_pair) 
    d = len(vocabulary)

    #### 2. edit the original molecule  
    ####### 2.1 delete & 2.2 replace 
    for leaf_idx, _ in leaf_extend_idx_pair:
        leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
        if type(leaf_atom_idx_lst)==int:  ### single atom
            new_leaf_atom_idx_lst = [leaf_atom_idx_lst]
        else:  #### ring     
            ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
            new_leaf_atom_idx_lst = []
            remaining_atoms_idx_lst = []
            for i,v in enumerate(origin_substructure_lst):
                if i==leaf_idx:
                    continue 
                if type(v)==int:
                    remaining_atoms_idx_lst.append(v)
                else: #### list 
                    remaining_atoms_idx_lst.extend(v)
            new_leaf_atom_idx_lst = [leaf_atom_idx for leaf_atom_idx in leaf_atom_idx_lst if leaf_atom_idx not in remaining_atoms_idx_lst]
        ### leaf_atom_idx_lst v.s. new_leaf_atom_idx_lst 
        ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
        result = delete_substructure_at_idx(editmol = origin_mol, atom_idx_lst = new_leaf_atom_idx_lst) 
        if result is None: 
            continue
        delete_mol, old_idx2new_idx = result
        delete_smiles = Chem.MolToSmiles(delete_mol)
        if delete_smiles is None or '.' in delete_smiles:
            continue
        delete_smiles = canonical(delete_smiles)
        new_smiles_set.add(delete_smiles)  #### 2.1 delete done
        ####  2.2 replace  a & b 
        ######### (a) get neighbor substr
        neighbor_substructures_idx = [idx for idx,value in enumerate(origin_adjacency_matrix[leaf_idx]) if value==1]
        assert len(neighbor_substructures_idx)==1 
        neighbor_substructures_idx = neighbor_substructures_idx[0]
        neighbor_atom_idx_lst = origin_substructure_lst[neighbor_substructures_idx]
        if type(neighbor_atom_idx_lst)==int:
            neighbor_atom_idx_lst = [neighbor_atom_idx_lst] 
        ######### (b) add new substructure  todo, enumerate several possibility 
        added_substructure_lst = list(np.argsort(-node_indicator[leaf_idx]))[:topk]  ### topk 
        for substructure_idx in added_substructure_lst: 
            new_substructure = vocabulary[substructure_idx]
            for new_bond in bondtype_list:
                for leaf_atom_idx in neighbor_atom_idx_lst:
                    new_leaf_atom_idx = old_idx2new_idx[leaf_atom_idx] 
                    if ith_substructure_is_atom(substructure_idx):
                        new_smiles = add_atom_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                        new_smiles_set.add(new_smiles)
                    else:
                        new_smiles_batch = add_fragment_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                                    fragment = new_substructure, new_bond = new_bond)
                        new_smiles_set = new_smiles_set.union(new_smiles_batch)



    ####### 2.3 add   todo: use adjacency_weight to further narrow scope
    for leaf_idx, extend_idx in leaf_extend_idx_pair:
        expand_prob = (adjacency_weight[leaf_idx][extend_idx] + adjacency_weight[extend_idx][leaf_idx])/2  ### [-inf, inf]
        # print("expand prob", expand_prob)
        if expand_prob < -3:
            continue 
        leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
        if type(leaf_atom_idx_lst)==int:  ### int: single atom;   else: list of integer
            leaf_atom_idx_lst = [leaf_atom_idx_lst]
        for leaf_atom_idx in leaf_atom_idx_lst:
            added_substructure_lst = list(np.argsort(-node_indicator[extend_idx]))[:topk]
            for substructure_idx in added_substructure_lst:
                new_substructure = vocabulary[substructure_idx]
                for new_bond in bondtype_list:
                    if ith_substructure_is_atom(substructure_idx):
                        new_smiles = add_atom_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                        new_smiles_set.add(new_smiles)
                    else:
                        new_smiles_batch = add_fragment_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                                    fragment = new_substructure , new_bond = new_bond)
                        new_smiles_set = new_smiles_set.union(new_smiles_batch)



    return new_smiles_set.difference(set([None]))  





def draw_smiles(smiles, figfile_name):
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToImageFile(mol, figfile_name, size = (300,180))
    return 



