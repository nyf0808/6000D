import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy 
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import SequentialSampler
import matplotlib.pyplot as plt
import numpy as np 
sigmoid = torch.nn.Sigmoid() 
from tqdm import tqdm 

from gnn_layer import GraphConvolution, GraphAttention
from chemutils import smiles2graph, vocabulary 

torch.manual_seed(4) 
np.random.seed(1)

# def sigmoid(x):
#     return 1/(1+np.exp(-x))
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, n_out, num_layer):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features = nfeat, out_features = nhid)
        self.gcs = [GraphConvolution(in_features = nhid, out_features = nhid) for i in range(num_layer)]
        self.gc2 = GraphConvolution(in_features = nhid, out_features = n_out)
        # self.dropout = dropout
        from chemutils import vocabulary 
        self.vocabulary_size = len(vocabulary) 
        self.nfeat = nfeat 
        self.nhid = nhid 
        self.n_out = n_out 
        self.num_layer = num_layer 
        # self.embedding = nn.Embedding(self.vocabulary_size, nfeat)
        self.embedding = nn.Linear(self.vocabulary_size, nfeat)
        self.criteria = torch.nn.BCEWithLogitsLoss() 
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.99))
        self.device = device 
        self = self.to(device) 

    def switch_device(self, device):
        self.device = device 
        self = self.to(device)

    def forward(self, node_mat, adj, weight):
        '''
            N: # substructure  &  d: vocabulary size

        Input: 
            node_mat:  
                [N,d]     row sum is 1.
            adj:    
                [N,N]
            weight:
                [N]  

        Output:
            scalar   prediction before sigmoid           [-inf, inf]
        '''
        node_mat, adj, weight = node_mat.to(self.device), adj.to(self.device), weight.to(self.device)
        x = self.embedding(node_mat)
        x = F.relu(self.gc1(x,adj))
        for gc in self.gcs:
            x = F.relu(gc(x,adj))
        x = self.gc2(x, adj)
        logits = torch.sum(x * weight.view(-1,1)) / torch.sum(weight)
        return logits 
        ## without sigmoid 

    def smiles2embed(self, smiles):
        idx_lst, node_mat, substructure_lst, atomidx_2substridx, adj, leaf_extend_idx_pair = smiles2graph(smiles)
        idx_vec = torch.LongTensor(idx_lst).to(device)
        node_mat = torch.FloatTensor(node_mat).to(device)
        adj = torch.FloatTensor(adj).to(device)
        weight = torch.ones_like(idx_vec).to(device)
        
        ### forward 
        node_mat, adj, weight = node_mat.to(self.device), adj.to(self.device), weight.to(self.device)
        x = self.embedding(node_mat) ## bug 
        x = F.relu(self.gc1(x,adj))
        for gc in self.gcs:
            x = F.relu(gc(x,adj))
        return torch.mean(x, 0)

    def smiles2twodim(self, smiles):
        embed = self.smiles2embed(smiles)
          

    def smiles2pred(self, smiles):
        idx_lst, node_mat, substructure_lst, atomidx_2substridx, adj, leaf_extend_idx_pair = smiles2graph(smiles)
        idx_vec = torch.LongTensor(idx_lst).to(device)
        node_mat = torch.FloatTensor(node_mat).to(device)
        adj = torch.FloatTensor(adj).to(device)
        weight = torch.ones_like(idx_vec).to(device)
        logits = self.forward(node_mat, adj, weight)
        pred = torch.sigmoid(logits) 
        return pred.item() 


    def update_molecule(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np):
        node_mask = torch.BoolTensor(node_mask_np).to(self.device)
        node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

        pred_lst = []
        # for i in tqdm(range(5000)): ### 5k 10k
        for i in range(5000): ### 5k 10k

            node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
            adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
            opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

            normalized_node_mat = torch.softmax(node_indicator, 1)
            normalized_adjacency_weight = torch.sigmoid(adjacency_weight)
            node_weight = torch.sum(normalized_adjacency_weight, 1)
            node_weight = torch.clamp(node_weight, max=1) 
            node_weight[node_mask] = 1 
            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)

            # target_y = Variable(torch.Tensor([max(sigmoid(pred_y.item()) + 0.05, 1.0)])[0], requires_grad=True)
            target_y = Variable(torch.Tensor([1.0])[0])
            cost = self.criteria(pred_y, target_y)
            opt_mol.zero_grad()
            cost.backward()
            opt_mol.step()

            node_indicator_np2, adjacency_weight_np2 = node_indicator.detach().numpy(), adjacency_weight.detach().numpy()
            node_indicator_np2[node_mask_np,:] = node_indicator_np[node_mask_np,:]
            adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

            if i%500==0:
                pred_lst.append(pred_y.item())

        # print('prediction', pred_lst)
        # return node_indicator, adjacency_weight  ### torch.FloatTensor 
        return node_indicator_np2, adjacency_weight_np2  #### np.array 

    def update_molecule_interpret(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np):
        node_mask = torch.BoolTensor(node_mask_np).to(self.device)
        node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

        pred_lst = []
        # for i in tqdm(range(5000)): ### 5k 10k
        for i in range(5000): ### 5k 10k

            node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
            adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
            opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

            normalized_node_mat = torch.softmax(node_indicator, 1)
            normalized_adjacency_weight = torch.sigmoid(adjacency_weight)
            node_weight = torch.sum(normalized_adjacency_weight, 1)
            node_weight = torch.clamp(node_weight, max=1) 
            node_weight[node_mask] = 1 
            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)

            # target_y = Variable(torch.Tensor([max(sigmoid(pred_y.item()) + 0.05, 1.0)])[0], requires_grad=True)
            target_y = Variable(torch.Tensor([1.0])[0])
            cost = self.criteria(pred_y, target_y)
            opt_mol.zero_grad()
            cost.backward()
            opt_mol.step()

            if i==0:
                node_indicator_grad = node_indicator.grad.detach().numpy()
                adjacency_weight_grad = adjacency_weight.grad.detach().numpy() 
            # print(node_indicator.grad.shape)
            # print(adjacency_weight.grad.shape)

            node_indicator_np2, adjacency_weight_np2 = node_indicator.detach().numpy(), adjacency_weight.detach().numpy()
            node_indicator_np2[node_mask_np,:] = node_indicator_np[node_mask_np,:]
            adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

            if i%500==0:
                pred_lst.append(pred_y.item())

        # print('prediction', pred_lst)
        # return node_indicator, adjacency_weight  ### torch.FloatTensor 
        return node_indicator_np2, adjacency_weight_np2, node_indicator_grad, adjacency_weight_grad  #### np.array 


    def learn(self, node_mat, adj, weight, target):
        pred_y = self.forward(node_mat, adj, weight)
        pred_y = pred_y.view(-1)
        cost = self.criteria(pred_y, target)
        self.opt.zero_grad() 
        cost.backward() 
        self.opt.step() 
        return cost.data.numpy(), pred_y.data.numpy() 

    def valid(self, node_mat, adj, weight, target):
        pred_y = self.forward(node_mat, adj, weight)
        pred_y = pred_y.view(-1)
        cost = self.criteria(pred_y, target)
        return cost.data.numpy(), pred_y.data.numpy() 












