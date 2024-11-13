import torch
import numpy as np
from torch import nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the support supervectors and the query supervectors of the network
    We do a 1-to-1 matching.
    """
    def __init__(self, dim, learnable_matcher):
        """Creates the matcher"""
        super().__init__()
        self.learnable_matcher = learnable_matcher
        if self.learnable_matcher:
            self.linear = nn.Sequential(
                nn.Linear(dim, dim // 2, bias=False),
                nn.ReLU(),
                nn.Linear(dim // 2, dim, bias=False),
                nn.ReLU()
            )

    def forward(self, support, query):
        """ Performs the matching
        Params:
            support: Tensor of dim [batch_size, num_supervectors, num_dims] with the support supervectors
            query: Tensor of dim [batch_size, num_supervectors, num_dims] with the query supervectors
        """
        cost_matrices = self._calc_cost_matrices(support, query)
        
        support_indices, query_indices = self._match(cost_matrices)

        support = torch.gather(support, 1, support_indices.unsqueeze(-1).repeat(1,1,support.shape[-1]))
        query = torch.gather(query, 1, query_indices.unsqueeze(-1).repeat(1,1,query.shape[-1]))
        if self.learnable_matcher:
            return support + self.linear(support), query + self.linear(query)
        else:
            return support, query
    
    def _calc_cost_matrices(self, support, query):
        norms_support = torch.norm(support, p=2, dim=-1, keepdim=True)
        norms_query = torch.norm(query, p=2, dim=-1, keepdim=True)
        cost_matrices = support @ query.transpose(1,2)
        cost_matrices = cost_matrices / (norms_support * norms_query.transpose(1, 2))
        cost_matrices = 1 - cost_matrices
        return cost_matrices
    
    @torch.no_grad()
    def _match(self, cost_matrices):
        device = cost_matrices.device
        cost_matrices = cost_matrices.cpu()
        support_indices = []
        query_indices = []
        for cost_matrix in cost_matrices:
            support_ind, query_ind = linear_sum_assignment(cost_matrix)
            support_indices.append(support_ind)
            query_indices.append(query_ind)
        support_indices = torch.from_numpy(np.array(support_indices)).to(device)
        query_indices = torch.from_numpy(np.array(query_indices)).to(device)
        return support_indices, query_indices

class NNMatcher(nn.Module):
    """
    This class computes an assignment between the support supervectors and the query supervectors of the network
    We do a 1-to-1 matching on query supervectors.
    """
    def __init__(self, dim, learnable_matcher):
        """Creates the matcher"""
        super().__init__()
        self.learnable_matcher = learnable_matcher
        if self.learnable_matcher:
            self.linear = nn.Sequential(
                nn.Linear(dim, dim // 2, bias=False),
                nn.ReLU(),
                nn.Linear(dim // 2, dim, bias=False),
                nn.ReLU()
            )

    def forward(self, support, query):
        """ Performs the matching
        Params:
            support: Tensor of dim [batch_size, num_supervectors, num_dims] with the support supervectors
            query: Tensor of dim [batch_size, num_supervectors, num_dims] with the query supervectors
        """
        cost_matrices = self._calc_cost_matrices(support, query)
        
        support_indices, query_indices = self._match(cost_matrices)

        support = torch.gather(support, 1, support_indices.unsqueeze(-1).repeat(1,1,support.shape[-1]))
        query = torch.gather(query, 1, query_indices.unsqueeze(-1).repeat(1,1,query.shape[-1]))
        if self.learnable_matcher:
            return support + self.linear(support), query + self.linear(query)
        else:
            return support, query
    
    def _calc_cost_matrices(self, support, query):
        norms_support = torch.norm(support, p=2, dim=-1, keepdim=True)
        norms_query = torch.norm(query, p=2, dim=-1, keepdim=True)
        cost_matrices = support @ query.transpose(1,2)
        cost_matrices = cost_matrices / (norms_support * norms_query.transpose(1, 2))
        cost_matrices = 1 - cost_matrices
        return cost_matrices
    
    @torch.no_grad()
    def _match(self, cost_matrices):
        cost_matrices = cost_matrices.transpose(-2,-1) # [batch_size, qry, spt]
        _, support_indices = cost_matrices.min(dim=-1)
        query_indices = torch.arange(
            0, support_indices.shape[1], dtype=torch.long
            ).unsqueeze(0).expand(support_indices.shape[0], -1).to(support_indices.device)
        return support_indices, query_indices