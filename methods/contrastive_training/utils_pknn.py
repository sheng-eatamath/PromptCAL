
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import *


### =================================================================================
###                                 pKNN method
### =================================================================================

def compute_consensus_gpu(N, K, knn_map):
    """
    compute consensus KNN with range @K on neighborhood @knn_map computed on data
    """
    consensus = torch.zeros(N, N).to(knn_map.device)
    idx = torch.cartesian_prod(torch.arange(K), torch.arange(K)).to(knn_map.device)
    idx = idx[idx[:, 0]!=idx[:, 1]] ### exclude self-self
    for i in range(N):
        consensus[knn_map[i, idx[:, 0]], knn_map[i, idx[:, 1]]] += 1
        consensus[i, knn_map[i, :]] += 1
        consensus[knn_map[i, :], i] += 1
    return consensus


@torch.no_grad()
def graph_diffusion(A, alpha=-1, P=None, step=1):
    """
    TPG diffusion process
    """
    assert isinstance(A, torch.Tensor), 'A should be torch.Tensor'
    Q = A.clone()
    P = A if P is None else P
    for t in range(step):
        Q = (P @ Q) @ P.T + torch.eye(Q.size(0), device=Q.device)
    return Q


@torch.no_grad()
def compute_diffusion_neighbor_consensus_pseudo_knn(features, features_mb, diffusion_params, k=20, transition_prob=None):
    """
    compute SemiAG on the graph generated from student @features and memory bank @features_mb
    SemiAG is controlled by @diffusion_params and neighborhood size @k
    """
    
    def compute_consensus_on_features(features, features_mb, k):
        B = features.size(0)
        stack_features = torch.cat([features, features_mb], dim=0) ### B+M, D
        K = k
        similarity, knn_ind, _ = compute_cosine_knn(stack_features, k=min(stack_features.size(0), K+1))
        knn_map = knn_ind[:, 1:]
        N = knn_map.shape[0]
        consensus = compute_consensus_gpu(N, K, knn_map)
        return similarity, consensus
    
    diffusion_steps = diffusion_params['diffusion_steps']
    q = diffusion_params['q']
    
    B = features.size(0)
    similarity, consensus = compute_consensus_on_features(features, features_mb, k)
    consensus = consensus/(consensus.sum(1, keepdims=True)+1e-10)
    ### perform graph diffusion
    diff_transition = graph_diffusion(consensus, step=diffusion_steps, P=transition_prob)
    ### thresholding
    T = diff_transition[diff_transition.nonzero(as_tuple=True)].mean()
    q = np.quantile(diff_transition[diff_transition>T].detach().cpu().numpy(), q=q)
    knn_affinity = (diff_transition>q)
    ### compute on samples from student
    similarity = similarity[:B, B:]
    knn_affinity = knn_affinity[:B, B:]
    return similarity.detach(), knn_affinity.detach(), consensus[:B, B:].detach()


@torch.no_grad()
def compute_pseudo_knn(features, features_mb, method='neighbor_consensus_diffusion', k=20, **kwargs):
    """
    Returns:
        knn_affinity
    """
    if method=='neighbor_consensus_diffusion':
        return compute_diffusion_neighbor_consensus_pseudo_knn(features, features_mb, diffusion_params=kwargs, k=k)
    else:
        raise NotImplementedError()
    return



def compute_cosine_knn(features, k=5):
    """ compute 2d feature KNN query
    Returns:
        knn_ind
        knn_val
    """
    assert len(features.shape)==2, f'not implemented for shape={features.shape}'
    
    similarity = torch.mm(features, features.T)
    k_query = similarity.topk(k=k)
    knn_ind = k_query.indices
    knn_val = k_query.values
    return similarity, knn_ind, knn_val
    

@torch.no_grad()
def compute_pos_affinity_with_lmask(class_labels, mask_lab, labels_mb, lmask_mb):
    """
    compute positive/negative affinity (@pos_affinity/@mutex_affinity) that 
    must hold from @class_labels and @mask_lab
    """
    label_match = (class_labels.repeat(2).view(-1, 1)==labels_mb.view(1, -1)) ### 2B x M
    mask_match = (mask_lab.int().repeat(2).view(-1, 1)==1) * (lmask_mb.int().view(1, -1)==1)
    pos_affinity = (label_match*mask_match)
    mutex_affinity = (label_match==0)&(mask_match==1)
    return pos_affinity, mutex_affinity

@torch.no_grad()
def update_knn_affinity(pos_affinity, mutex_affinity, knn_affinity, neg_affinity):
    """
    update predicted KNN affinity (@knn_affinity and @neg_affinity) with 
    must-hold ground-truth affinities (@pos_affinity and @mutex_affinity)
    """
    a = (pos_affinity==1)&(knn_affinity==0) ### neglected positives
    b = (mutex_affinity==1)&(knn_affinity==1) ### false positives
    c = (pos_affinity==1)&(neg_affinity==1) ### false negatives
    knn_affinity[a] = 1
    neg_affinity[a] = 0 # mutex
    knn_affinity[b] = 0 
    neg_affinity[b] = 1 # mutex
    knn_affinity[c] = 1
    neg_affinity[c] = 0 # mutex
    return knn_affinity, neg_affinity



def compute_knn_loss(knn_affinity, neg_affinity, features, features_mb, features_me, 
                     temperature=0.07, k=20, epoch=None, 
                     ):
    """ compute KNN loss on all features, 2*B x (Q + 1) pairs
    all features assumed to be already normalized, topK neighbors are computed
    
    Args:
        knn_affinity (Tensor[2*B, M]): affinity map with selected knn `positive` pairs
        neg_affinity (Tensor[2*B, M]): affinity map with selected knn `negative` pairs
        features (Tensor[2*B, D]): features from query encoder
        features_mb (Tensor[M, D]): features from memory bank
        features_me (Tensor[2*B, D]): features from momentum encoder, serving as positive pair for @features
        k (int): number of neighbors
    Returns:
        loss
    """
    assert ((knn_affinity+neg_affinity)==2).sum()==0, 'ERROR:: violate the mutex condition'
    
    device = features.device
    N, D = features.size()
    M = features_mb.size(0)
    
    ### compute similarity matrix
    pos = torch.cosine_similarity(features, features_me).unsqueeze(1)
    similarity_matrix = torch.mm(features, features_mb.T)
    ### compute label map
    label_map = torch.zeros_like(similarity_matrix).to(device) ### B x [Mpos, Mneg]
    label_map[knn_affinity==1] = 1
    label_map[neg_affinity==1] = -1
    
    ### concat with positive (different view)
    similarity_matrix = torch.cat([pos, similarity_matrix], dim=1)/temperature
    label_map = torch.cat([torch.ones_like(pos).to(device), label_map], dim=1)
    valid_map = label_map.clone().abs().detach()
    
    # for numerical stability
    similarity_matrix_copy = similarity_matrix.clone()
    similarity_matrix_copy[valid_map==0] = -2
    logits_max, _ = torch.max(similarity_matrix_copy, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    # compute log_prob
    exp_logits = torch.exp(logits)
    exp_logits = exp_logits*(valid_map!=0) ### nonzero for only pos/neg
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    log_prob = log_prob*(valid_map!=0) ### nonzero for only pos/neg
    label_map = label_map*(label_map!=-1) ### nonzero for only pos
    # compute mean of log-likelihood for each positive pair, then sum for each query sample
    mean_log_prob_pos = (label_map * log_prob).sum(1) / label_map.sum(1)
    loss = - mean_log_prob_pos.mean()
    return loss
    

@torch.no_grad()
def negative_sampling_from_membank(knn_affinity, similarity, neg_samples=-1):
    """ random sampling on negative samples
    
    Returns:
        neg_affinity: 1 for neg, 0 for neglected pos
        neg_samples: for `random` method, N neg = @neg_samples - N sim
    """
    if neg_samples==-1: ### use all
        return (~knn_affinity)
    
    device = knn_affinity.device
    B, M = knn_affinity.size()

    neg_affinity = torch.zeros_like(knn_affinity).to(device)
    qi, kj = (knn_affinity==0).nonzero(as_tuple=True) ### neg ind
    for i in range(B):
        row_select = (qi==i)
        assert row_select.sum()>0, 'ERROR:: neg not exists'
        idx_select = torch.randperm(row_select.sum().int().item()).to(device)[:neg_samples-knn_affinity[i, :].sum()]
        neg_affinity[qi[row_select][idx_select], kj[row_select][idx_select]] = 1
    return neg_affinity
    

