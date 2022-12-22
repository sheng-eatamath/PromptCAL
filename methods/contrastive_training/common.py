import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.cluster import KMeans
from ..clustering.faster_mix_k_means_pytorch import K_Means
from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.cluster_utils import my_mixed_eval, cluster_acc

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score

### =====================================================================================================
### Loss Function
### =====================================================================================================

def info_nce_logits(features, args):
    device = args.device
    
    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # B x B
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device) # B x B
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        
        Notes:
            `out` mode loss
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) ### [B*nview, D]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) ### repeat @nview times
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood for each positive pair, then sum for each query sample
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    
class SupConLossWithMembank(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLossWithMembank, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, mfeatures=None, mlabels=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        [revised]
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            
            mfeatures: hidden vector of shape [bsz, ...] from memory bank
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        ### >>>
        if mfeatures is not None:
            assert mfeatures.shape==features.shape
            assert (mlabels is not None) and (mlabels.shape==torch.Size([mfeatures.size(0)])), f'mlabels={mlabels.shape if mlabels is not None else None}'
        ### <<<

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) ### [B*nview, D]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        ### >>>
        ### append @contrast_features with @mfeatures
        if (mfeatures is not None) and (mlabels is not None):
            contrast_feature = torch.cat(torch.unbind(mfeatures, dim=1), dim=0) ### [B*nview, D]
            mask = torch.eq(labels.view(-1, 1), mlabels.view(1, -1)).float().to(device)
        ### <<<
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability, normalization
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) ### repeat @nview times
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
    


### =====================================================================================================
### Loss Helper for Contrastive Loss
### =====================================================================================================
def compute_cross_contrastive_loss_sup(q_features, q_labels, q_mask, k_features, k_labels, k_mask, sup_cont_crit=None, n_views=2):
    """ compute sup contrastive loss on labeled features
    assume @q_features and @k_features are already normalized and in the same order, @n_views=2
    """
    q_sup_con_feats, q_sup_con_labels = get_cont_features_for_sup_contrastive_loss(q_features, q_labels, q_mask)
    k_sup_con_feats, k_sup_con_labels = get_cont_features_for_sup_contrastive_loss(k_features, k_labels, k_mask)
    sup_contrastive_loss = sup_cont_crit(features=q_sup_con_feats, labels=q_sup_con_labels, mask=None, mfeatures=k_sup_con_feats, mlabels=k_sup_con_labels)
    return sup_contrastive_loss


def compute_cross_contrastive_loss_unsup(q_features, k_features, temperature=0.07, n_views=2):
    """ compute unsup contrastive loss on all features
    assume @q_features and @k_features are already normalized and in the same order, @n_views=2
    """
    device = q_features.device
    B = q_features.size(0)//2
    q_labels = k_labels = torch.arange(B, device=device).repeat(2)
    labels = (q_labels.view(-1, 1) == k_labels.view(1, -1)).float() # B x B
    similarity_matrix = torch.mm(q_features, k_features.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.size(0), dtype=torch.bool).to(device) # B x B
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    # compute loss
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / temperature
    unsup_contrastive_loss = torch.nn.CrossEntropyLoss()(logits, labels)
    return unsup_contrastive_loss, logits, labels


def get_cont_features_for_sup_contrastive_loss(features, class_labels, mask_lab):
    f1, f2 = [f[mask_lab] for f in features.chunk(2)] ### nview[B', C, H, W]
    sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) ### [B, nview, C, H, W]
    sup_con_labels = class_labels[mask_lab]
    return sup_con_feats, sup_con_labels


def get_cont_features_for_unsup_contrastive_loss(args, features, mask_lab):
    if args.contrast_unlabel_only:
        # Contrastive loss only on unlabelled instances
        f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
        con_feats = torch.cat([f1, f2], dim=0)
    else:
        # Contrastive loss for all examples
        con_feats = features
    return con_feats


### =====================================================================================================
### Forward and Test
### =====================================================================================================
def forward(x, model, projection_head=None, predict_token='cls', 
            return_z_features=False, num_prompts=5, mode='train', 
            num_cop=2, **kwargs):
    """
    forward pass, includes `cls`, `cls-vptm`, and `cop` path
    Args:
        predict_token: [cls (inference), cls-vptm (1st stage), cop (2nd stage)]
        return_z_features: if true, return embeddings from backbone
        mode: [train, test, teacher]
    """
    if predict_token=='cls':
        features = model(x)
        if projection_head is not None:
            z_features = features
            features = projection_head(z_features)
        features = F.normalize(features, dim=-1)
        if return_z_features is True:
            z_features = F.normalize(z_features, dim=-1)
            return features, z_features
        return features
    elif predict_token == 'cls-vptm':
        assert return_z_features==False, f'not implemented error'
        assert 'aux_projection_head' in kwargs.keys()
        features = model(x, True)
        shape_feature = features[:, 1:1+num_cop, :].size()
        features = [features[:, 0, :], features[:, 1:1+num_cop, :].view(shape_feature[0], -1, shape_feature[-1]).mean(dim=1)]
        if projection_head is not None: # training mode
            features[0] = projection_head(features[0])
            features[1] = kwargs['aux_projection_head'](features[1])
            features[0] = F.normalize(features[0], dim=-1)
            features[1] = F.normalize(features[1], dim=-1)
        feat = features[0]
        aux_feat = features[1]

        if projection_head is None:
            feat = F.normalize(feat, dim=-1)
            aux_feat = F.normalize(aux_feat, dim=-1)
        if mode=='train':
            return feat, aux_feat
        elif mode=='test':
            return feat
    elif predict_token in ['cop']:
        if mode=='train':
            features = model(x, True)
            features = [features[:, 0, :], features[:, 1:1+num_prompts, :]]
            
            z_features = features[0]
            z_features = F.normalize(z_features, dim=-1)
            
            features[0] = projection_head(features[0])
            features[0] = F.normalize(features[0], dim=-1)
            
            prompt_features = features[1][:, :num_cop, :].mean(dim=1)
            z_prompt_features = F.normalize(prompt_features, dim=-1)
            prompt_features = kwargs['aux_projection_head'](prompt_features)
            prompt_features = F.normalize(prompt_features, dim=-1)
            if return_z_features:
                features = [features[0], z_features, prompt_features, z_prompt_features]
            else:
                features = [features[0], prompt_features]
            return features
        elif mode=='test':
            with torch.no_grad():
                features = model(x)
                features = F.normalize(features, dim=-1)
            return features
        elif mode=='teacher':
            with torch.no_grad():
                features = model(x, True)
                features = [features[:, 0, :], features[:, 1:1+num_prompts, :]]
                
                z_features = features[0]
                z_features = F.normalize(z_features, dim=-1)
                
                features[0] = projection_head(features[0])
                features[0] = F.normalize(features[0], dim=-1)
                
                prompt_features = features[1][:, :num_cop, :].mean(dim=1)
                z_prompt_features = F.normalize(prompt_features, dim=-1)
                prompt_features = kwargs['aux_projection_head_t'](prompt_features)
                prompt_features = F.normalize(prompt_features, dim=-1)
            if return_z_features:
                features = [features[0], z_features, prompt_features, z_prompt_features]
            else:
                features = [features[0], prompt_features]             
            return features
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return


@torch.no_grad()
def test_kmeans(model, test_loader,
                epoch, save_name,
                args, use_fast_Kmeans=False, 
                predict_token='cls',
                return_silhouette=False,
                ):
    """ KMeans validaation on @model
    Args:
        return_silhouette: if true, to return unsupervised silhouette score
    """
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    with tqdm(total=len(test_loader)) as pbar:
        for batch_idx, (images, label, _) in enumerate(test_loader):
            images = images.cuda(args.device)
            feats = forward(images, model, projection_head=None, predict_token=predict_token, mode='test')
            all_feats.append(feats.detach().cpu())
            targets = np.append(targets, label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                            else False for x in label]))
            
            pbar.update(1)

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = torch.cat(all_feats, dim=0)
    if use_fast_Kmeans is True:
        begin = time.time()
        all_feats = all_feats.to(args.device)
        kmeans = K_Means(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-6, max_iterations=500, init='k-means++', 
                n_init=20, random_state=0, n_jobs=1, 
                pairwise_batch_size=None if all_feats.size(0)<args.fast_kmeans_batch_size else args.fast_kmeans_batch_size, 
                mode=None)
        kmeans.fit(all_feats)
        preds = kmeans.labels_.detach().cpu().numpy()
        end = time.time()
        print(f'time={end-begin}')
        all_feats = all_feats.detach().cpu().numpy()
    else:
        begin = time.time()
        all_feats = all_feats.numpy()
        kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0, verbose=0).fit(all_feats)
        end = time.time()
        print(f'time={end-begin}')
        preds = kmeans.labels_
        
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)
    
    res, ratio = my_mixed_eval(targets, preds, mask, all_feats)
    for k, v in res.items():
        args.writer.add_scalar(f'{save_name}-cluster/{k}', v, epoch)
        print(f'cluster/{k}={v}')
        
    if return_silhouette==True:
        return all_acc, old_acc, new_acc, res['unlabelled_sil']
    else:
        return all_acc, old_acc, new_acc
        

@torch.no_grad()
def test_kmeans_val(model, test_loader,
                epoch, save_name,
                args, use_fast_Kmeans=False, 
                predict_token='cls',
                return_silhouette=False,
                stage=2,
                ):
    """ KMeans validaation on @model, Inductive GNCD setting
    Args:
        return_silhouette: if true, to return unsupervised silhouette score
        stage: which training stage
    """
    model.eval()

    all_feats = []
    targets = np.array([])
    cls_mask = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    with tqdm(total=len(test_loader)) as pbar:
        for batch_idx, (images, label, _, labeled) in enumerate(test_loader):
            images = images.cuda(args.device)
            feats = forward(images, model, projection_head=None, predict_token=predict_token, mode='test')
            feats = F.normalize(feats, dim=-1)
            all_feats.append(feats.detach().cpu())
            targets = np.append(targets, label.cpu().numpy())
            cls_mask = np.append(cls_mask, np.array([True if x.item() in range(len(args.train_classes))
                                            else False for x in label]))
            mask = np.append(mask, np.array([x.item() for x in labeled]))

            pbar.update(1)
            
    mask = mask.astype(np.bool)
    cls_mask = cls_mask.astype(np.bool)
    
    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = torch.cat(all_feats, dim=0)
    if stage==1: ### only consider lbl data
        if use_fast_Kmeans is True:
            begin = time.time()
            all_feats = all_feats.to(args.device)
            kmeans = K_Means(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-6, max_iterations=500, init='k-means++', 
                    n_init=20, random_state=0, n_jobs=1, 
                    pairwise_batch_size=None if all_feats.size(0)<args.fast_kmeans_batch_size else args.fast_kmeans_batch_size, 
                    mode=None)
            kmeans.fit(all_feats)
            preds = kmeans.labels_.detach().cpu().numpy()
            end = time.time()
            print(f'time={end-begin}')
            all_feats = all_feats.detach().cpu().numpy()
        else:
            begin = time.time()
            all_feats = all_feats.numpy()
            kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0, verbose=0).fit(all_feats)
            end = time.time()
            print(f'time={end-begin}')
            preds = kmeans.labels_
        
        lbl_acc = cluster_acc(y_true=targets[mask], y_pred=preds[mask])
        args.writer.add_scalar('Surveillance/lbl_acc', lbl_acc, epoch)
        return lbl_acc, None, None
    else: ### compute lbl_acc + total_sil score
        if use_fast_Kmeans is True:
            begin = time.time()
            all_feats = all_feats.to(args.device)
            kmeans = K_Means(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-6, max_iterations=500, init='k-means++', 
                    n_init=20, random_state=0, n_jobs=1, 
                    pairwise_batch_size=None if all_feats.size(0)<args.fast_kmeans_batch_size else args.fast_kmeans_batch_size, 
                    mode=None)
            kmeans.fit(all_feats)
            preds = kmeans.labels_.detach().cpu().numpy()
            end = time.time()
            print(f'time={end-begin}')
            all_feats = all_feats.detach().cpu().numpy()
        else:
            begin = time.time()
            all_feats = all_feats.numpy()
            kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0, verbose=0).fit(all_feats)
            end = time.time()
            print(f'time={end-begin}')
            preds = kmeans.labels_
        

        # -----------------------
        # EVALUATE
        # -----------------------
        lbl_acc = cluster_acc(y_true=targets[mask], y_pred=preds[mask])
        args.writer.add_scalar('Surveillance/lbl_acc', lbl_acc, epoch)
        unlbl_sil = silhouette_score(all_feats[~mask], preds[~mask])
        args.writer.add_scalar('Surveillance/unlbl_sil', unlbl_sil, epoch)
        
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=cls_mask,
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                        writer=args.writer)
        
        res, ratio = my_mixed_eval(targets, preds, cls_mask, all_feats)
        for k, v in res.items():
            args.writer.add_scalar(f'{save_name}-cluster/{k}', v, epoch)
            print(f'cluster/{k}={v}')
            
        return lbl_acc, unlbl_sil, res['total_sil']


def epoch_checkpoint(model, projection_head, args, epoch=None, best=False, other_modules=None, postfix=''):
    def save_other_modules():
        if other_modules is not None:
            for k, v in other_modules.items():
                print(f'save {k} at {epoch}')
                torch.save(v.state_dict(), args.model_path[:-3] + f'_{k}_{epoch}.pt')
        return
    if epoch is None:
        if best==True:
            torch.save(model.state_dict(), args.model_path[:-3] + f'_best_{postfix}.pt')
            print("model saved to {}.".format(args.model_path[:-3] + f'_best_{postfix}.pt'))
            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best_{postfix}.pt')
            print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best_{postfix}.pt'))
            if other_modules is not None:
                for k, v in other_modules.items():
                    print(f'save {k} at {epoch}')
                    torch.save(v.state_dict(), args.model_path[:-3] + f'_{k}_best_{postfix}.pt')
        else:
            torch.save(model.state_dict(), args.model_path)
            print("model saved to {}.".format(args.model_path))
            torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')
            print("projection head saved to {}.".format(args.model_path[:-3] + '_proj_head.pt'))
            if other_modules is not None:
                for k, v in other_modules.items():
                    print(f'save {k} at {epoch}')
                    torch.save(v.state_dict(), args.model_path[:-3] + f'_{k}.pt')
    else:
        torch.save(model.state_dict(), args.model_path[:-3] + f'_{epoch}.pt')
        print("model saved to {}.".format(args.model_path[:-3] + f'_{epoch}.pt'))
        torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_{epoch}.pt')
        print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_{epoch}.pt'))
        if other_modules is not None:
            for k, v in other_modules.items():
                print(f'save {k} at {epoch}')
                torch.save(v.state_dict(), args.model_path[:-3] + f'_{k}_{epoch}.pt')
    return


@torch.no_grad()
def compute_knn_statistics_with_affinity(query_label, mb_labels, knn_affinity, eps=1e-10):
    """compute KNN precision & recall
    
    Returns:
        float: knn precision
        float: knn recall
        int: total num of positive predictions
    """
    knn_match = (query_label.view(-1, 1)==mb_labels.view(1, -1))
    precision = (knn_match[knn_affinity==1].float()+1e-10).mean() if (knn_affinity==1).sum().item()!=0 else torch.tensor(0.0).to(knn_affinity.device)
    recall = (((knn_match==1)&(knn_affinity==1)).sum(1)/(knn_match.sum(1)+1e-10)).mean()
    num = (knn_affinity==1).sum()
    return precision, recall, num


def forward_single_inkd(model, aux_model, images, loss_func=F.mse_loss, distill='feature'):
    """ forward for computing INKD loss """
    z_model = model(images)
    with torch.no_grad():
        z_aux_model = aux_model(images)
        
    z_model = F.normalize(z_model, p=2, dim=-1)
    z_aux_model = F.normalize(z_aux_model, p=2, dim=-1)
    if distill=='features':
        return loss_func(z_model, z_aux_model)


def annealing_decay(eta_max, eta_min, t, T=200):
    return eta_max - (eta_max-eta_min)*t/T
    
    
def annealing_linear_ramup(eta_min, eta_max, t, T=200):
    return eta_min + (eta_max-eta_min)*min(t, T)/T
    

### ------------------------------------------------------------------------------------------------------
#   DATASET UTILS
### ------------------------------------------------------------------------------------------------------

def my_get_transform(transform_name='imagenet', use_train=True):
    if transform_name=='imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image_size = 224
        interpolation = 3
        crop_pct = 0.875

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    else:
        raise NotImplementedError()
    return train_transform if use_train else test_transform


def get_auxiliary_dataset(dataset_name='imagenet', dataloader_name='default', batch_size=128, 
                          dataloader_param=dict(
                              num_workers=8,
                              shuffle=True,
                              )):
    """ get my auxiliary dataset
    implemented for [ImageNet, ]
    implemented for [torchvision loader, ]
    """
    if dataset_name=='imagenet':
        dataset = ImageFolder(root=r'/home/sheng/dataset/imagenet-img', transform=my_get_transform('imagenet'))
    else:
        raise NotImplementedError()
    
    if dataloader_name=='default':
        dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_param)
    else:
        raise NotImplementedError()
    return dataloader


# if __name__=='__main__':
#     query_label = torch.tensor([0,1,2,3])
#     mb_labels = torch.arange(10)
#     topk_indices = torch.tensor([
#         [0,1,2],
#         [1,2,3],
#         [2,3,0],
#         [0,2,3],``
#     ])
#     print(compute_knn_statistics(query_label, mb_labels, topk_indices))