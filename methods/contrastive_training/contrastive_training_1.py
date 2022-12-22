import argparse
import os
import time

from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader


from config import exp_root, dino_pretrain_path, ibot_pretrain_path

from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.cluster_utils import mixed_eval, AverageMeter
from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, seed_torch

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits, get_datasets_with_gcdval

from models import vision_transformer as vits
from models import vpt_vision_transformer as vpt_vit
from models.model_create import create_dino_backbone
from ..clustering.faster_mix_k_means_pytorch import K_Means
from .common import *
from guohao.mymeter import MyMeter


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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



def info_nce_logits(features, args):
    """ self-supervised contrastive loss """
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
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def train(projection_head, model, train_loader, test_loader, unlabelled_train_loader, args, aux_dataloader=None, aux_model=None, val_loader=None):

    model_params = list(projection_head.parameters()) + list(model.parameters())

    sup_con_crit = SupConLoss()
    best_test_acc_lab = 0
    i_iter = 0
    
    iter_aux_dataloader = iter(aux_dataloader)
    aux_projection_head = aux_model[1]
    aux_model = aux_model[0]
    aux_model.eval()
    aux_projection_head.train()
    model_params += list(aux_projection_head.parameters())

    
    optimizer = SGD(model_params, lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )
            
    for epoch in range(args.epochs):

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()
        mymeter = MyMeter()

        projection_head.train()
        model.train()
                    
        ### training
        with tqdm(total=len(train_loader)) as pbar:
            for batch_idx, batch in enumerate(train_loader):

                images, class_labels, uq_idxs, mask_lab = batch
                mask_lab = mask_lab[:, 0] ### B
                class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
                images = torch.cat(images, dim=0).to(device) ### [nview*B, 3, H, W]
                    
                features = forward(images, model, projection_head, 
                                   predict_token=args.predict_token, 
                                   num_prompts=args.num_dpr,
                                   num_cop=args.num_dpr,
                                   aux_projection_head=aux_projection_head,
                                   )
                
                loss = 0.0
                
                ### unpack features
                aux_loss = 0
                aux_features = features[1]
                features = features[0]
                
                ### DPR loss
                contrastive_logits, contrastive_labels = info_nce_logits(features=aux_features, args=args)
                unsup_contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # Supervised contrastive loss
                f1, f2 = [f[mask_lab] for f in aux_features.chunk(2)] ### nview[B', C, H, W]
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) ### [B, nview, C, H, W]
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
                
                # Total loss
                aux_loss += (1 - args.sup_con_weight) * unsup_contrastive_loss + args.sup_con_weight * sup_con_loss
                loss += args.w_prompt_clu * aux_loss
                mymeter.add('aux_loss', aux_loss.item())
                    
                    
                # Choose which instances to run the contrastive loss on
                if args.contrast_unlabel_only:
                    # Contrastive loss only on unlabelled instances
                    f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                    con_feats = torch.cat([f1, f2], dim=0)
                else:
                    # Contrastive loss for all examples
                    con_feats = features

                contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args)
                unsup_contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # Supervised contrastive loss
                f1, f2 = [f[mask_lab] for f in features.chunk(2)] ### nview[B', C, H, W]
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) ### [B, nview, C, H, W]
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
                
                # Total loss
                loss += (1 - args.sup_con_weight) * unsup_contrastive_loss + args.sup_con_weight * sup_con_loss
                
                # INKD
                if epoch<args.inkd_T:
                    try:
                        aux_images, _ = next(iter_aux_dataloader)
                        aux_images = aux_images.to(args.device)
                    except StopIteration as e:
                        iter_aux_dataloader = iter(aux_dataloader)
                        aux_images, _ = next(iter_aux_dataloader)
                        aux_images = aux_images.to(args.device)
                    inkd_loss = forward_single_inkd(model, aux_model, aux_images, loss_func=F.mse_loss, distill='features')
                    loss += max(0, annealing_decay(args.w_inkd_loss, args.w_inkd_loss_min, epoch, args.inkd_T)) * inkd_loss
                    mymeter.add('inkd_loss', inkd_loss.item())
                else:
                    mymeter.add('inkd_loss', 0)


                # Record
                _, pred = contrastive_logits.max(1)
                acc = (pred == contrastive_labels).float().mean().item()
                train_acc_record.update(acc, pred.size(0))
                loss_record.update(loss.item(), class_labels.size(0))
                
                mymeter.add('unsup_contrastive_loss', unsup_contrastive_loss.item())
                mymeter.add('sup_con_loss', sup_con_loss.item())
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                
                # optimize
                optimizer.step()
                
                
                pbar.update(1)
                pbar.set_postfix(
                    loss=loss_record.avg,
                    train_acc_record=train_acc_record.avg,
                    inkd_loss=mymeter.mean('inkd_loss'),
                    unsup_contrastive_loss=mymeter.mean('unsup_contrastive_loss'),
                    sup_con_loss=mymeter.mean('sup_con_loss'),
                    aux_loss=mymeter.mean('aux_loss'),
                    epoch=epoch,
                )
                
                i_iter += 1

       # Step schedule
        exp_lr_scheduler.step()
        
        torch.save(model.state_dict(), args.model_path)
        print("model saved to {}.".format(args.model_path))
        torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')
        print("projection head saved to {}.".format(args.model_path[:-3] + '_proj_head.pt'))

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss/unsup_contrastive_loss', mymeter.mean('unsup_contrastive_loss'), epoch)
        args.writer.add_scalar('Loss/sup_con_loss', mymeter.mean('sup_con_loss'), epoch)
        args.writer.add_scalar('Loss/inkd_loss', mymeter.mean('inkd_loss'), epoch)
        args.writer.add_scalar('Loss/total', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)
        
        print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg))
        
        # ----------------
        # clustering
        # ----------------
        if epoch%args.eval_interval==(args.eval_interval-1):
            if epoch%args.kmeans_interval==(args.kmeans_interval-1):
                with torch.no_grad():
                    print('Testing on unlabelled examples in the training data...')
                    all_acc, old_acc, new_acc = test_kmeans(model, unlabelled_train_loader,
                                                            epoch=epoch, save_name='Train ACC Unlabelled',
                                                            args=args, predict_token='cls')
                    print('Testing on disjoint test set...')
                    all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader,
                                                                        epoch=epoch, save_name='Test ACC',
                                                                        args=args, predict_token='cls')
                    print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))
                    print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                                            new_acc_test))

                    if args.use_val==True:
                        print('Testing on val set...')
                        lbl_score, unlbl_score, total_sil = test_kmeans_val(model, val_loader,
                                                                            epoch=epoch, save_name='Val ACC',
                                                                            args=args, use_fast_Kmeans=False, 
                                                                            predict_token='cls', 
                                                                            return_silhouette=True,
                                                                            stage=2,
                                                                            )
                        old_acc_test = lbl_score
                    
                args.writer.add_scalar('Surveillance/val_score', old_acc_test, epoch)
                if (old_acc_test > best_test_acc_lab):
                    print(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
                    print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))
                    torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
                    print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
                    torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
                    print("projection head saved to {}.".format(args.model_path[:-3] + f'_aux_proj_head_best.pt'))
                    torch.save(aux_projection_head.state_dict(), args.model_path[:-3] + f'_aux_proj_head_best.pt')
                    best_test_acc_lab = old_acc_test
            else:
                with torch.no_grad():
                    print('Testing on unlabelled examples in the training data...')
                    all_acc, old_acc, new_acc = test_kmeans(model, unlabelled_train_loader,
                                                            epoch=epoch, save_name='Fast Train ACC Unlabelled',
                                                            args=args, 
                                                            use_fast_Kmeans=True, 
                                                            predict_token='cls')

                    print('Testing on disjoint test set...')
                    all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader,
                                                                        epoch=epoch, save_name='Fast Test ACC',
                                                                        args=args,
                                                                        use_fast_Kmeans=True, 
                                                                        predict_token='cls')
                    print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))
                    print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                                            new_acc_test))
                    
                    if args.use_val==True:
                        print('Testing on val set...')
                        lbl_score, unlbl_score, total_sil = test_kmeans_val(model, val_loader,
                                                                            epoch=epoch, save_name='Fast Val ACC',
                                                                            args=args, use_fast_Kmeans=True, 
                                                                            predict_token='cls', 
                                                                            return_silhouette=True,
                                                                            stage=2,
                                                                            )
                        old_acc_test = lbl_score
                        
                args.writer.add_scalar('Surveillance/val_score', old_acc_test, epoch)
                if args.use_fast_kmeans and (old_acc_test > best_test_acc_lab):
                    print(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
                    print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))
                    torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
                    print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
                    torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
                    print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best.pt'))
                    print("projection head saved to {}.".format(args.model_path[:-3] + f'_aux_proj_head_best.pt'))
                    torch.save(aux_projection_head.state_dict(), args.model_path[:-3] + f'_aux_proj_head_best.pt')
                    best_test_acc_lab = old_acc_test
                
        if epoch%args.checkpoint_interval==(args.checkpoint_interval-1):
            epoch_checkpoint(model, projection_head, args)
        if epoch%(args.early_stop+1)==args.early_stop:
            break
        
    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ### DEFAULT
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--devices', type=str, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=-1) ### deprecated
    
    ### META
    parser.add_argument('--exp_root', type=str, default=exp_root) ### folder to save results
    parser.add_argument('--runner_name', type=str, default='default')
    parser.add_argument('--exp_id', type=str, default=None) ### experiment id

    ### SCHEDULE
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--early_stop', type=int, default=-1)
    
    ### MODEL
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--grad_from_block', type=int, default=11) ### how many layers are frozen
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--use_vpt', type=str2bool, default=True) ### use prompt-adapted backbone; always true
    
    ### DATASET
    parser.add_argument('--dataset_name', type=str, default='scars') ### [scars, cifar100, cifar10, cub, aircraft, imagenet_100_gcd]
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True) ### always true; follow GCD
    parser.add_argument('--use_val', type=str2bool, default=False) ### if true, enable Inductive GNCD setting
    parser.add_argument('--val_split', type=float, default=0.1) ### ratio of held-out validation data
    
    ### CONTRASTIVE LOSS
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    parser.add_argument('--temperature', type=float, default=1.0)
    
    ### KMEANS
    parser.add_argument('--kmeans_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--use_fast_kmeans', type=str2bool, default='False') ### if true, use GPU side KMeans
    parser.add_argument('--fast_kmeans_batch_size', type=int, default=20000)
    
    ### INKD (ImageNet Knowledge Distillation loss)
    parser.add_argument('--inkd_T', type=int, default=5) ### total epoch during which INKD is computed
    parser.add_argument('--w_inkd_loss', type=float, default=0.01) ### initial INKD loss weight
    parser.add_argument('--w_inkd_loss_min', type=float, default=0.001) ### min decayed INKD loss weight
    parser.add_argument('--inkd_batch', type=int, default=128) ### INKD batch size
    
    ### DPR
    parser.add_argument('--num_dpr', type=int, default=2) ### number of supervised prompts
    parser.add_argument('--w_prompt_clu', type=float, default=0.35) ### DPR loss weight
    parser.add_argument('--predict_token', type=str, default='cls-vptm')
    parser.add_argument('--vpt_dropout', type=float, default=0.0)
    parser.add_argument('--num_prompts', type=int, default=5) ### number of total prompts
    parser.add_argument('--n_shallow_prompts', type=int, default=0) ### number of SHALLOW prompts prepended
    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device(args.device)
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    
    init_experiment(args, runner_name=[args.runner_name], exp_id=args.exp_id)
    seed_torch(args.seed)
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model in ['vit_dino', 'vit_ibot']:
        args.interpolation = 3
        args.crop_pct = 0.875
        if args.base_model=='vit_dino':
            args.pretrain_path = dino_pretrain_path
            args.model_arch = 'vit_base'
            model = vits.__dict__[args.model_arch]()
            args.feat_dim = 768
        elif args.base_model=='vit_ibot':
            args.pretrain_path = ibot_pretrain_path
            args.model_arch = 'vit_base'
            model = vits.__dict__[args.model_arch]()
            args.feat_dim = 768
        else:
            raise NotImplementedError()

        state_dict = torch.load(args.pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        ### VPT
        if args.use_vpt:
            vptmodel = vpt_vit.__dict__[args.model_arch](
                num_prompts=args.num_prompts,
                vpt_dropout=args.vpt_dropout,
                n_shallow_prompts=args.n_shallow_prompts,
            )
            vptmodel.load_from_state_dict(state_dict, False)
            model = vptmodel
            vpt_vit.configure_parameters(model=model, grad_layer=args.grad_from_block) ### configure parameters [freeze/unfreeze]
        else:
            ### ViT
            for m in model.parameters():
                m.requires_grad = False
            # Only finetune layers from block 'args.grad_from_block' onwards
            for name, m in model.named_parameters():
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= args.grad_from_block:
                        m.requires_grad = True
        model.to(device)
    else:
        raise NotImplementedError    

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    if args.use_val==True:
        train_dataset, test_dataset, unlabelled_train_examples_test, val_datasets, datasets = get_datasets_with_gcdval(args.dataset_name,
                                                                                                                train_transform,
                                                                                                                test_transform,
                                                                                                                args)
    else:
        train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                            train_transform,
                                                                                            test_transform,
                                                                                            args)
    
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)
    if args.use_val==True:
        val_loader = DataLoader(val_datasets, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None    

    # --------------------
    # INKD
    # --------------------
    ### create auxiliary model (DINO)
    aux_model = create_dino_backbone(args, args.pretrain_path, args.device, arch=args.model_arch)
    ### create auxiliary dataset (ImageNet)
    aux_dataloader = get_auxiliary_dataset(dataset_name='imagenet', batch_size=args.inkd_batch)
    
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projection_head.to(device)
    ### projection head for DPR
    aux_projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    aux_projection_head.to(device)
    

    ### parallelism
    if args.devices is not None:
        device_list = [int(x) for x in args.devices.split(',')]
        model = nn.DataParallel(model, device_ids=device_list)
        projection_head = nn.DataParallel(projection_head, device_ids=device_list)
        aux_model = nn.DataParallel(aux_model, device_ids=device_list)
        aux_projection_head = nn.DataParallel(aux_projection_head, device_ids=device_list)
    

    # ----------------------
    # TRAIN
    # ----------------------
    train(projection_head, model, train_loader, test_loader_labelled, test_loader_unlabelled, args, 
          aux_dataloader=aux_dataloader, aux_model=[aux_model, aux_projection_head], val_loader=val_loader)