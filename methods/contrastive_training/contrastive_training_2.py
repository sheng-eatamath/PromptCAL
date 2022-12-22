import argparse
import os
import time
from tqdm import tqdm


import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from config import exp_root, dino_pretrain_path
from project_utils.cluster_utils import mixed_eval, AverageMeter
from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights, seed_torch
from project_utils.cluster_and_log_utils import log_accs_from_preds
from data.augmentations import get_transform
# from data.get_datasets import get_datasets, get_class_splits
from data.get_datasets import get_datasets, get_class_splits, get_datasets_with_gcdval


### >>>
from models import vision_transformer as vits
from models import vpt_vision_transformer as vpt_vit
from ..clustering.faster_mix_k_means_pytorch import K_Means
from models.model_create import create_dino_backbone, create_model, create_projection_head
from .common import *
from .utils_pknn import *
from project_utils.cluster_utils import my_mixed_eval
from guohao.ema import EMA
from guohao.memory_bank import MemoryBank
from guohao.mymeter import MyMeter
from guohao.mylogging import MyLogger
### <<<


def train(projection_head, model, train_loader, test_loader, unlabelled_train_loader, args, 
          teacher=None, aux_projection_head=None, verbose=False, 
          val_loader=None):
    """
    Args:
        projection_head (nn.Module): dino head
        model (nn.Module): backbone
        train_loader (DataLoader)
        test_loader (DataLoader)
        val_loader (DataLoader)
        unlabelled_train_loader (DataLoader)
        args (Config)
        teacher (nn.Module): same initialization from student at the beginning
        aux_projection_head (nn.Module): projection head for DPR loss
    """

    assert args.predict_token in ['cop']
    
    
    debug = MyLogger('ERROR')
    debug.set_verbose(verbose)
    
    model_params = list(projection_head.parameters()) + list(model.parameters())

    ### >>> MoCo initialize
    sup_con_crit = SupConLossWithMembank()
    membank_label = MemoryBank(max_size=args.membank_size, embedding_size=1, name='unsup_label') ### for class label
    membank_lmask = MemoryBank(max_size=args.membank_size, embedding_size=1, name='unsup_lmask') ### for labeled-or-not
    membank_unsup_z = MemoryBank(max_size=args.membank_size, embedding_size=args.feat_dim, name='unsup_z') ### for CLS embedding
    membank_unsup_dpr_z = MemoryBank(max_size=args.membank_size, embedding_size=args.feat_dim, name='unsup_dpr_z') ### for DPR embedding
    ### <<<
        
    best_test_acc_lab = 0
    best_test_score = 0
    i_iter = 0

    assert aux_projection_head is not None, f'aux_projection_head is None={aux_projection_head is None}'
    aux_projection_head.train()
    model_params += list(aux_projection_head.parameters())
    model_t, projection_head_t, aux_projection_head_t = teacher
    model_t.eval()
    projection_head_t.eval()
    aux_projection_head_t.eval()

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
        aux_projection_head.train()
                    
        ### >>> MoCo initialize
        if epoch==0:
            ema = EMA(momentum=args.momentum_ema, verbose=verbose)
            tlist = [model_t, projection_head_t]
            slist = [model, projection_head]
            tlist.append(aux_projection_head_t)
            slist.append(aux_projection_head)
            ema.initialize_teacher_from_student(teacher=tlist, student=slist)
        ### <<<
        
        ### training
        with tqdm(total=len(train_loader)) as pbar:
            for batch_idx, batch in enumerate(train_loader):
                
                images, class_labels, uq_idxs, mask_lab = batch
                mask_lab = mask_lab[:, 0] ### B
                class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
                images = torch.cat(images, dim=0).to(device) ### [nview*B, 3, H, W]
                
                # Model forward
                ### >>>
                ### forward student
                features = forward(images, model, projection_head, 
                                predict_token=args.predict_token, 
                                num_prompts=args.num_prompts,
                                num_cop=args.num_dpr,
                                aux_projection_head=aux_projection_head,
                                mode='train',
                                return_z_features=True,
                                )
                ### forward teacher
                with torch.no_grad():
                    features_me = forward(images, model_t, projection_head_t, 
                                    predict_token=args.predict_token, 
                                    num_prompts=args.num_prompts,
                                    num_cop=args.num_dpr,
                                    aux_projection_head_t=aux_projection_head_t,
                                    mode='teacher',
                                    return_z_features=True,
                                    )
                ### <<<
                
                loss = 0.0
                
                
                ### >>> unpack features
                """
                    [features, z_features, pp_features, z_pp_features]
                    [features_me, z_features_me, pp_features_me, z_pp_features_me]
                """
                [features, z_features, pp_features, z_pp_features] = features
                [features_me, z_features_me, pp_features_me, z_pp_features_me] = features_me
                ### <<<
                
                
                ### >>> semi-sup contrastive loss
                ### for CLS loss
                sup_con_loss = 0
                unsup_contrastive_loss = 0
                loss_sup = compute_cross_contrastive_loss_sup(features, class_labels, mask_lab,
                                                                    features_me, class_labels, mask_lab,
                                                                    sup_cont_crit=sup_con_crit)
                loss_unsup, contrastive_logits, contrastive_labels = compute_cross_contrastive_loss_unsup(
                    features, features_me, temperature=args.temperature,
                    )
                sup_con_loss += loss_sup
                unsup_contrastive_loss += loss_unsup
                        
                mymeter.add('unsup_contrastive_loss', unsup_contrastive_loss.item())
                mymeter.add('sup_con_loss', sup_con_loss.item())
                
                ### for DPR loss
                aux_sup_con_loss = 0
                aux_unsup_contrastive_loss = 0
                loss_sup_aux = compute_cross_contrastive_loss_sup(pp_features, class_labels, mask_lab,
                                                                pp_features_me, class_labels, mask_lab,
                                                                sup_cont_crit=sup_con_crit)
                loss_unsup_aux, aux_contrastive_logits, aux_contrastive_labels = compute_cross_contrastive_loss_unsup(
                    pp_features, pp_features_me, temperature=args.temperature,
                    )
                aux_sup_con_loss += loss_sup_aux
                aux_unsup_contrastive_loss += loss_unsup_aux
                ### <<<
                
                
                
                ### >>> pKNN loss
                knn_num = knn_precision = knn_recall = pp_knn_precision = pp_knn_recall = pp_knn_num = torch.tensor(0.0)
                if len(membank_label): ### non-empty membank
                    ### pKNN from CLS
                    pknn_features_me = torch.cat([f for f in z_features_me.chunk(2)][::-1], dim=0)
                    features_mb, _ = membank_unsup_z.query()
                    features_mb = features_mb.to(args.device)
                    q_features = z_features
                    k_features_mb = features_mb
                    k_features_me = z_features_me
                    labels_mb, _ = membank_label.query()
                    labels_mb = labels_mb.to(device)
                    lmask_mb, _ = membank_lmask.query()
                    lmask_mb = lmask_mb.to(device)
                    pos_affinity, mutex_affinity = compute_pos_affinity_with_lmask(class_labels, mask_lab, labels_mb, lmask_mb)
                    similarity, knn_affinity, transition_prob = compute_pseudo_knn(pknn_features_me, features_mb, 
                                                                    method=args.knn_method, k=args.knn,
                                                                    diffusion_steps=args.diffusion_steps,
                                                                    q=args.diffusion_q,
                                                                    mask_lab=mask_lab.repeat(2),
                                                                    )

                    neg_affinity = negative_sampling_from_membank(knn_affinity, similarity=similarity, neg_samples=args.neg_sample)
                    knn_precision, knn_recall, knn_num = compute_knn_statistics_with_affinity(
                        class_labels.repeat(2), labels_mb.int(), knn_affinity, eps=1e-10
                    )
                    knn_affinity, neg_affinity = update_knn_affinity(pos_affinity, mutex_affinity, knn_affinity, neg_affinity)
                    
                    ### pKNN from ppfeatures
                    pknn_pp_features_me = torch.cat([f for f in z_pp_features_me.chunk(2)][::-1], dim=0)
                    pp_features_mb, _ = membank_unsup_dpr_z.query()
                    pp_features_mb = pp_features_mb.to(device)
                    q_pp_features = z_pp_features
                    k_pp_features_mb = pp_features_mb
                    k_pp_features_me = pknn_pp_features_me
                    
                    pp_similarity, pp_knn_affinity, pp_transition_prob = compute_pseudo_knn(pknn_pp_features_me, pp_features_mb, 
                                                                method=args.knn_method, k=args.knn,
                                                                diffusion_steps=args.diffusion_steps,
                                                                q=args.diffusion_q,
                                                                mask_lab=mask_lab.repeat(2),
                                                                )
                    

                    pp_neg_affinity = negative_sampling_from_membank(pp_knn_affinity, similarity=pp_similarity, neg_samples=args.neg_sample)
                    pp_knn_precision, pp_knn_recall, pp_knn_num = compute_knn_statistics_with_affinity(
                        class_labels.repeat(2), labels_mb.int(), pp_knn_affinity, eps=1e-10
                    )
                    pp_knn_affinity, pp_neg_affinity = update_knn_affinity(pos_affinity, mutex_affinity, pp_knn_affinity, pp_neg_affinity)
                            
                    
                    ### pKNN loss on CLS
                    knn_contrastive_loss = compute_knn_loss(knn_affinity, neg_affinity, q_features, k_features_mb, k_features_me, 
                                        temperature=args.knn_temperature, k=args.knn, epoch=epoch, 
                                        )
                    
                    ### pKNN loss on pp
                    pp_knn_contrastive_loss = compute_knn_loss(pp_knn_affinity, pp_neg_affinity, q_pp_features, 
                                                                k_pp_features_mb, k_pp_features_me, 
                                                                temperature=args.knn_temperature, k=args.knn, 
                                                                epoch=epoch, 
                                                                )
                else:
                    knn_contrastive_loss = pp_knn_contrastive_loss = torch.tensor(0.0, device=device)
                
                mymeter.add('knn_contrastive_loss', knn_contrastive_loss.item())
                mymeter.add('pp_knn_contrastive_loss', pp_knn_contrastive_loss.item())
                
                
                ### compute loss weight for pknn
                w_knn_loss = annealing_linear_ramup(0, args.w_knn_loss, i_iter, args.w_knn_loss_rampup_T*len(train_loader))
                
                ### compute aux total loss
                aux_loss = args.sup_con_weight*aux_sup_con_loss + (1-args.sup_con_weight)*((1-w_knn_loss)*aux_unsup_contrastive_loss \
                    + w_knn_loss*(pp_knn_contrastive_loss))
                loss += args.w_prompt_clu * aux_loss
                mymeter.add('aux_loss', aux_loss.item())
                
                ### compute cls total loss
                loss += args.sup_con_weight*sup_con_loss + (1-args.sup_con_weight)*((1-w_knn_loss)*unsup_contrastive_loss \
                        + w_knn_loss*(knn_contrastive_loss))

                # Record loss
                loss_record.update(loss.item(), class_labels.size(0))

                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                
                # optimize
                optimizer.step()
                
                
                # MoCo update & Memory Bank
                tlist = [model_t, projection_head_t, aux_projection_head_t]
                slist = [model, projection_head, aux_projection_head]
                ema.after_train_iter(teacher=tlist, student=slist)

                membank_label.add(v=class_labels.repeat(2).view(-1, 1).float(), y=None)
                membank_lmask.add(v=mask_lab.repeat(2).view(-1, 1).float(), y=None)
                membank_unsup_z.add(v=z_features_me, y=None)
                membank_unsup_dpr_z.add(v=z_pp_features_me, y=None)
                
                pbar.update(1)
                pbar.set_postfix(
                    loss=loss_record.avg,
                    unsup_contrastive_loss=mymeter.mean('unsup_contrastive_loss'),
                    sup_con_loss=mymeter.mean('sup_con_loss'),
                    knn_contrastive_loss=mymeter.mean('knn_contrastive_loss'),
                    pp_knn_contrastive_loss=mymeter.mean('pp_knn_contrastive_loss'),
                    aux_loss=mymeter.mean('aux_loss'),
                    epoch=epoch,
                )
                
                i_iter += 1

        exp_lr_scheduler.step()
        epoch_checkpoint(model, projection_head, args, epoch=None)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss/unsup_contrastive_loss', mymeter.mean('unsup_contrastive_loss'), epoch)
        args.writer.add_scalar('Loss/sup_con_loss', mymeter.mean('sup_con_loss'), epoch)
        args.writer.add_scalar('Loss/knn_contrastive_loss', mymeter.mean('knn_contrastive_loss'), epoch)
        args.writer.add_scalar('Loss/total', loss_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)
        
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        

            
        # ----------------
        # clustering
        # ----------------
        other_modules = {
            'model_t': model_t,
        }
        if epoch%args.eval_interval==(args.eval_interval-1):
            if epoch%args.kmeans_interval==(args.kmeans_interval-1):
                with torch.no_grad():
                    print('Testing on unlabelled examples in the training data...')
                    kmeans_res = test_kmeans(model, unlabelled_train_loader,
                                                            epoch=epoch, save_name='Train ACC Unlabelled',
                                                            args=args, predict_token=args.predict_token,
                                                            )
                    all_acc, old_acc, new_acc = kmeans_res
                        
                    if epoch%args.eval_interval_t==(args.eval_interval_t-1):
                        _ = test_kmeans(model_t, unlabelled_train_loader,
                                                                epoch=epoch, save_name='Train ACC Unlabelled Teacher',
                                                                args=args, predict_token=args.predict_token,
                                                                )
                    print('Testing on disjoint test set...')
                    all_acc_test, old_acc_test, new_acc_test, score = test_kmeans(model, test_loader,
                                                                        epoch=epoch, save_name='Test ACC',
                                                                        args=args, predict_token=args.predict_token,
                                                                        return_silhouette=True,
                                                                        )
                    print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))
                    print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                                            new_acc_test))
                    score = (score+old_acc_test)/2
                    
                    if args.use_val==True:
                        print('Testing on val set...')
                        lbl_score, unlbl_score, total_sil = test_kmeans_val(model, val_loader,
                                                                            epoch=epoch, save_name='Val ACC',
                                                                            args=args, use_fast_Kmeans=False, 
                                                                            predict_token=args.predict_token, 
                                                                            return_silhouette=True,
                                                                            stage=2,
                                                                            )
                        score = (lbl_score+total_sil)/2
                        
                args.writer.add_scalar(f'Surveillance/val_score', score, epoch)
                print(f'score={score}')
                    
                if score > best_test_score:
                    print(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
                    print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))
                    epoch_checkpoint(model, projection_head, args, epoch=None, best=True, other_modules=other_modules, postfix='score')
                    best_test_score = score
            else:
                with torch.no_grad():
                    print('Testing on unlabelled examples in the training data...')
                    kmeans_res = test_kmeans(model, unlabelled_train_loader,
                                            epoch=epoch, save_name='Fast Train ACC Unlabelled',
                                            args=args, 
                                            use_fast_Kmeans=True, 
                                            predict_token=args.predict_token,
                                            )
                    all_acc, old_acc, new_acc = kmeans_res
                        

                    print('Testing on disjoint test set...')
                    all_acc_test, old_acc_test, new_acc_test, score = test_kmeans(model, test_loader,
                                                                        epoch=epoch, save_name='Fast Test ACC',
                                                                        args=args,
                                                                        use_fast_Kmeans=True, 
                                                                        predict_token=args.predict_token,
                                                                        return_silhouette=True,
                                                                        )
                    print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))
                    print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                                            new_acc_test))
                    score = (score+old_acc_test)/2
                        
                    if args.use_val==True:
                        print('Testing on val set...')
                        lbl_score, unlbl_score, total_sil = test_kmeans_val(model, val_loader,
                                                                            epoch=epoch, save_name='Fast Val ACC',
                                                                            args=args, use_fast_Kmeans=True, 
                                                                            predict_token=args.predict_token, 
                                                                            return_silhouette=True,
                                                                            stage=2,
                                                                            )
                        score = (lbl_score+total_sil)/2
                        

                    args.writer.add_scalar(f'Surveillance/val_score', score, epoch)
                    
                    if epoch%args.eval_interval_t==(args.eval_interval_t-1):
                        _ = test_kmeans(model_t, unlabelled_train_loader,
                                            epoch=epoch, save_name='Train ACC Unlabelled Teacher',
                                            args=args, use_fast_Kmeans=True, 
                                            predict_token=args.predict_token,
                                            )
                        
                print(f'score={score}')
                    
                if args.use_fast_kmeans and (score > best_test_score):
                    print(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
                    print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))
                    epoch_checkpoint(model, projection_head, args, epoch=None, best=True, other_modules=other_modules, postfix='score')
                    best_test_score = score
                    
        if epoch%args.checkpoint_interval==(args.checkpoint_interval-1):
            epoch_checkpoint(epoch, model, projection_head, args)
    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ### DEFAULT
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--num_workers_test', default=8, type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--devices', type=str, default=None)
    
    ### SCHEDULE
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    ### CONTRASTIVE LOSS
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    
    ### LOGGING
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--runner_name', type=str, default='default')
    parser.add_argument('--exp_id', type=str, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=-1)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--eval_interval_t', type=int, default=1) ### evaluation interval for teacher model
    
    ### KMEANS
    parser.add_argument('--kmeans_interval', type=int, default=1)
    parser.add_argument('--use_fast_kmeans', type=str2bool, default='False') ### if true, use GPU side KMeans (recommended)
    parser.add_argument('--fast_kmeans_batch_size', type=int, default=20000)
    
    ### DATASET
    parser.add_argument('--use_val', type=str2bool, default=False) ### if true, enable Inductive GNCD setting
    parser.add_argument('--val_split', type=float, default=0.1) ### ratio of held-out validation data
    parser.add_argument('--dataset_name', type=str, default='scars') ### [scars, cifar100, cifar10, cub, aircraft, imagenet_100_gcd]
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    
    ### MODEL
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--load_from_head', type=str, default=None)
    parser.add_argument('--load_from_model', type=str, default=None)
    parser.add_argument('--warmup_model_dir', type=str, default=None)

    ### VPT
    parser.add_argument('--use_vpt', type=str2bool, default=True)
    parser.add_argument('--vpt_dropout', type=float, default=0.0)
    parser.add_argument('--num_prompts', type=int, default=5) ### number of total prompts
    parser.add_argument('--predict_token', type=str, default='cop')
    parser.add_argument('--n_shallow_prompts', type=int, default=0) ### number of SHALLOW prompts prepended
    
    ### DPR
    parser.add_argument('--num_dpr', type=int, default=2) ### number of supervised prompts
    parser.add_argument('--w_prompt_clu', type=float, default=0.35) ### DPR loss weight
    
    ### MOCO
    parser.add_argument('--membank_size', type=int, default=4096)
    parser.add_argument('--neg_sample', type=int, default=1024) ### num neg. samples
    parser.add_argument('--momentum_ema', type=float, default=0.999) ### MoCo momentum
    
    ### PKNN (Pseudo-KNN)
    parser.add_argument('--knn', type=int, default=20) ### KNN range
    parser.add_argument('--knn_method', type=str, default='neighbor_consensus_diffusion')
    parser.add_argument('--knn_temperature', type=float, default=0.07) ### temperature for pKNN loss
    parser.add_argument('--w_knn_loss', type=float, default=0.6)
    parser.add_argument('--w_knn_loss_rampup_T', type=int, default=2) ### ramup T
    
    ### DIFUSSION
    parser.add_argument('--diffusion_steps', type=int, default=1)
    parser.add_argument('--diffusion_q', type=float, default=0.5) ### threshould quantile
    
    
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
    if args.base_model == 'vit_dino':
        args.interpolation = 3
        args.crop_pct = 0.875
        args.dino_pretrain_path = dino_pretrain_path
        
        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536
        
        model, projection_head = create_model(args, device)
        teacher = create_model(args, device)
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
                              sampler=sampler, drop_last=True,
                              pin_memory=True,
                              )
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers_test,
                                        batch_size=args.batch_size, shuffle=False,
                                        pin_memory=True,
                                        )
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers_test,
                                      batch_size=args.batch_size, shuffle=False,
                                      pin_memory=True,
                                      )
    if args.use_val==True:
        val_loader = DataLoader(val_datasets, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None
    
    
    # --------------------
    # TEACHER
    # --------------------
    aux_projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                            out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    aux_projection_head.to(device)
    aux_projection_head_t = create_projection_head(args, device, use_checkpoint=False)
    teacher += [aux_projection_head_t.to(device)]
    aux_projection_head.load_state_dict(aux_projection_head_t.state_dict(), strict=True)
    
    if args.devices is not None:
        device_list = [int(x) for x in args.devices.split(',')]
        model = nn.DataParallel(model, device_ids=device_list)
        projection_head = nn.DataParallel(projection_head, device_ids=device_list)
        aux_projection_head = nn.DataParallel(aux_projection_head, device_ids=device_list)
        for i in range(len(teacher)):
            teacher[i] = nn.DataParallel(teacher[i], device_ids=device_list)

        
    # ----------------------
    # TRAIN
    # ----------------------
    train(projection_head, model, train_loader, test_loader_labelled, test_loader_unlabelled, args, 
          teacher=teacher, aux_projection_head=aux_projection_head, 
          val_loader=val_loader)