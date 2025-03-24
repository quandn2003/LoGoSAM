"""
Training the model
Extended from original implementation of ALPNet.
"""
from scipy.ndimage import distance_transform_edt as eucl_distance
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from models.grid_proto_fewshot import FewShotSeg
from torch.utils.tensorboard import SummaryWriter
from dataloaders.dev_customized_med import med_fewshot
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple
from util.metric import Metric

import wandb
wandb.login(key="cb3c663e48e7f5e8ea89cbd09b4377b857855e95")
import matplotlib.pyplot as plt
from config_ssl_upload import ex
from tqdm.auto import tqdm
# import Tensor
from torch import Tensor
from typing import List, Tuple, Union, cast, Iterable, Set, Any, Callable, TypeVar

def initialize_wandb(config, run_name=None):
    """Initialize wandb for experiment tracking"""
    if not config.get('wandb_enabled', True):
        return None
        
    if run_name is None:
        run_name = f"{config['dataset']}_{config['model']['which_model']}"
    
    print(f"run_name: {run_name}")
    
    print(f"Project: {config.get('wandb_project', 'LoGoSAM')}")
    
    wandb_run = wandb.init(
        project=config.get('wandb_project', "LoGoSAM"),
        name=run_name,
        config=config,
        reinit=True
    )
    return wandb_run

def log_visualization_to_wandb(query_images, query_pred, query_labels, support_images, support_fg_mask, iteration, phase="train"):
    """Log visualizations to wandb"""
    # Log a sample visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Sample visualization (iteration {iteration})")
    
    # Convert tensors to numpy arrays for visualization
    query_img = t2n(query_images[0][0].permute(1, 2, 0))
    query_img = to01(query_img)
    
    # Get the predicted segmentation mask
    pred_mask = t2n(torch.sigmoid(query_pred[0, 0]))
    
    # Ground truth mask
    gt_mask = t2n(query_labels[0])
    
    # Support image and mask
    support_img = t2n(support_images[0][0][0].permute(1, 2, 0))
    support_img = to01(support_img)
    support_mask = t2n(support_fg_mask[0][0][0])
    
    # Visualize
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_mask, cmap='jet')
    axes[0, 1].set_title("Prediction Heatmap")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(query_img)
    axes[0, 2].imshow(pred_mask > 0.5, alpha=0.5, cmap='Reds')
    axes[0, 2].set_title("Prediction Overlay")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(gt_mask, cmap='gray')
    axes[1, 0].set_title("Ground Truth")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(support_img)
    axes[1, 1].set_title("Support Image")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(support_img)
    axes[1, 2].imshow(support_mask, alpha=0.5, cmap='Reds')
    axes[1, 2].set_title("Support Mask Overlay")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({f"{phase}_visualization": wandb.Image(fig)})
    plt.close(fig)

def get_dice_loss(prediction: torch.Tensor, target: torch.Tensor, smooth=1.0):
    '''
    prediction: (B, 1, H, W)
    target: (B, H, W)
    '''
    if prediction.shape[1] > 1:
        # use only the foreground prediction
        prediction = prediction[:, 1, :, :]
    prediction = torch.sigmoid(prediction)
    intersection = (prediction * target).sum(dim=(-2, -1))
    union = prediction.sum(dim=(-2, -1)) + target.sum(dim=(1, 2)) + smooth

    dice = (2.0 * intersection + smooth) / union
    dice_loss = 1.0 - dice.mean()

    return dice_loss


def get_train_transforms(_config):
    tr_transforms = myaug.transform_with_label(
        {'aug': myaug.get_aug(_config['which_aug'], _config['input_size'][0])})
    return tr_transforms

    
def get_dataset_base_name(data_name):
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
    elif data_name == 'CHAOST2_Superpix_672':
        baseset_name = 'CHAOST2'
    elif data_name == 'SABS_Superpix_448':
        baseset_name = 'SABS'
    elif data_name == 'SABS_Superpix_672':
        baseset_name = 'SABS'
    elif 'lits' in data_name.lower():
        baseset_name = 'LITS17'
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    return baseset_name

def get_nii_dataset(_config):
    data_name = _config['dataset']
    baseset_name = get_dataset_base_name(data_name)
    tr_transforms = get_train_transforms(_config)
    tr_parent = SuperpixelDataset(  # base dataset
        which_dataset=baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split=_config['eval_fold'],
        mode='train',
        # dummy entry for superpixel dataset
        min_fg=str(_config["min_fg_data"]),
        image_size=_config["input_size"][0],
        transforms=tr_transforms,
        nsup=_config['task']['n_shots'],
        scan_per_load=_config['scan_per_load'],
        exclude_list=_config["exclude_cls_list"],
        superpix_scale=_config["superpix_scale"],
        fix_length=_config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (
            data_name == 'CHAOST2_Superpix') else _config["max_iters_per_load"],
        use_clahe=_config['use_clahe'],
        use_3_slices=_config["use_3_slices"],
        tile_z_dim=3 if not _config["use_3_slices"] else 1,
    )
    
    return tr_parent


def get_dataset(_config):
    return get_nii_dataset(_config)


@ex.automain
def main(_run, _config, _log):
    precision = torch.float32
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])

    writer = SummaryWriter(f'{_run.observers[0].dir}/logs')
    _log.info('###### Create model ######')
    if _config['reload_model_path'] != '':
        _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    else:
        _config['reload_model_path'] = None
    model = FewShotSeg(image_size=_config['input_size'][0], pretrained_path=_config['reload_model_path'], cfg=_config['model'])

    model = model.to(device, precision)
    model.train()
    
    _log.info('###### Load data ######')
    data_name = _config['dataset']
    tr_parent = get_dataset(_config)

    # dataloaders
    trainloader = DataLoader(
        tr_parent,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    elif _config['optim_type'] == 'adam':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=_config['lr'], eps=1e-5)
    else:
        raise NotImplementedError

    scheduler = MultiStepLR(
        optimizer, milestones=_config['lr_milestones'],  gamma=_config['lr_step_gamma'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(
        ignore_index=_config['ignore_label'], weight=my_weight)

    i_iter = 0  # total number of iteration
    # number of times for reloading
    n_sub_epoches = max(1, _config['n_steps'] // _config['max_iters_per_load'], _config["epochs"])
    log_loss = {'loss': 0, 'align_loss': 0}

    _log.info('###### Training ######')
    epoch_losses = []
    # Initialize wandb
    wandb_run = initialize_wandb(_config, run_name=f"{_config['dataset']}_{_config['model']['which_model']}_train")
    wandb_enabled = wandb_run is not None
    print(f"n_sub_epoches: {n_sub_epoches}")
    for sub_epoch in range(n_sub_epoches):
        _log.info(
            f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        pbar = tqdm(trainloader)
        optimizer.zero_grad()
        for idx, sample_batched in enumerate(tqdm(trainloader)):
            losses = []
            i_iter += 1
            support_images = [[shot.to(device, precision) for shot in way]
                              for way in sample_batched['support_images']]
            support_fg_mask = [[shot[f'fg_mask'].float().to(device, precision) for shot in way]
                               for way in sample_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().to(device, precision) for shot in way]
                               for way in sample_batched['support_mask']]

            query_images = [query_image.to(device, precision)
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.long().to(device) for query_label in sample_batched['query_labels']], dim=0)

            loss = 0.0
            out = model(support_images, support_fg_mask, support_bg_mask, query_images, isval=False, val_wsize=None)
            query_pred, align_loss, _, coarse_pred, _, _, _ = out
                 
            query_loss = criterion(query_pred.float(), query_labels.long())
            loss += query_loss + align_loss
            pbar.set_postfix({'loss': loss.item()})
            loss.backward()
            if (idx + 1) % _config['grad_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            losses.append(loss.item())
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            _run.log_scalar('loss', query_loss)
            _run.log_scalar('align_loss', align_loss)

            log_loss['loss'] += query_loss
            log_loss['align_loss'] += align_loss

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:
                writer.add_scalar('loss', loss, i_iter)
                writer.add_scalar('query_loss', query_loss, i_iter)
                writer.add_scalar('align_loss', align_loss, i_iter)

                # Log metrics to wandb
                if wandb_enabled:
                    wandb.log({
                        'loss': loss.item(),
                        'query_loss': query_loss,
                        'align_loss': align_loss,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'iteration': i_iter
                    })

                    # Log visualizations to wandb every print_interval
                    log_visualization_to_wandb(
                        query_images,
                        query_pred,
                        query_labels,
                        support_images,
                        support_fg_mask,
                        i_iter,
                        phase="train"
                    )
                    
                    # Log coarse segmentation if available
                    if coarse_pred is not None:
                        # Create figure for coarse segmentation
                        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                        coarse_mask = t2n(torch.sigmoid(coarse_pred[0, 0]))
                        query_img = t2n(query_images[0][0].permute(1, 2, 0))
                        query_img = to01(query_img)
                        
                        axes[0].imshow(coarse_mask, cmap='jet')
                        axes[0].set_title("Coarse Segmentation Heatmap")
                        axes[0].axis('off')
                        
                        axes[1].imshow(query_img)
                        axes[1].imshow(coarse_mask > 0.5, alpha=0.5, cmap='Blues')
                        axes[1].set_title("Coarse Segmentation Overlay")
                        axes[1].axis('off')
                        
                        plt.tight_layout()
                        wandb.log({'coarse_segmentation': wandb.Image(fig)})
                        plt.close(fig)

                loss = log_loss['loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0

                print(
                    f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss},')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            if (i_iter - 1) >= _config['n_steps']:
                break  # finish up
        epoch_losses.append(np.mean(losses))
        print(f"Epoch {sub_epoch} loss: {np.mean(losses)}")
        if wandb_enabled:
            wandb.log({'epoch': sub_epoch, 'epoch_loss': np.mean(losses)})
    
    # Close wandb run when training is finished
    if wandb_enabled:
        wandb.finish()
