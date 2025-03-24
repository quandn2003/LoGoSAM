"""
Validation script
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import wandb
wandb.login(key="cb3c663e48e7f5e8ea89cbd09b4377b857855e95")  # Using the same key from validation_protosam

from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot_val

from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from dataloaders.niftiio import convert_to_sitk
import dataloaders.augutils as myaug

from util.metric import Metric
from util.consts import IMG_SIZE
from util.utils import cca, sliding_window_confidence_segmentation, plot_3d_bar_probabilities, save_pred_gt_fig, plot_heatmap_of_probs
from config_ssl_upload import ex

from tqdm import tqdm
import SimpleITK as sitk
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from util.utils import set_seed, t2n, to01, compose_wt_simple
# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"


def initialize_wandb(config, run_name=None):
    """Initialize wandb for experiment tracking"""
    if not config.get('wandb_enabled', True):
        return None
        
    if run_name is None:
        run_name = f"{config['dataset']}_validation"
        
    wandb_run = wandb.init(
        project=config.get('wandb_project', "LoGoSAM"),
        name=run_name,
        config=config,
        reinit=True
    )
    return wandb_run


def log_validation_results_to_wandb(query_images, pred, gt, support_images, support_fg_mask, assign_mats=None, proto_grid=None, iteration=0, class_label=None):
    """Log validation visualization to wandb"""
    # Create figure for results visualization
    if assign_mats is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convert tensors to numpy arrays for visualization
    # Handle query image which is a list containing tensor
    query_img = query_images[0][0].cpu().numpy()  # [C, H, W]
    # If image has multiple channels, use the middle channel for visualization
    if len(query_img.shape) == 3 and query_img.shape[0] > 1:
        query_img = query_img[query_img.shape[0]//2]  # Take middle channel for visualization
    # Normalize to 0-1 range
    query_img = (query_img - query_img.min()) / (query_img.max() - query_img.min() + 1e-8)
    
    # Masks
    pred_mask = pred
    gt_mask = gt[0].cpu().numpy()
    
    # Support image (first one in the first way)
    support_img = support_images[0][0][0].cpu().numpy()  # [C, H, W]
    if len(support_img.shape) == 3 and support_img.shape[0] > 1:
        support_img = support_img[support_img.shape[0]//2]  # Take middle channel
    support_img = (support_img - support_img.min()) / (support_img.max() - support_img.min() + 1e-8)
    
    support_mask = support_fg_mask[0][0].cpu().numpy()
    if len(support_mask.shape) == 3:  # If it's [1, H, W], squeeze it
        support_mask = support_mask.squeeze(0)
    
    # Display images
    axes[0, 0].imshow(query_img, cmap='gray')
    axes[0, 0].set_title("Query Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(query_img, cmap='gray')
    axes[0, 1].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[0, 1].set_title(f"Prediction (Class {class_label})")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(gt_mask, cmap='gray')
    axes[1, 0].set_title("Ground Truth")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(support_img, cmap='gray')
    axes[1, 1].imshow(support_mask, alpha=0.5, cmap='Reds')
    axes[1, 1].set_title("Support Image & Mask")
    axes[1, 1].axis('off')
    
    # Add attention/assignment visualization if available
    if assign_mats is not None:
        # Create heatmap from assignment matrix
        assign_heatmap = assign_mats[0].cpu().numpy()
        
        # Average the assignment matrix over channels if needed
        if len(assign_heatmap.shape) > 2:
            assign_heatmap = np.mean(assign_heatmap, axis=0)
            
        axes[0, 2].imshow(assign_heatmap, cmap='hot')
        axes[0, 2].set_title("Assignment Heatmap")
        axes[0, 2].axis('off')
        
        # Overlay assignment on query image
        axes[1, 2].imshow(query_img, cmap='gray')
        axes[1, 2].imshow(assign_heatmap, alpha=0.7, cmap='hot')
        axes[1, 2].set_title("Assignment Overlay")
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Log to wandb
    image_name = f"validation_result_class{class_label}_iter{iteration}" if class_label is not None else f"validation_result_{iteration}"
    wandb.log({image_name: wandb.Image(fig)})
    plt.close(fig)
    
    # If proto_grid is available, visualize it
    if proto_grid is not None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        proto_grid_np = proto_grid.cpu().numpy()
        
        # Handle different proto_grid shapes
        if len(proto_grid_np.shape) == 4:  # Shape: (1, 1, H, W)
            proto_grid_np = proto_grid_np.squeeze()  # Remove batch and channel dimensions
        elif len(proto_grid_np.shape) == 3:  # Shape: (1, H, W)
            proto_grid_np = proto_grid_np.squeeze(0)  # Remove batch dimension
            
        ax.imshow(proto_grid_np, cmap='viridis')
        ax.set_title("Prototype Grid")
        ax.axis('off')
        plt.tight_layout()
        proto_name = f"proto_grid_class{class_label}_iter{iteration}" if class_label is not None else f"proto_grid_{iteration}"
        wandb.log({proto_name: wandb.Image(fig)})
        plt.close(fig)


def test_time_training(_config, model, image, prediction):
    model.train()
    data_name = _config['dataset']
    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(
        ignore_index=_config['ignore_label'], weight=my_weight)
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    elif _config['optim_type'] == 'adam':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=_config['lr'], eps=1e-5)
    else:
        raise NotImplementedError
    optimizer.zero_grad()
    scheduler = MultiStepLR(
        optimizer, milestones=_config['lr_milestones'],  gamma=_config['lr_step_gamma'])
    
    tr_transforms = myaug.transform_with_label(
        {'aug': myaug.get_aug(_config['which_aug'], _config['input_size'][0])})
   
    comp = np.concatenate([image.transpose(1, 2, 0), prediction[None,...].transpose(1,2,0)], axis= -1)
    print("Test Time Training...")
    pbar = tqdm(range(_config['n_steps']))
    
    # Store losses for wandb
    losses = []
    
    for idx in pbar:
        query_image, query_label = tr_transforms(comp, c_img=image.shape[0], c_label=1, nclass=2, use_onehot=False)
        support_image, support_label = tr_transforms(comp, c_img=image.shape[0], c_label=1, nclass=2, use_onehot=False)
        query_label = torch.from_numpy(query_label.transpose(2,1,0)).cuda().long()

        query_images = [torch.from_numpy(query_image.transpose(2, 1, 0)).unsqueeze(0).cuda().float().requires_grad_(True)]
        support_fg_mask = [[torch.from_numpy(support_label.transpose(2, 1, 0)).cuda().float().requires_grad_(True)]]
        support_bg_mask = [[torch.from_numpy(1 - support_label.transpose(2, 1, 0)).cuda().float().requires_grad_(True)]]
        support_images = [[torch.from_numpy(support_image.transpose(2, 1, 0)).unsqueeze(0).cuda().float().requires_grad_(True)]]
    
        out = model(support_images, support_fg_mask, support_bg_mask, query_images, isval=False, val_wsize=None)
        query_pred, align_loss, _, _, _, _, _ = out
        
        loss = 0.0
        loss += criterion(query_pred.float(), query_label.long())
        loss += align_loss
        loss.backward()
        
        losses.append(loss.item())
        
        if (idx + 1) % _config['grad_accumulation_steps'] == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Log training progress to wandb
        if idx % 5 == 0:  # Log every 5 steps to avoid too many points
            wandb.log({
                "ttt_loss": loss.item(),
                "ttt_step": idx
            })
            
            # Periodically visualize the training progress
            if idx % 10 == 0:
                pred = query_pred.argmax(dim=1)[0].detach().cpu().numpy()
                
                # Create visualization
                fig, ax = plt.subplots(2, 2, figsize=(10, 10))
                
                # Query image
                query_img = query_images[0][0].detach().cpu().permute(1, 2, 0).numpy()
                query_img = (query_img - query_img.min()) / (query_img.max() - query_img.min() + 1e-8)
                if query_img.shape[2] == 1:
                    query_img = np.repeat(query_img, 3, axis=2)
                
                # Support image
                support_img = support_images[0][0][0].detach().cpu().permute(1, 2, 0).numpy()
                support_img = (support_img - support_img.min()) / (support_img.max() - support_img.min() + 1e-8)
                if support_img.shape[2] == 1:
                    support_img = np.repeat(support_img, 3, axis=2)
                
                # Display
                ax[0, 0].imshow(query_img)
                ax[0, 0].set_title("Query Image")
                ax[0, 0].axis('off')
                
                ax[0, 1].imshow(query_img)
                ax[0, 1].imshow(pred, alpha=0.5, cmap='jet')
                ax[0, 1].set_title(f"Current Prediction (Step {idx})")
                ax[0, 1].axis('off')
                
                ax[1, 0].imshow(support_img)
                ax[1, 0].imshow(support_fg_mask[0][0].detach().cpu().numpy(), alpha=0.5, cmap='Reds')
                ax[1, 0].set_title("Support Image")
                ax[1, 0].axis('off')
                
                ax[1, 1].imshow(query_label[0].detach().cpu().numpy(), cmap='gray')
                ax[1, 1].set_title("Target Mask")
                ax[1, 1].axis('off')
                
                plt.tight_layout()
                wandb.log({f"ttt_progress_step{idx}": wandb.Image(fig)})
                plt.close(fig)
                
    # Plot loss curve
    if len(losses) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Test-Time Training Loss')
        wandb.log({"ttt_loss_curve": wandb.Image(fig)})
        plt.close(fig)
        
    model.eval()
    return model


def compute_dice_score(pred, gt, labels):
    """
    Compute dice score between prediction and ground truth
    Args:
        pred: prediction array
        gt: ground truth array
        labels: list of labels to compute dice for
    Returns:
        dice score
    """
    dice_scores = []
    for label in labels:
        pred_mask = (pred == label)
        gt_mask = (gt == label)
        
        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)
        
        if union == 0:
            dice_scores.append(1.0 if intersection == 0 else 0.0)
        else:
            dice_scores.append(2.0 * intersection / union)
    
    return np.mean(dice_scores)

def compute_iou_score(pred, gt, labels):
    """
    Compute IoU score between prediction and ground truth
    Args:
        pred: prediction array
        gt: ground truth array
        labels: list of labels to compute IoU for
    Returns:
        IoU score
    """
    iou_scores = []
    for label in labels:
        pred_mask = (pred == label)
        gt_mask = (gt == label)
        
        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask)
        
        if union == 0:
            iou_scores.append(1.0 if intersection == 0 else 0.0)
        else:
            iou_scores.append(intersection / union)
    
    return np.mean(iou_scores)


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    # Initialize wandb
    wandb_run = initialize_wandb(_config)
    wandb_enabled = wandb_run is not None

    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    model = FewShotSeg(image_size=_config['input_size'][0],
                       pretrained_path=_config['reload_model_path'], cfg=_config['model'])

    model = model.cuda()
    model.eval()

    _log.info('###### Load data ######')
    # Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix' or data_name == 'SABS_Superpix_448' or data_name == 'SABS_Superpix_672':
        baseset_name = 'SABS'
        max_label = 13
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
        max_label = 3
    elif data_name == 'CHAOST2_Superpix' or data_name == 'CHAOST2_Superpix_672':
        baseset_name = 'CHAOST2'
        max_label = 4
    elif 'lits' in data_name.lower():
        baseset_name = 'LITS17'
        max_label = 4
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - \
        DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]


    _log.info(
        f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(
        f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    if baseset_name == 'SABS':
        tr_parent = SuperpixelDataset(  # base dataset
            which_dataset=baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split=_config['eval_fold'],
            mode='val',  # 'train',
            # dummy entry for superpixel dataset
            min_fg=str(_config["min_fg_data"]),
            image_size=_config['input_size'][0],
            transforms=None,
            nsup=_config['task']['n_shots'],
            scan_per_load=_config['scan_per_load'],
            exclude_list=_config["exclude_cls_list"],
            superpix_scale=_config["superpix_scale"],
            fix_length=_config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (
                data_name == 'CHAOST2_Superpix') else None,
            use_clahe=_config['use_clahe'],
            norm_mean=0.18792 * 256 if baseset_name == 'LITS17' else None,
            norm_std=0.25886 * 256 if baseset_name == 'LITS17' else None
        )
        norm_func = tr_parent.norm_func
    else:
        norm_func = get_normalize_op(modality='MR', fids=None)

    te_dataset, te_parent = med_fewshot_val(
        dataset_name=baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split=_config['eval_fold'],
        scan_per_load=_config['scan_per_load'],
        act_labels=test_labels,
        npart=_config['task']['npart'],
        nsup=_config['task']['n_shots'],
        extern_normalize_func=norm_func,
        image_size=_config["input_size"][0],
        use_clahe=_config['use_clahe'],
        use_3_slices=_config["use_3_slices"]
    )

    # dataloaders
    testloader = DataLoader(
        te_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )

    _log.info('###### Set validation nodes ######')
    mar_val_metric_node = Metric(max_label=max_label, n_scans=len(
        te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'])

    _log.info('###### Starting validation ######')
    mar_val_metric_node.reset()
    if _config["sliding_window_confidence_segmentation"]:
        print("Using sliding window confidence segmentation")  # TODO delete this

    save_pred_buffer = {}  # indexed by class

    for curr_lb in test_labels:
        te_dataset.set_curr_cls(curr_lb)
        support_batched = te_parent.get_support(curr_class=curr_lb, class_idx=[
                                                curr_lb], scan_idx=_config["support_idx"], npart=_config['task']['npart'])

        # way(1 for now) x part x shot x 3 x H x W] #
        support_images = [[shot.cuda() for shot in way]
                          for way in support_batched['support_images']]  # way x part x [shot x C x H x W]
        suffix = 'mask'
        support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                          for way in support_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                          for way in support_batched['support_mask']]

        curr_scan_count = -1  # counting for current scan
        _lb_buffer = {}  # indexed by scan
        _lb_vis_buffer = {}

        last_qpart = 0  # used as indicator for adding result to buffer

        for idx, sample_batched in enumerate(tqdm(testloader)):
            # we assume batch size for query is 1
            _scan_id = sample_batched["scan_id"][0]
            if _scan_id in te_parent.potential_support_sid:  # skip the support scan, don't include that to query
                continue
            if sample_batched["is_start"]:
                ii = 0
                curr_scan_count += 1
                print(
                    f"Processing scan {curr_scan_count + 1} / {len(te_dataset.dataset.pid_curr_load)}")
                _scan_id = sample_batched["scan_id"][0]
                outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                # original image read by itk: Z, H, W, in prediction we use H, W, Z
                outsize = (_config['input_size'][0],
                          _config['input_size'][1], outsize[0])
                _pred = np.zeros(outsize)
                _pred.fill(np.nan)
                # assign proto shows in the query image which proto is assigned to each pixel, proto_grid is the ids of the prototypes in the support image used, support_images are the 3 support images, support_img_parts are the parts of the support images used for each query image
                _vis = {'assigned_proto': [None] * _pred.shape[-1], 'proto_grid': [None] * _pred.shape[-1],
                        'support_images': support_images, 'support_img_parts': [None] * _pred.shape[-1]}

            # the chunck of query, for assignment with support
            q_part = sample_batched["part_assign"]
            query_images = [sample_batched['image'].cuda()]
            query_labels = torch.cat(
                [sample_batched['label'].cuda()], dim=0)
            if 1 not in query_labels and not sample_batched["is_end"] and _config["skip_no_organ_slices"]:
                ii += 1
                continue
            # [way, [part, [shot x C x H x W]]] ->
            # way(1) x shot x [B(1) x C x H x W]
            sup_img_part = [[shot_tensor.unsqueeze(
                0) for shot_tensor in support_images[0][q_part]]]
            sup_fgm_part = [[shot_tensor.unsqueeze(
                0) for shot_tensor in support_fg_mask[0][q_part]]]
            sup_bgm_part = [[shot_tensor.unsqueeze(
                0) for shot_tensor in support_bg_mask[0][q_part]]]

            # query_pred_logits, _, _, assign_mats, proto_grid, _, _ = model(
            #     sup_img_part, sup_fgm_part, sup_bgm_part, query_images, isval=True, val_wsize=_config["val_wsize"], show_viz=True)
            with torch.no_grad():
                out = model(sup_img_part, sup_fgm_part, sup_bgm_part,
                        query_images, isval=True, val_wsize=_config["val_wsize"])
            query_pred_logits, _, _, assign_mats, proto_grid, _, _ = out
            pred = np.array(query_pred_logits.argmax(dim=1)[0].cpu())
                
            if _config["ttt"]: 
                state_dict = model.state_dict()
                model = test_time_training(_config, model, sample_batched['image'].numpy()[0], pred)
                out = model(sup_img_part, sup_fgm_part, sup_bgm_part,
                        query_images, isval=True, val_wsize=_config["val_wsize"])
                query_pred_logits, _, _, assign_mats, proto_grid, _, _ = out
                pred = np.array(query_pred_logits.argmax(dim=1)[0].cpu())
                if _config["reset_after_slice"]:
                    model.load_state_dict(state_dict)
                    
            query_pred = query_pred_logits.argmax(dim=1).cpu()
            query_pred = F.interpolate(query_pred.unsqueeze(
                0).float(), size=query_labels.shape[-2:], mode='nearest').squeeze(0).long().numpy()[0]

            # Log visualization to wandb
            if wandb_enabled and (idx % 20 == 0 or ii < 5):  # Log every 20 samples and first 5 slices
                log_validation_results_to_wandb(
                    query_images, 
                    query_pred, 
                    query_labels, 
                    sup_img_part, 
                    sup_fgm_part, 
                    assign_mats, 
                    proto_grid, 
                    iteration=idx, 
                    class_label=curr_lb
                )
                
                # Log metrics for this sample
                gt_np = np.array(query_labels[0].cpu())
                dice_score = compute_dice_score(query_pred, gt_np, labels=[curr_lb])
                iou_score = compute_iou_score(query_pred, gt_np, labels=[curr_lb])
                
                # Log metrics together in a single dictionary
                wandb.log({
                    f"metrics/class{curr_lb}": {
                        "Dice": dice_score,
                        "IoU": iou_score,
                        "step": idx,
                        "slice": ii
                    }
                })

            if _config["debug"]:
                os.makedirs("debug/preds", exist_ok=True)
                save_pred_gt_fig(
                    query_images, 
                    query_pred, 
                    query_labels, 
                    sup_img_part[0], 
                    sup_fgm_part[0][0],
                    f'debug/preds/scan_{_scan_id}_label_{curr_lb}_{idx}_gt_vs_pred.png'
                )
                
            if _config['do_cca']:
                query_pred = cca(query_pred, query_pred_logits)
                if _config["debug"]:
                    save_pred_gt_fig(query_images, query_pred, query_labels,
                                    f'debug/scan_{_scan_id}_label_{curr_lb}_{idx}_gt_vs_pred_after_cca.png')

            _pred[..., ii] = query_pred.copy()
            _vis['assigned_proto'][ii] = assign_mats
            _vis['proto_grid'][ii] = proto_grid.cpu()
            _vis['support_img_parts'][ii] = q_part

            if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']) and not sample_batched["is_end"]:
                mar_val_metric_node.record(query_pred, np.array(
                    query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count)
            else:
                pass

            ii += 1
            # now check data format
            if sample_batched["is_end"]:
                if _config['dataset'] != 'C0':
                    _lb_buffer[_scan_id] = _pred.transpose(
                        2, 0, 1)  # H, W, Z -> to Z H W
                else:
                    _lb_buffer[_scan_id] = _pred
                _lb_vis_buffer[_scan_id] = _vis

        save_pred_buffer[str(curr_lb)] = _lb_buffer

        # save results
        for curr_lb, _preds in save_pred_buffer.items():
            for _scan_id, _pred in _preds.items():
                _pred *= float(curr_lb)
                itk_pred = convert_to_sitk(
                    _pred, te_dataset.dataset.info_by_scan[_scan_id])
                fid = os.path.join(
                    f'{_run.observers[0].dir}/interm_preds', f'scan_{_scan_id}_label_{curr_lb}.nii.gz')
                sitk.WriteImage(itk_pred, fid, True)
                _log.info(f'###### {fid} has been saved ######')


    # compute dice scores by scan
    m_classDice, _, m_meanDice, _, m_rawDice = mar_val_metric_node.get_mDice(
        labels=sorted(test_labels), n_scan=None, give_raw=True)

    m_classPrec, _, m_meanPrec, _,  m_classRec, _, m_meanRec, _, m_rawPrec, m_rawRec = mar_val_metric_node.get_mPrecRecall(
        labels=sorted(test_labels), n_scan=None, give_raw=True)

    # Log to wandb
    if wandb_enabled:
        # Create line plot for final metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot metrics for each class
        x = np.arange(len(sorted(test_labels)))
        width = 0.35
        
        ax.bar(x - width/2, m_classDice, width, label='Dice')
        ax.bar(x + width/2, m_rawDice, width, label='IoU')
        
        ax.set_ylabel('Score')
        ax.set_title('Final Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {label}' for label in sorted(test_labels)])
        ax.legend()
        
        plt.tight_layout()
        wandb.log({"final_metrics_plot": wandb.Image(fig)})
        plt.close(fig)
        
        # Log summary metrics
        wandb.log({
            "final_metrics": {
                "mean_dice": m_meanDice,
                "mean_iou": np.mean(m_rawDice)
            }
        })
        
        # Create summary table
        metrics_table = wandb.Table(columns=["Class", "Dice", "IoU"])
        for i, class_label in enumerate(sorted(test_labels)):
            metrics_table.add_data(class_label, m_classDice[i], m_rawDice[i])
        wandb.log({"metrics_summary": metrics_table})
        
        # Finish wandb run
        wandb.finish()

    mar_val_metric_node.reset()  # reset this calculation node

    # write validation result to log file
    _run.log_scalar('mar_val_batches_classDice', m_classDice.tolist())
    _run.log_scalar('mar_val_batches_meanDice', m_meanDice.tolist())
    _run.log_scalar('mar_val_batches_rawDice', m_rawDice.tolist())

    _run.log_scalar('mar_val_batches_classPrec', m_classPrec.tolist())
    _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec.tolist())
    _run.log_scalar('mar_val_batches_rawPrec', m_rawPrec.tolist())

    _run.log_scalar('mar_val_batches_classRec', m_classRec.tolist())
    _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec.tolist())
    _run.log_scalar('mar_val_al_batches_rawRec', m_rawRec.tolist())

    _log.info(f'mar_val batches classDice: {m_classDice}')
    _log.info(f'mar_val batches meanDice: {m_meanDice}')

    _log.info(f'mar_val batches classPrec: {m_classPrec}')
    _log.info(f'mar_val batches meanPrec: {m_meanPrec}')

    _log.info(f'mar_val batches classRec: {m_classRec}')
    _log.info(f'mar_val batches meanRec: {m_meanRec}')

    print("============ ============")

    _log.info(f'End of validation')
    return 1
