"""
Validation script
"""
import math
import os
import pandas as pd
import csv
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import time
import matplotlib.pyplot as plt
import wandb
wandb.login(key="cb3c663e48e7f5e8ea89cbd09b4377b857855e95")
from models.ProtoSAM import ProtoSAM,  ALPNetWrapper, SamWrapperWrapper, InputFactory, ModelWrapper, TYPE_ALPNET, TYPE_SAM
from models.ProtoMedSAM import ProtoMedSAM
from models.grid_proto_fewshot import FewShotSeg
from models.segment_anything.utils.transforms import ResizeLongestSide
from models.SamWrapper import SamWrapper
# from dataloaders.PolypDataset import get_polyp_dataset, get_vps_easy_unseen_dataset, get_vps_hard_unseen_dataset, PolypDataset, KVASIR, CVC300, COLON_DB, ETIS_DB, CLINIC_DB
from dataloaders.PolypDataset import get_polyp_dataset, PolypDataset
from dataloaders.PolypTransforms import get_polyp_transform
from dataloaders.SimpleDataset import SimpleDataset
from dataloaders.ManualAnnoDatasetv2 import get_nii_dataset
from dataloaders.common import ValidationDataset
from config_ssl_upload import ex

import tqdm
from tqdm.auto import tqdm
import cv2
from collections import defaultdict

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

# Supported Datasets
CHAOS = "chaos"
SABS = "sabs"
POLYPS = "polyps"

ALP_DS = [CHAOS, SABS]

ROT_DEG = 0

def get_bounding_box(segmentation_map):
    """Generate bounding box from a segmentation map. one bounding box to include the extreme points of the segmentation map."""
    if isinstance(segmentation_map, torch.Tensor):
        segmentation_map = segmentation_map.cpu().numpy()
    
    bbox = cv2.boundingRect(segmentation_map.astype(np.uint8))
    # plot bounding boxes for each contours
    # plt.figure()
    # x, y, w, h = bbox
    # plt.imshow(segmentation_map)
    # plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=2))
    # plt.savefig("debug/bounding_boxes.png") 

    return bbox

def calc_iou(boxA, boxB):
    """
    boxA: [x, y, w, h]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def eval_detection(pred_list):
    """
    pred_list: list of dictionaries with keys 'pred_bbox', 'gt_bbox' and score (prediction confidence score).
    compute AP50, AP75, AP50:95:10
    """
    iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)
    ap_dict = {iou: [] for iou in iou_thresholds}
    for iou_threshold in iou_thresholds:
        tp, fp = 0, 0
        
        for pred in pred_list:
            pred_bbox = pred['pred_bbox']
            gt_bbox = pred['gt_bbox']
            
            iou = calc_iou(pred_bbox, gt_bbox)
            
            if iou >= iou_threshold:
                tp += 1
            else:
                fp += 1

        precision = tp / (tp + fp)
        recall = tp / len(pred_list) 
        f1 = 2 * (precision * recall) / (precision + recall)        

        ap_dict[iou_threshold] = {
            'iou_threshold': iou_threshold,
            'tp': tp,
            'fp': fp,
            'n_gt': len(pred_list),
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Convert results to a DataFrame and save to CSV
    results = []
    for iou_threshold in iou_thresholds:
        results.append(ap_dict[iou_threshold])
    
    df = pd.DataFrame(results)
    return df


def plot_pred_gt_support(query_image, pred, gt, support_images, support_masks, score=None, save_path="debug/pred_vs_gt.png"):
    """
    pred: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    gt: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    support: 4d tensor of shape (N, C, H, W) where 1 represents foreground and 0 represents background
    """
    if support_images:
        if isinstance(support_images, list):
            support_images = torch.cat(support_images, dim=0).clone().detach()
        if isinstance(support_masks, list):
            support_masks = torch.cat(support_masks, dim=0).clone().detach()
        if len(query_image.shape) == 3:
            query_image = query_image.permute(1, 2, 0).clone().detach()
        if len(support_images.shape) == 4:
            support_images = support_images.clone().detach().permute(0, 2, 3, 1)
        n_support_rows = math.ceil(support_images.shape[0] / 2)
    else:
        n_support_rows = 1
    fig, ax = plt.subplots(n_support_rows + 1, 2)
    query_image = (query_image - query_image.min()) / (query_image.max() - query_image.min())
    ax[0, 0].imshow(query_image.cpu().detach())
    ax[0, 0].imshow(pred, alpha=0.5)
    ax[0, 0].set_title("pred")
    ax[0, 1].imshow(query_image.cpu().detach())
    ax[0, 1].imshow(gt, alpha=0.5)
    ax[0, 1].set_title("gt")
    if support_images is not None:
        for i in range(1, n_support_rows + 1):
            support_images[(i - 1) * 2] = (support_images[(i - 1) * 2] - support_images[(i - 1) * 2].min()) / (support_images[(i - 1) * 2].max() - support_images[(i - 1) * 2].min())
            ax[i, 0].imshow(support_images[(i - 1) * 2].cpu().detach())
            ax[i, 0].imshow(support_masks[(i - 1) * 2].cpu(), alpha=0.5)
            ax[i, 0].set_title(f"support")
            if (i - 1) * 2 + 1 < support_images.shape[0]:
                support_images[(i - 1) * 2 + 1] = (support_images[(i - 1) * 2 + 1] - support_images[(i - 1) * 2 + 1].min()) / (support_images[(i - 1) * 2 + 1].max() - support_images[(i - 1) * 2 + 1].min())
                ax[i, 1].imshow(support_images[(i - 1) * 2 + 1].cpu().detach())
                ax[i, 1].imshow(support_masks[(i - 1) * 2 + 1].cpu(), alpha=0.5)
                ax[i, 1].set_title(f"support")
    if score is not None:
        # plt.title(f"score: {score}") 
        fig.suptitle(f"sam score: {score}")
    fig.savefig(save_path)
    plt.close(fig)


def get_dice_iou_precision_recall(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    gt: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    """
    if gt.sum() == 0:
        print("gt is all background")
        return {"dice": 0, "precision": 0, "recall": 0}

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def get_alpnet_model(_config) -> ModelWrapper:
    alpnet = FewShotSeg(
       _config["input_size"][0],
       _config["reload_model_path"],
       _config["model"]
    )
    alpnet.cuda()
    alpnet_wrapper = ALPNetWrapper(alpnet)
    
    return alpnet_wrapper

def get_sam_model(_config) -> ModelWrapper:
    sam_args = {
        "model_type": "vit_h",
        "sam_checkpoint": "pretrained_model/sam_vit_h.pth"
    }
    sam = SamWrapper(sam_args=sam_args).cuda()
    sam_wrapper = SamWrapperWrapper(sam)
    return sam_wrapper  

def get_model(_config) -> ProtoSAM:
    # Initial Segmentation Model
    if _config["base_model"] == TYPE_ALPNET:
        base_model = get_alpnet_model(_config)
    else:
        raise NotImplementedError(f"base model {_config['base_model']} not implemented")
    
    # ProtoSAM model
    if _config["protosam_sam_ver"] in  ("sam_h", "sam_b"):
        sam_h_checkpoint = "pretrained_model/sam_vit_h.pth"
        sam_b_checkpoint = "pretrained_model/sam_vit_b.pth"
        sam_checkpoint = sam_h_checkpoint if _config["protosam_sam_ver"] == "sam_h" else sam_b_checkpoint
        model = ProtoSAM(image_size = (1024, 1024),
                    coarse_segmentation_model=base_model,
                    use_bbox=_config["use_bbox"],
                    use_points=_config["use_points"],
                    use_mask=_config["use_mask"],
                    debug=_config["debug"],
                    num_points_for_sam=1,
                    use_cca=_config["do_cca"],
                    point_mode=_config["point_mode"],
                    use_sam_trans=True, 
                    coarse_pred_only=_config["coarse_pred_only"],
                    sam_pretrained_path=sam_checkpoint,
                    use_neg_points=_config["use_neg_points"],) 
    elif _config["protosam_sam_ver"] == "medsam":
        model = ProtoMedSAM(image_size = (1024, 1024),
                            coarse_segmentation_model=base_model,
                            debug=_config["debug"],
                            use_cca=_config["do_cca"],
        )
    else:
        raise NotImplementedError(f"protosam_sam_ver {_config['protosam_sam_ver']} not implemented")
    
    return model


def get_support_set_polyps(_config, dataset:PolypDataset):
    n_support = _config["n_support"]
    (support_images, support_labels, case) = dataset.get_support(n_support=n_support)
    
    return support_images, support_labels, case


def get_support_set_alpds(config, dataset:ValidationDataset):
    support_set = dataset.get_support_set(config)
    support_fg_masks = support_set["support_labels"]
    support_images = support_set["support_images"]
    support_scan_id = support_set["support_scan_id"]
    return support_images, support_fg_masks, support_scan_id


def get_support_set(_config, dataset):
    if _config["dataset"].lower() == POLYPS:
        support_images, support_fg_masks, case = get_support_set_polyps(_config, dataset)
    elif any(item in _config["dataset"].lower() for item in ALP_DS):
        support_images, support_fg_masks, support_scan_id = get_support_set_alpds(_config, dataset)
    else:
        raise NotImplementedError(f"dataset {_config['dataset']} not implemented")
    return support_images, support_fg_masks, support_scan_id


def update_support_set_by_scan_part(support_images, support_labels, qpart):
    qpart_support_images = [support_images[qpart]]
    qpart_support_labels = [support_labels[qpart]]
    
    return qpart_support_images, qpart_support_labels


def manage_support_sets(sample_batched, all_support_images, all_support_fg_mask, support_images, support_fg_mask, qpart=None):
    if sample_batched['part_assign'][0] != qpart:
        qpart = sample_batched['part_assign'][0]
        support_images, support_fg_mask = update_support_set_by_scan_part(all_support_images, all_support_fg_mask, qpart)
            
    return support_images, support_fg_mask, qpart


def initialize_wandb(config, run_name=None):
    """Initialize wandb for experiment tracking"""
    if not config.get('wandb_enabled', True):
        return None
        
    if run_name is None:
        run_name = f"{config['dataset']}_{config['base_model']}_inference"
        
    wandb_run = wandb.init(
        project=config.get('wandb_project', "LoGoSAM"),
        name=run_name,
        config=config,
        reinit=True
    )
    return wandb_run

def log_inference_results_to_wandb(query_image, pred, gt, support_images, support_masks, raw_sam_output=None, attn=None, iteration=0):
    """Log inference visualization to wandb"""
    # Create figure for results visualization
    if attn is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convert tensors to numpy arrays for visualization
    query_img = query_image.cpu().numpy().transpose(1, 2, 0)
    if query_img.shape[2] == 1:  # If grayscale, repeat channels
        query_img = np.repeat(query_img, 3, axis=2)
    # Normalize to 0-1 range
    query_img = (query_img - query_img.min()) / (query_img.max() - query_img.min() + 1e-8)
    
    # Masks
    pred_mask = pred.cpu().numpy()
    gt_mask = gt.cpu().numpy()
    
    # Support image (first one)
    support_img = support_images[0][0].cpu().numpy().transpose(1, 2, 0)
    if support_img.shape[2] == 1:
        support_img = np.repeat(support_img, 3, axis=2)
    support_img = (support_img - support_img.min()) / (support_img.max() - support_img.min() + 1e-8)
    
    support_mask = support_masks[0][0].cpu().numpy()
    
    # Display images
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_mask, cmap='jet')
    axes[0, 1].set_title("Prediction")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(gt_mask, cmap='gray')
    axes[1, 0].set_title("Ground Truth")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(support_img)
    axes[1, 1].imshow(support_mask, alpha=0.5, cmap='Reds')
    axes[1, 1].set_title("Support Image & Mask")
    axes[1, 1].axis('off')
    
    # Add attention visualization if available
    if attn is not None:
        # Create heatmap from attention 
        attention_map = attn.cpu().numpy()
        axes[0, 2].imshow(attention_map, cmap='hot')
        axes[0, 2].set_title("Attention Heatmap")
        axes[0, 2].axis('off')
        
        # Overlay attention on query image
        axes[1, 2].imshow(query_img)
        axes[1, 2].imshow(attention_map, alpha=0.7, cmap='hot')
        axes[1, 2].set_title("Attention Overlay")
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({f"inference_result_{iteration}": wandb.Image(fig)})
    plt.close(fig)
    
    # Log raw SAM output if available
    if raw_sam_output is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        raw_output = raw_sam_output.cpu().numpy()
        
        ax[0].imshow(raw_output, cmap='viridis')
        ax[0].set_title("Raw SAM Output")
        ax[0].axis('off')
        
        ax[1].imshow(query_img)
        ax[1].imshow(raw_output > 0.5, alpha=0.5, cmap='Greens')
        ax[1].set_title("Raw SAM Output Overlay")
        ax[1].axis('off')
        
        plt.tight_layout()
        wandb.log({f"raw_sam_output_{iteration}": wandb.Image(fig)})
        plt.close(fig)


@ex.automain
def main(_run, _config, _log):
    seed = _config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # Initialize wandb for inference visualization
    wandb_run = initialize_wandb(_config, run_name=f"{_config['dataset']}_{_config['base_model']}_inference")
    wandb_enabled = wandb_run is not None

    _log.info('###### Set CUDA ######')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _log.info('###### Setup DATASET ######')
    data_name = _config['dataset']
    if wandb_enabled:
        wandb.config.update({"data_name": data_name})
    
    if data_name == POLYPS:
        # When using polyps dataset
        polyp_ds = get_polyp_dataset(_config, eval=True)
        _log.info('Loading polyp dataset')
        
        val_set = polyp_ds
        support_data = get_support_set_polyps(_config, polyp_ds)
    else:
        # MedicalDataset
        val_set = get_nii_dataset(_config)
        support_data = get_support_set_alpds(_config, val_set)
        _log.info('Loading ALP dataset')
    
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    all_support_images, all_support_fg_mask = support_data
    
    _log.info('###### Setup Model ######')
    if _config["base_model"] == "alpnet":
        model_wrapper = get_alpnet_model(_config)
    elif _config["base_model"] == "sam":
        model_wrapper = get_sam_model(_config)
    else:
        raise ValueError(f"Unsupported segmentor model type: {_config['base_model']}")
        
    model = get_model(_config)
    model.build(model_wrapper)
    model = model.to(device)
    model.eval()

    # Setup for calculating metrics
    metrics = defaultdict(list)
    
    # Add timing setup
    qids = []
    sample_metrics = []
    
    _log.info('###### Starting Validation ######')
    
    with torch.no_grad():
        for idx, sample_batched in enumerate(tqdm(val_loader)):
            qid = sample_batched["id"][0] if "id" in sample_batched else idx
            
            # Skip this sample if qid is in qids
            if qid in qids:
                continue
            
            support_images, support_fg_mask, qpart = manage_support_sets(
                sample_batched, all_support_images, all_support_fg_mask, None, None, None
            )
            
            # Images shape: 1, 3/1, H, W
            query_image = sample_batched["image"].to(device)
            query_mask = sample_batched["label"].to(device)
            
            # Record the time for measuring inference speed
            start_time = time.time()
            result = model.slide_inference(
                query_image,
                support_images,
                support_fg_mask,
                query_mask.shape[-2:]
            )
            end_time = time.time()
            
            # Extract results
            pred_mask, attentions = result.pred_mask, result.attentions
            raw_sam_output = result.raw_sam_output if hasattr(result, 'raw_sam_output') else None
            coarse_pred = result.coarse_pred if hasattr(result, 'coarse_pred') else None
            
            # Calculate metrics for current sample
            metric_dict = {}
            metric_dict.update(get_dice_iou_precision_recall(pred_mask, query_mask))
            
            # Log to wandb
            for metric_name, metric_value in metric_dict.items():
                metrics[metric_name].append(metric_value)
                if wandb_enabled:
                    wandb.log({f"{metric_name}_sample_{idx}": metric_value})
            
            if wandb_enabled and (idx % 10 == 0 or idx < 5):
                # Log visualizations to wandb every 10 samples and first 5 samples
                log_inference_results_to_wandb(
                    query_image[0],
                    pred_mask[0],
                    query_mask[0],
                    support_images,
                    support_fg_mask,
                    raw_sam_output[0] if raw_sam_output is not None else None,
                    attentions[0] if attentions is not None else None,
                    idx
                )
                
                # If coarse prediction is available, log it too
                if coarse_pred is not None:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    coarse_mask = coarse_pred[0].cpu().numpy()
                    query_img = query_image[0].cpu().numpy().transpose(1, 2, 0)
                    if query_img.shape[2] == 1:
                        query_img = np.repeat(query_img, 3, axis=2)
                    query_img = (query_img - query_img.min()) / (query_img.max() - query_img.min() + 1e-8)
                    
                    axes[0].imshow(coarse_mask, cmap='jet')
                    axes[0].set_title("Coarse Segmentation")
                    axes[0].axis('off')
                    
                    axes[1].imshow(query_img)
                    axes[1].imshow(coarse_mask > 0.5, alpha=0.5, cmap='Blues')
                    axes[1].set_title("Coarse Segmentation Overlay")
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    wandb.log({f"coarse_segmentation_{idx}": wandb.Image(fig)})
                    plt.close(fig)
            
            # Save the sample metrics
            sample_metric = {
                "id": qid,
                "time": end_time - start_time,
                **metric_dict
            }
            sample_metrics.append(sample_metric)
            qids.append(qid)
    
    # Calculate and log average metrics
    avg_metrics = {name: np.mean(values) for name, values in metrics.items()}
    for name, value in avg_metrics.items():
        _log.info(f'Average {name}: {value:.4f}')
        if wandb_enabled:
            wandb.log({f"avg_{name}": value})
    
    # Save metrics as CSV
    save_path = os.path.join(_config['path']['log_dir'], 'metrics.csv')
    with open(save_path, 'w', newline='') as csvfile:
        fieldnames = ['id', 'time'] + list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample_metric in sample_metrics:
            writer.writerow(sample_metric)
    
    # Create summary table for wandb
    if wandb_enabled:
        metrics_table = wandb.Table(dataframe=pd.DataFrame(sample_metrics))
        wandb.log({"metrics_results": metrics_table})
        
        # Finish wandb run
        wandb.finish()
    
    return avg_metrics
