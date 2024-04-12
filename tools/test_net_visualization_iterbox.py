#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import copy
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    AmodalVisibleEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.data import transforms as T
from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes, pairwise_iou
from detectron2.modeling import GeneralizedRCNNWithTTA
from tools.add_config import add_config
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize
from torch.utils.data import Dataset, DataLoader, random_split
import json
import cv2
import numpy as np
import pycocotools.mask as mask_utils
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Optional, Tuple
import random
# import SAM model
from segment_anything import SamPredictor, sam_model_registry
from tools.transforms import ResizeLongestSide
from pycocotools.mask import encode
def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        if cfg.MODEL.AISFormer.AMODAL_EVAL == True:
            evaluator_list.append(AmodalVisibleEvaluator(dataset_name, cfg, False, output_dir=output_folder))
        else:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def rle_to_mask(rle):
    mask = mask_utils.decode(rle)
    return mask
def polys_to_mask(polygons, height, width):
	rles = mask_utils.frPyObjects(polygons, height, width)
	rle = mask_utils.merge(rles)
	mask = mask_utils.decode(rle)
	return mask

class AmodalDataset(Dataset):
    def __init__(self, dataset='kins'):
        self.dataset = dataset
        self.cur_imgid = None
        self.cur_imgemb = None
        self.ais_aug = T.AugmentationList([T.ResizeShortestEdge(short_edge_length=(800, 800), max_size=3000, sample_style='choice')])
        self.data_root = imgemb_root[dataset]
        self.img_root = img_root[dataset]
        self.anno_path = anno_path
        with open(self.anno_path) as f:
            anns = json.load(f)
            self.imgs_info = anns['images']
            self.anns_info = anns['annotations']

    def __getitem__(self, index):
        img_id = self.imgs_info[index]['id']
        img_name = self.imgs_info[index]['file_name']
        path = os.path.join(self.data_root, img_name.split(img_suffix[self.dataset])[0]+'.pt')
        img_path = os.path.join(self.img_root, img_name)
        
        # for aisformer input data dictionary
        ais_data = {'file_name': img_path}
        img = utils.read_image(ais_data['file_name'], format='BGR')
        ais_data['height'] = img.shape[0]
        ais_data['width'] = img.shape[1]
        ais_data['image_id'] = img_id
        aug_input = T.AugInput(img)
        transforms = self.ais_aug(aug_input)
        img = aug_input.image
        ais_data["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

        # load pre-computed img embeddings
        self.cur_imgemb = torch.load(path)
        img_emb = self.cur_imgemb['feature'].squeeze(0).to(device)
        input_size = self.cur_imgemb['input_size']
        original_size = self.cur_imgemb['original_size']
        h, w = original_size

        # instances area
        area = torch.tensor([anno['area'] for anno in self.anns_info[str(img_id)]])
        
        # amodal mask GT
        asegm = [anno["segmentation"] for anno in self.anns_info[str(img_id)]]
        if self.dataset == 'kins':
            asegm = np.stack([polys_to_mask(mask, h, w) for mask in asegm])
        if self.dataset == 'cocoa':
            asegm = np.stack([rle_to_mask(mask) for mask in asegm])
        asegm = torch.as_tensor(asegm, dtype=torch.float, device=device).unsqueeze(1)  
        
        # random point prompt
        point_torch = []
        
        # amodal bbox GT
        box_torch = []
        abbox = np.array([anno["bbox"] for anno in self.anns_info[str(img_id)]])
        box_torch = np.hstack((abbox[:, :2], abbox[:, :2] + abbox[:, 2:]))
        box_torch = torch.as_tensor(box_torch, dtype=torch.float, device=device)

        return img_emb, asegm, box_torch, point_torch, original_size, input_size, ais_data, img_path, area

    def __len__(self):
        return len(self.imgs_info)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    from detectron2.config import CfgNode as CN
    cfg.DICE_LOSS = CN()
    cfg.DICE_LOSS = False
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

#oracle = True
#pred_iou = False
matcher_iou = 0.5
#filter_threshold = 0.5
vit_type = 'vit_h'
dataset_name = 'cocoa'
#ais_weight = '/work/weientai18/aisformer/aisformer_R_50_FPN_1x_amodal_kins_160000_resume/model_final.pth'
#ais_weight = '/work/weientai18/aisformer/aisformer_R_50_FPN_1x_amodal_kins_160000_resume/model_0119999_best.pth'
#ais_config = '/work/weientai18/aisformer/aisformer_R_50_FPN_1x_amodal_kins_160000_resume/config.yaml'
ais_config = '/work/weientai18/aisformer/full_training_cocoa_pre/config.yaml'
ais_weight = '/work/weientai18/aisformer/full_training_cocoa_pre/model_0007999.pth'
#result_save_path = '/work/weientai18/result_h_oracle_0.8.json'
sam_ckpt = '/work/weientai18/amodal_dataset/checkpoint/model_20240328_034832_59_GTamodal_coco'
vis_save_root = '/work/weientai18/aissam_iterbox_cocoa'


img_suffix = {
    'kins':'.png',
    'cocoa':'.jpg'
}
vit_dict = {
    'vit_b':"/home/weientai18/SAM/SAM_ckpt/sam_vit_b_01ec64.pth", 
    'vit_h':"/home/weientai18/SAM/SAM_ckpt/sam_vit_h_4b8939.pth"
}
anno_dict = {
    'kins':"/work/weientai18/amodal_dataset/KITTI_AMODAL_DATASET/mod_instances_val_2.json",
    'cocoa':"/work/weientai18/amodal_dataset/COCO_AMODAL_DATASET/mod_COCOA_cls_val2014.json"
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_root = {
    'kins':'/home/weientai18/ais/data/datasets/KINS/test_imgs',
    'cocoa':'/home/weientai18/ais/data/datasets/COCOA/test_imgs'
}
imgemb_root = {
    'kins':'/work/weientai18/amodal_dataset/testing_imgemb_h',
    'cocoa':'/work/weientai18/amodal_dataset/coco_testing_imgemb_h'
}
anno_path = anno_dict[dataset_name]
anchor_matcher = Matcher(
        thresholds=[matcher_iou], labels=[0, 1], allow_low_quality_matches=False
    )
result_list = []

def generate_random_colors(num_colors):
    R = random.sample(range(50, 200), num_colors)
    G = random.sample(range(50, 200), num_colors)
    B = random.sample(range(50, 200), num_colors)
    colors = list(zip(R, G, B))
    
    return colors

def vis(img_path, ais_box_all, sam_mask_all, box_2_all, sam_mask_2_all, gt_mask_all, fp_idxs_all, small_idx, lm_idx):
    
    sml_dic = {
        'small':small_idx,
        'lm':lm_idx
    }
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_name = img_path.split('/')[-1]
    for size in ['small', 'lm']:
        g = sml_dic[size]
        #fp_idxs = list(set(fp_idxs_all.cpu().numpy()) & set(g.cpu().numpy()))
        fp_idxs = (g.unsqueeze(1) == fp_idxs_all).nonzero(as_tuple=False)[:, 0]
        #fp_idxs = (fp_idxs_all == g).nonzero(as_tuple=False)[:, 0]
        ais_box = ais_box_all[g]
        p2_box = box_2_all[g]
        sam_mask = sam_mask_all[g]
        sam_2_mask = sam_mask_2_all[g]
        gt_mask = gt_mask_all[g]

        save_path = os.path.join(vis_save_root, size, img_name)
        sam_2_img = copy.deepcopy(img)
        sam_img = copy.deepcopy(img)
        gt_img = copy.deepcopy(img)
        sam_2_img = cv2.addWeighted(sam_2_img, 0.2, sam_2_img, 0, 0)
        sam_img = cv2.addWeighted(sam_img, 0.2, sam_img, 0, 0)
        gt_img = cv2.addWeighted(gt_img, 0.2, gt_img, 0, 0)
        color_fp = (100, 0, 255)
        color_tp = (0, 255, 100)
        num_colors = len(g)
        random_colors = generate_random_colors(num_colors)
        for i in range(num_colors):
            b, g, r = random_colors[i]
            smask = sam_mask[i].squeeze(0).cpu().numpy().astype(np.uint8)
            s2mask = sam_2_mask[i].squeeze(0).cpu().numpy().astype(np.uint8)
            gmask = gt_mask[i].squeeze(0).cpu().numpy().astype(np.uint8)
            smask = np.stack((b*smask, g*smask, r*smask), axis=2)
            s2mask = np.stack((b*s2mask, g*s2mask, r*s2mask), axis=2)
            gmask = np.stack((b*gmask, g*gmask, r*gmask), axis=2)
            sam_img = cv2.addWeighted(sam_img, 1, smask, 0.7, 0)
            sam_2_img = cv2.addWeighted(sam_2_img, 1, s2mask, 0.7, 0)
            gt_img = cv2.addWeighted(gt_img, 1, gmask, 0.7, 0)
        for i in range(num_colors):
            ais = ais_box[i]
            p2 = p2_box[i]
            color = color_fp if i in fp_idxs else color_tp
            for image in [sam_img, gt_img]:
                image = cv2.rectangle(image, (int(ais[0]), int(ais[1])), (int(ais[2]), int(ais[3])), color, 1)
            sam_2_img = cv2.rectangle(sam_2_img, (int(p2[0]), int(p2[1])), (int(p2[2]), int(p2[3])), color, 1)
        final_img = np.concatenate((gt_img, sam_img), axis=0)
        final_img = np.concatenate((final_img, sam_2_img), axis=0)
        cv2.imwrite(save_path, final_img)


def find_bounding_boxes(masks):
    
    bounding_boxes = []
    for i in range(masks.size(0)):
        mask = masks[i, 0]  # Get the i-th mask with shape (H, W)
        
        # Find the non-zero indices
        non_zero_indices = torch.nonzero(mask, as_tuple=False)
        
        if non_zero_indices.size(0) == 0:
            # No non-zero indices, append an empty box or handle appropriately
            bounding_boxes.append([0, 0, 0, 0])
            continue
        
        y_min, x_min = torch.min(non_zero_indices, dim=0).values
        y_max, x_max = torch.max(non_zero_indices, dim=0).values
        
        # Append bounding box (converting to x_min, y_min, x_max, y_max format)
        bounding_boxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
    
    return torch.as_tensor(bounding_boxes, dtype=torch.float, device=device)


def main(args):
    torch.manual_seed(0)
    # setting up AISFormer model
    cfg = setup(args)
    global aisformer 
    aisformer = Trainer.build_model(cfg)
    DetectionCheckpointer(aisformer, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    aisformer.to(device)
    aisformer.eval()

    # setting up SAM model
    global sam_model 
    sam_model = sam_model_registry[vit_type](checkpoint=vit_dict[vit_type])
    sam_model.mask_decoder.load_state_dict(torch.load(sam_ckpt))
    mask_threshold = sam_model.mask_threshold
    sam_model.to(device)
    sam_model.eval()

    global transform
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    # Create datasets for training & validation
    dataset = AmodalDataset(dataset_name)

    # test dataloader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    samples = random.sample(range(len(dataset)), 10)
    empty_img = 0
    for i, data in enumerate(data_loader):
        if i not in samples:
            continue
        data = [None if x == [] else x for x in data]
        image_embedding, asegm, bbox, point, original_size, input_size, ais_data, img_path, area = data
        print(img_path[0].split('/')[-1])
        with torch.no_grad():
            ais_data['image'] = torch.squeeze(ais_data['image'], 0).to(device)
            ais_data['height'] = ais_data['height'].item()
            ais_data['width'] = ais_data['width'].item()
            ais_data['image_id'] = ais_data['image_id'].item()
            img_id = ais_data['image_id']
            original_size = [j.item() for j in original_size]
            input_size = [j.item() for j in input_size]
            asegm = torch.squeeze(asegm, 0)
            bbox = torch.squeeze(bbox, 0)
            area = torch.squeeze(area, 0)
            small_GT = torch.squeeze((area <= 1024).nonzero(as_tuple=False)).to(device)
            output = aisformer([ais_data,])
            output = output[0]['instances']
            ais_box = output.pred_boxes.tensor
            ais_box_copy = ais_box.clone()
            ais_cls = output.pred_classes
            ais_score = output.scores
            ais_mask = output.pred_amodal_masks
            pred_box = Boxes(ais_box)
            gt_box = Boxes(bbox)
            match_quality_matrix = pairwise_iou(gt_box, pred_box)
            matched_idxs, anchor_labels = anchor_matcher(match_quality_matrix)
        
            ais_box = transform.apply_boxes_torch(ais_box, original_size)
            ais_box = torch.as_tensor(ais_box, dtype=torch.float, device=device)
            if ais_box.shape[0] == 0:
                empty_img += 1
                continue

            # sam prediction
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=ais_box,
                masks=None,
            )
            
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_size).to(device)
            pred_mask = upscaled_masks > mask_threshold
            

            # round 2 mask prediction
            round2_box = find_bounding_boxes(pred_mask)
            round2_box_copy = round2_box.clone()
            round2_box = transform.apply_boxes_torch(round2_box, original_size)
            round2_box = torch.as_tensor(round2_box, dtype=torch.float, device=device)
            sparse_emb_2, dense_emb_2 = sam_model.prompt_encoder(
                points=None,
                boxes=round2_box,
                masks=None,
            )
            low_res_2, iou_pred_2 = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb_2,
                dense_prompt_embeddings=dense_emb_2,
                multimask_output=False,
            )
            upscaled_2 = sam_model.postprocess_masks(low_res_2, input_size, original_size).to(device)
            pred_mask_2 = upscaled_2 > mask_threshold




            # GT masks
            gt_mask = asegm[matched_idxs].clone()

            # oracle masks
            fp_idxs = (anchor_labels == 0).nonzero(as_tuple=False).squeeze(1)
            oracle_mask = asegm[matched_idxs].clone()
            oracle_mask[fp_idxs] = pred_mask[fp_idxs].to(oracle_mask.dtype)
            
            # index for small instances
            small_idx = torch.nonzero(matched_idxs.unsqueeze(1) == small_GT, as_tuple=False)[:, 0]
            lm_idx = ~(matched_idxs.unsqueeze(1) == small_GT).any(1)
            lm_idx = torch.nonzero(lm_idx, as_tuple=True)[0]

            vis(img_path[0], ais_box_copy, pred_mask, round2_box_copy, pred_mask_2, gt_mask, fp_idxs, small_idx, lm_idx)

            '''
            if pred_iou:
                tp_index = (iou_predictions >= filter_threshold).nonzero() 
                tp_index = tp_index[:, 0]
                pred_mask = pred_mask[tp_index]
                ais_cls = ais_cls[tp_index]
                ais_score = ais_score[tp_index]
            
            if small_idx.any() and visualize:
                idxs = torch.nonzero(matched_idxs.unsqueeze(1) == small_idx, as_tuple=False)[:, 0]
                small_num = idxs.shape[0]
                small_asegm = match_asegm[idxs]
                small_box = match_box[idxs]
                small_aisbox = ais_box_copy[idxs]
                small_aismask = ais_mask[idxs]
                small_pred = pred_mask[idxs]
                if small_num > 15:
                    vis(img_path[0], small_aisbox, small_box, small_pred, small_asegm, small_aismask)
            '''
    
    print('num of empty prediction:', empty_img)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.config_file = ais_config
    args.resume = False
    args.eval_only = True
    args.num_gpus = 1
    args.num_machines = 1
    args.machine_rank = 0
    args.dist_url = 'tcp://127.0.0.1:64153'
    args.opts = ['MODEL.WEIGHTS', ais_weight]
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines = args.num_machines,
        machine_rank = args.machine_rank,
        dist_url = args.dist_url,
        args = (args,),
    )
