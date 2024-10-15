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
import io
from PIL import Image
from amodal_efficient_sam.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits, build_efficient_sam_vitslora
from tqdm import tqdm
from pathlib import Path
import re
from amodal_efficient_sam.amodal_visualize import vis_box_test, vis_box_point_test

ais_weight = {
    'kins':'/work/u6693411/aisformer/kins/model_0119999_best.pth',
    'cocoa':'/work/u6693411/aisformer/cocoa/model_0007999.pth'
}
ais_config = {
    'kins':'/work/u6693411/aisformer/kins/config.yaml',
    'cocoa':'/work/u6693411/aisformer/cocoa/config.yaml'
}
img_suffix = {
    'kins':'.png',
    'cocoa':'.jpg'
}
anno_dict = {
    'kins':"/work/u6693411/amodal_dataset/kins/KITTI_AMODAL_DATASET/mod_instances_val_2.json",
    'cocoa':"/work/u6693411/amodal_dataset/cocoa/COCO_AMODAL_DATASET/mod_COCOA_cls_val2014_occ.json"
}
img_root = {
    'kins':'/work/u6693411/aisformer/data/datasets/KINS/test_imgs',
    'cocoa':'/home/u6693411/amodal_dataset/cocoa/val'
}

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
    def __init__(self, args):
        self.dataset = args.dataset_name
        self.ais_aug = T.AugmentationList([T.ResizeShortestEdge(short_edge_length=(800, 800), max_size=3000, sample_style='choice')])
        self.img_root = img_root[self.dataset]
        self.anno_path = anno_dict[self.dataset]
        with open(self.anno_path) as f:
            anns = json.load(f)
            self.imgs_info = anns['images']
            self.anns_info = anns['annotations']

    def __getitem__(self, index):
        img_id = self.imgs_info[index]['id']
        img_name = self.imgs_info[index]['file_name']
        img_path = os.path.join(self.img_root, img_name)
        image_np = np.array(Image.open(img_path))
        img_tensor = transforms.ToTensor()(image_np)

        # for aisformer input data dictionary
        ais_data = {'file_name': img_path}
        img = utils.read_image(ais_data['file_name'], format='BGR')
        ais_data['height'] = h = img.shape[0]
        ais_data['width'] = w = img.shape[1]
        ais_data['image_id'] = img_id
        aug_input = T.AugInput(img)
        transform = self.ais_aug(aug_input)
        img = aug_input.image
        ais_data["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

        # amodal mask GT
        asegm = [anno["segmentation"] for anno in self.anns_info[str(img_id)]]
        if self.dataset == 'kins':
            asegm = np.stack([polys_to_mask(mask, h, w) for mask in asegm])
        if self.dataset == 'cocoa':
            asegm = np.stack([rle_to_mask(mask) for mask in asegm])
        asegm = torch.as_tensor(asegm, dtype=torch.float).unsqueeze(1)  

        # amodal box GT
        abbox = np.array([anno["bbox"] for anno in self.anns_info[str(img_id)]])
        box_torch = np.hstack((abbox[:, :2], abbox[:, :2] + abbox[:, 2:]))
        box_torch = torch.as_tensor(box_torch, dtype=torch.float)

        return img_tensor, asegm, ais_data, box_torch

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

def pick_rand_point(isegm, asegm, abbox, num_fp, num_bp):
        # (ins_per_img, 1, H, W)
        # xyxy format box
        
        N, C, H, W = asegm.shape
        amask_tmp = torch.ones((N, C, H, W), dtype=asegm.dtype)
        abbox[:, [0, 1]] = abbox[:, [0, 1]] - 10
        abbox[:, [2, 3]] = abbox[:, [2, 3]] + 10
        abbox[:, [0, 2]] = np.clip(abbox[:, [0, 2]], 0, W)
        abbox[:, [1, 3]] = np.clip(abbox[:, [1, 3]], 0, H)
        abbox = np.hstack((abbox[:, :2], abbox[:, 2:] - abbox[:, :2]))
        for i in range(N):
            x, y, w, h = abbox[i].astype(int)
            amask_tmp[i, :, y:y+h, x:x+w] = asegm[i, :, y:y+h, x:x+w]
        
        cor_list = []
        lab_list = []

        for i in range(N):
            # Identifying foreground and background points
            fore_p = (isegm[i, 0] == 1).nonzero(as_tuple=False)
            back_p = (amask_tmp[i, 0] == 0).nonzero(as_tuple=False)

            # Handle the case where there are no foreground points
            if fore_p.shape[0] == 0:
                print("No pos point, use all neg point")
                indices = torch.randperm(back_p.shape[0])[:num_fp+num_bp]
                cor = back_p[indices]
                lab = torch.tensor([0]).repeat(num_fp + num_bp)
            elif back_p.shape[0] == 0:
                print("No neg point, use all pos point")
                indices = torch.randperm(fore_p.shape[0])[:num_fp+num_bp]
                cor = fore_p[indices]
                lab = torch.tensor([1]).repeat(num_fp + num_bp)
            else:
                # Handle when there are fewer foreground points than requested
                if fore_p.shape[0] < num_fp:
                    # Repeat the available foreground points to meet the required number
                    fore_p = fore_p.repeat((num_fp // fore_p.shape[0]) + 1, 1)[:num_fp]
                indices_f = torch.randperm(fore_p.shape[0])[:num_fp]
                cor_f = fore_p[indices_f]
                lab_f = torch.tensor([1]).repeat(num_fp)

                # Handle when there are fewer background points than requested
                if back_p.shape[0] < num_bp:
                    # Repeat the available background points to meet the required number
                    back_p = back_p.repeat((num_bp // back_p.shape[0]) + 1, 1)[:num_bp]
                indices_b = torch.randperm(back_p.shape[0])[:num_bp]
                cor_b = back_p[indices_b]
                lab_b = torch.tensor([0]).repeat(num_bp)

                cor = torch.cat((cor_f, cor_b))
                lab = torch.cat((lab_f, lab_b))

            # Adjusting coordinates order from (x, y) to (y, x) for consistency
            cor = cor[:, [1, 0]]
            cor_list.append(cor)
            lab_list.append(lab)

        cor_tensor = torch.stack(cor_list).to(torch.float)
        lab_tensor = torch.stack(lab_list)
        return cor_tensor, lab_tensor

'''def pick_rand_point(isegm, num_fp):
    
    N, C, H, W = isegm.shape
    cor_list = []
    lab_list = []

    for i in range(N):
        # Identifying foreground and background points
        fore_p = (isegm[i, 0] == 1).nonzero(as_tuple=False)
        #back_p = (amask_tmp[i, 0] == 0).nonzero(as_tuple=False)

        if fore_p.shape[0] == 0:
            print("  no foreground point")
            cor = torch.tensor([])
            lab = torch.tensor([])
            lab_list.append(lab)
        else:
            indices_f = torch.randperm(fore_p.shape[0])[:num_fp]
            cor_f = fore_p[indices_f]
            cor = cor_f
            lab_f = torch.tensor([1]).repeat(min(fore_p.shape[0], num_fp))
            lab_list.append(lab_f)
        # Adjusting coordinates order from (x, y) to (y, x) for consistency
        cor = cor[:, [1, 0]]
        cor_list.append(cor)
    cor_tensor = torch.stack(cor_list)
    lab_tensor = torch.stack(lab_list)
    #vis_point(img_name, cor_tensor, isegm, num_fp, num_bp)
    return cor_tensor, lab_tensor'''

class AIS_eval:
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        cfg = setup(args)
        self.aisformer = Trainer.build_model(cfg)
        DetectionCheckpointer(self.aisformer, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        self.aisformer.to(self.device)
        self.aisformer.eval()

    @staticmethod
    def parse_model_info(filename):
        # Extract the base filename without the path
        base_name = os.path.basename(filename)
        
        # Use regex to find either "full" or "lora" followed by digits, and optionally another underscore and digits for lora rank
        match = re.search(r'(full|lora)(\d+)(?:_(\d+))?', base_name)
        
        if match:
            model_type = match.group(1)  # This will be either "full" or "lora"
            number = int(match.group(2))  # This will be the number following "full" or "lora"
            lora_rank = int(match.group(3)) if match.group(3) and model_type == "lora" else None
            return model_type, number, lora_rank
        else:
            return "decoder_only", 0, 0
    @staticmethod
    def parse_prompt_info(filename):
        # Extract the base filename without the path
        base_name = os.path.basename(filename)
        
        # Use regex to find either "full" or "lora" followed by digits, and optionally another underscore and digits for lora rank
        match = re.search(r'(box|random)_(random|amodal)', base_name)
    
        if match:
            
            prompt_type = match.group(1)
            return prompt_type
        else:
            print('no prompt type!!')
            return 0
    def load_model(self):
        if self.model_type == "lora":
            print(f'Building LoRA EfficientSAM, # rank: {self.lora_rank}, # encoder block: {self.model_block}')
            self.model = build_efficient_sam_vitslora(lora_rank=self.lora_rank, block_num=self.model_block)
        else:
            print(f'Building Original EfficientSAM, # encoder block to load ckpt: {self.model_block}')
            self.model = build_efficient_sam_vits()

    def load_ckpt(self, ckpt_name):
        self.ckpt_name = ckpt_name
        print(f'\nLoading checkpoint: {self.ckpt_name}')
        model_type, model_block, lora_rank = self.parse_model_info(ckpt_name)
        self.model_type = model_type
        self.model_block= model_block
        self.lora_rank = lora_rank

        self.load_model()
        checkpoint = torch.load(os.path.join(self.args.test_ckpt_root, ckpt_name))
        self.model.mask_decoder.load_state_dict(checkpoint['decoder'])
        # load encoder ckpt if block number is not none
        if self.model_block:
            self.model.image_encoder.load_state_dict(checkpoint['encoder'])
        
        self.model.to(self.device)
        self.model.eval()

    def load_dataset(self):
        self.dataset = AmodalDataset(self.args)
        self.testing_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        print(f'Dataset Length for {self.args.dataset_name} test set: {len(self.testing_loader)}')

    def save_instance_result(self, img_id, masks, classes, scores):
        result_list = []
        if self.args.dataset_name == 'cocoa':
            ais_to_ann = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
        if self.args.dataset_name == 'kins':
            ais_to_ann = {0: 1, 1: 2, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
        
        for i, m in enumerate(masks):
            mask = torch.squeeze(m, 0)
            np_mask = mask.detach().cpu().numpy().astype(np.uint8)
            rle_mask = encode(np.asfortranarray(np_mask))
            rle_mask['counts'] = rle_mask['counts'].decode('ascii')
            cat_id = classes[i].item()
            score = scores[i].item()
            result_list.append({'image_id': img_id, 'category_id': ais_to_ann[cat_id], 'segmentation': rle_mask, 'score': score})
        return result_list
    
    def check_save_root(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

    def save_json(self, result_dict):
        root = os.path.join(self.args.result_save_root, self.ckpt_name)
        self.check_save_root(root)
        path = os.path.join(root, 'result_iou.json')
        with open(path, 'w') as f:
            json.dump(result_dict["iou"], f)
        path = os.path.join(root, 'result.json')
        with open(path, 'w') as f:
            json.dump(result_dict["no_iou"], f)
        
    def run_eval(self):
        result_dict = {"no_iou":[], "iou":[]}
        mask_threshold = 0
        empty_img = 0
        for i, data in enumerate(tqdm(self.testing_loader)):
            img_tensor, asegm, ais_data, abbox = data
            if self.args.mode_visualize and i >= self.args.vis_num:
                break
            with torch.no_grad():
                ais_data['image'] = torch.squeeze(ais_data['image'], 0).to(self.device)
                ais_data['height'] = ais_data['height'].item()
                ais_data['width'] = ais_data['width'].item()
                ais_data['image_id'] = ais_data['image_id'].item()
                img_id = ais_data['image_id']
                output = self.aisformer([ais_data,])
                output = output[0]['instances']
                ais_box = output.pred_boxes.tensor
                ais_cls = output.pred_classes
                ais_score = output.scores
                ais_mask = output.pred_amodal_masks
                ais_mask_vis = output.pred_visible_masks
                ais_box = torch.as_tensor(ais_box, dtype=torch.float, device=self.device)
                if ais_box.shape[0] == 0:
                    empty_img += 1
                    continue
                ais_box = ais_box.reshape((-1, 2, 2)).unsqueeze(0)
                box_label = torch.tensor([2,3])
                box_label = box_label.repeat(ais_box.shape[1], 1).unsqueeze(0).to(self.device)
                prompt = ais_box
                prompt_label = box_label
                
                # evaluate with point + box
                if self.args.point_and_box:

                    ais_point, point_label = pick_rand_point(
                        ais_mask_vis.unsqueeze(1).cpu().detach(), 
                        ais_mask.unsqueeze(1).cpu().detach(), 
                        ais_box.reshape((-1, 4)).cpu().detach(), 
                        self.args.num_fp,
                        self.args.num_bp
                        )
                    ais_point = ais_point.unsqueeze(0).to(self.device)
                    point_label = point_label.unsqueeze(0).to(self.device)
                    prompt = torch.cat((ais_box, ais_point), 2)
                    prompt_label = torch.cat((box_label, point_label), 2)
                    
                
                # EfficientSAM prediction
                predicted_logits, predicted_iou = self.model(
                    img_tensor.to(self.device),
                    prompt,
                    prompt_label,
                )
                #sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
                #predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
                #predicted_logits = torch.take_along_dim(
                #    predicted_logits, sorted_ids[..., None, None], dim=2
                #)
                predicted_logits = predicted_logits[:, :, slice(0, 1), :, :]
                predicted_iou = predicted_iou[:, :, slice(0, 1)]
                
                pred_mask = torch.ge(predicted_logits, mask_threshold).squeeze(0)
                
                if not self.args.mode_visualize:
                    result_no_iou = self.save_instance_result(img_id, pred_mask, ais_cls, ais_score)
                    ais_score *= predicted_iou.squeeze()
                    result_iou = self.save_instance_result(img_id, pred_mask, ais_cls, ais_score)
                    result_dict["iou"].extend(result_iou)
                    result_dict["no_iou"].extend(result_no_iou)
                else:
                    score_filter = ais_score >= self.args.vis_score_filter
                    score_filter = score_filter.cpu().detach()
                    
                    # Apply the filter to both scores and masks
                    ais_score = ais_score[score_filter]
                    pred_mask = pred_mask.squeeze(0)[score_filter]
                    prompt = prompt.squeeze(0)[score_filter]
                    abbox = abbox.reshape((-1, 2, 2))
                    asegm = asegm.squeeze(0)

                    if self.args.point_and_box:
                        vis_box_point_test(
                            os.path.join(self.args.vis_save_root, self.ckpt_name), 
                            img_tensor.squeeze(0), 
                            prompt[:, :2, :], 
                            prompt[:, 2:, :], 
                            pred_mask, 
                            abbox,
                            asegm, 
                            self.args.num_fp, 
                            self.args.num_bp,
                            str(ais_data['image_id'])+'.png'
                        )
                    else:
                        vis_box_test(
                            os.path.join(self.args.vis_save_root, self.ckpt_name), 
                            img_tensor.squeeze(0), 
                            prompt[:, :2, :], 
                            pred_mask, 
                            abbox, 
                            asegm, 
                            str(ais_data['image_id'])+'.png'
                        )
                
        
        if not self.args.mode_visualize:
            self.save_json(result_dict)

    def Eval(self):
        all_ckpt = os.listdir(self.args.test_ckpt_root)
        for ckpt in all_ckpt:
            if self.args.point_and_box:
                prompt_type = self.parse_prompt_info(ckpt)
                if prompt_type != 'random':
                    continue
            self.load_ckpt(ckpt)
            self.load_dataset()
            self.run_eval()

def main(args):
    evaluator = AIS_eval(args)
    evaluator.Eval()

def everytype2bool(v):
    v = v.lower()
    if v.isnumeric():
        return bool(int(v))
    if v in ['', 'no', 'none', 'false']:
        return False
    return True

def get_parser(parser):
    
    parser.add_argument('--dataset_name', type=str, default='cocoa', help='Name of the dataset to test with AISFormer, can be kins / cocoa')
    parser.add_argument('--point_and_box', type=everytype2bool, default=False, help='Use box + point as prompt')
    parser.add_argument('--num_fp', type=int, default=5, help='Number of positive point prompts')
    parser.add_argument('--num_bp', type=int, default=5, help='Number of negative point prompts')
    parser.add_argument('--result_save_root', type=str, default='/work/u6693411/nv_ais_result_x3', help='Root for saving aisformer eval result')
    parser.add_argument('--test_ckpt_root', type=str, default='/work/u6693411/ckpt_x3', help='Root of the ckpt to be evaluated')
    parser.add_argument('--mode_visualize', type=everytype2bool, default=True, help='Visualize only, do not save prediction result')
    parser.add_argument('--vis_save_root', type=str, default='/work/u6693411/x3_vis', help='Visualize only, do not save prediction result')
    parser.add_argument('--vis_num', type=int, default=15, help='Number of images to visualize for each ckpt version')
    parser.add_argument('--vis_score_filter', type=float, default=0.3, help='Not visualizing instances with score < this threshold')

    return parser

if __name__ == "__main__":
    torch.manual_seed(0)
    parser = default_argument_parser()
    parser = get_parser(parser)
    args = parser.parse_args()
    args.config_file = ais_config[args.dataset_name]
    args.resume = False
    args.eval_only = True
    args.num_gpus = 1
    args.num_machines = 1
    args.machine_rank = 0
    args.dist_url = 'tcp://127.0.0.1:64153'
    args.opts = ['MODEL.WEIGHTS', ais_weight[args.dataset_name]]
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines = args.num_machines,
        machine_rank = args.machine_rank,
        dist_url = args.dist_url,
        args = (args,),
    )
