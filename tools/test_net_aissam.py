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
import torchvision.transforms.functional as TF
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

def generate_random_colors(num_colors):
    R = random.sample(range(50, 200), num_colors)
    G = random.sample(range(50, 200), num_colors)
    B = random.sample(range(50, 200), num_colors)
    colors = list(zip(R, G, B))
    return colors

def blend(b, g, r, mask, img):
    mask = mask.cpu().numpy().astype(np.uint8)
    mask = np.stack((b*mask, g*mask, r*mask), axis=2)
    img = cv2.addWeighted(img, 1, mask, 0.7, 0)
    return img

def vis(img_path, ais_box, gt_box, sam_mask, ais_mask, gt_mask):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_name = img_path.split('/')[-1]

    save_path = os.path.join('/work/u6693411/', img_name)
    sam_img = copy.deepcopy(img)
    ais_img = copy.deepcopy(img)
    gt_img = copy.deepcopy(img)
    sam_img = cv2.addWeighted(sam_img, 0.2, sam_img, 0, 0)
    ais_img = cv2.addWeighted(ais_img, 0.2, ais_img, 0, 0)
    gt_img = cv2.addWeighted(gt_img, 0.2, gt_img, 0, 0)
    ais_num_colors = generate_random_colors(ais_box.shape[0])
    gt_num_colors = generate_random_colors(gt_box.shape[0])
    color_gt = (100, 0, 255)
    color_gt_small = (0, 255, 100)
    color_ais = (255, 100, 0)
    
    for i in range(gt_box.shape[0]):
        b, g, r = gt_num_colors[i]
        gt_img = blend(b, g, r, gt_mask[i].squeeze(0), gt_img)
    
    for i in range(gt_box.shape[0]):
        gt = gt_box[i]
        color = color_gt
        #color = color_gt_small if i in small_idx else color_gt
        gt_img = cv2.rectangle(gt_img, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), color, 1)

    for i in range(ais_box.shape[0]):
        b, g, r = ais_num_colors[i]
        sam_img = blend(b, g, r, sam_mask[i].squeeze(0), sam_img)
        ais_img = blend(b, g, r, ais_mask[i], ais_img)

    for i in range(ais_box.shape[0]):
        ais = ais_box[i]
        color = color_ais
        sam_img = cv2.rectangle(sam_img, (int(ais[0]), int(ais[1])), (int(ais[2]), int(ais[3])), color, 1)
        ais_img = cv2.rectangle(ais_img, (int(ais[0]), int(ais[1])), (int(ais[2]), int(ais[3])), color, 1)
    
    final_img = np.concatenate((gt_img, sam_img), axis=0)
    final_img = np.concatenate((final_img, ais_img), axis=0)
    cv2.imwrite(save_path, final_img)

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

oracle = False
matcher_iou = 0.6
pred_iou = True
load_mask_ckpt = True
vit_type = 'vit_h'
dataset_name = 'cocoa'
ais_weight = '/work/u6693411/aisformer/cocoa/model_0007999.pth'
#ais_weight = '/work/u6693411/aisformer/kins/model_0119999_best.pth'
ais_config = '/work/u6693411/aisformer/cocoa/config.yaml'
#ais_config = '/work/u6693411/aisformer/kins/config.yaml'
result_save_root = '/work/u6693411'
result_save_path = 'box+point+mask+randinit_cocoa' # _{}
sam_ckpt = '/work/u6693411/amodal_dataset/checkpoint/model_20240611_222841' #_{}_
sam_ckpt_suffix = 'box+point+mask+randinit_cocoa'
prompt_ckpt_suffix = 'box+point+mask+randinit_cocoa_prompt'
ckpt_id = [17]
# visualize results
draw_output = True

# To run mask decoder for second time (using mask input)
second_iter = False

img_suffix = {
    'kins':'.png',
    'cocoa':'.jpg'
}
vit_dict = {
    'vit_b':"/home/u6693411/SAM/SAM_ckpt/sam_vit_b_01ec64.pth", 
    'vit_h':"/home/u6693411/SAM/SAM_ckpt/sam_vit_h_4b8939.pth"
}
anno_dict = {
    'kins':"/work/u6693411/amodal_dataset/kins/KITTI_AMODAL_DATASET/mod_instances_val_2.json",
    'cocoa':"/work/u6693411/amodal_dataset/cocoa/COCO_AMODAL_DATASET/mod_COCOA_cls_val2014.json"
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_root = {
    'kins':'/home/u6693411/ais/data/datasets/KINS/test_imgs',
    'cocoa':'/home/u6693411/ais/data/datasets/COCOA/test_imgs'
}
imgemb_root = {
    'kins':'/work/u6693411/amodal_dataset/testing_imgemb_h',
    'cocoa':'/work/u6693411/amodal_dataset/coco_testing_imgemb_h'
}
anno_path = anno_dict[dataset_name]
anchor_matcher = Matcher(
        thresholds=[matcher_iou], labels=[0, 1], allow_low_quality_matches=False
    )


def save_instance_result(img_id, masks, classes, scores, suffix):
    if dataset_name == 'cocoa':
        ais_to_ann = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
    if dataset_name == 'kins':
        ais_to_ann = {0: 1, 1: 2, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
    if suffix not in result_list.keys():
        result_list[suffix] = []
    for i, m in enumerate(masks):
        mask = torch.squeeze(m, 0)
        np_mask = mask.detach().cpu().numpy().astype(np.uint8)
        rle_mask = encode(np.asfortranarray(np_mask))
        rle_mask['counts'] = rle_mask['counts'].decode('ascii')
        cat_id = classes[i].item()
        score = scores[i].item()
        result_list[suffix].append({'image_id': img_id, 'category_id': ais_to_ann[cat_id], 'segmentation': rle_mask, 'score': score})

def find_bounding_box(mask):
    true_positions = torch.nonzero(mask, as_tuple=False)
    
    top_left = true_positions.min(dim=0).values
    bottom_right = true_positions.max(dim=0).values
    
    return [top_left[0].item(), top_left[1].item(), bottom_right[0].item(), bottom_right[1].item()]

def resize_mask(ref_mask: torch.Tensor, longest_side: int = 256):
    """
    Resize a mask to have its longest side equal to the specified value using PyTorch.

    Args:
        ref_mask (torch.Tensor): The mask to be resized. Expected shape is (H, W).
        longest_side (int, optional): The length of the longest side after resizing. Default is 256.
    """
    height, width = ref_mask.shape[:2]
    if height > width:
        new_height = longest_side
        new_width = int(width * (new_height / height))
    else:
        new_width = longest_side
        new_height = int(height * (new_width / width))
    # Resize the mask using bilinear interpolation
    ref_mask = TF.resize(ref_mask.unsqueeze(0), [new_height, new_width], interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

    # Calculate padding
    pad_height = 256 - new_height
    pad_width = 256 - new_width
    padding = [0, 0, pad_width, pad_height]  # Padding on the right and bottom

    # Apply padding
    ref_mask = TF.pad(ref_mask, padding, fill=0)

    return ref_mask

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
    global transform
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    global result_list
    for ID in ckpt_id:
        
        result_list = {} 
        sam_model.mask_decoder.load_state_dict(torch.load(sam_ckpt+'_{}_'.format(ID)+sam_ckpt_suffix))
        print('****** loading {} epoch {} ******'.format(sam_ckpt_suffix, ID))
        if load_mask_ckpt:
            sam_model.prompt_encoder.load_state_dict(torch.load(sam_ckpt+'_{}_'.format(ID)+prompt_ckpt_suffix))
            print('****** loading {} epoch {} ******'.format(prompt_ckpt_suffix, ID))
        mask_threshold = sam_model.mask_threshold
        sam_model.to(device)
        sam_model.eval()


        # Create datasets for training & validation
        dataset = AmodalDataset(dataset_name)

        # test dataloader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        samples = random.sample(range(len(dataset)), 10)
        empty_img = 0
        #filter_preds = {}
        for i, data in enumerate(data_loader):
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
                area = torch.squeeze(area)
                
                small_idx = torch.squeeze((area <= 1024).nonzero(as_tuple=False)).to(device)
                output = aisformer([ais_data,])
                output = output[0]['instances']
                ais_box = output.pred_boxes.tensor
                ais_box_copy = ais_box.clone()
                ais_cls = output.pred_classes
                ais_score = output.scores
                ais_mask = output.pred_amodal_masks
                resized_segm = torch.zeros((ais_mask.shape[0], 256, 256))
                #aug_kernel = [3, 5, 7, 9, 11, 13]
                for idx, j in enumerate(ais_mask):
                    tmp = resize_mask(j)
                    resized_segm[idx,:,:] = tmp[:,:]
                resized_segm = torch.as_tensor(resized_segm, dtype=torch.float, device=device).unsqueeze(1)
                pred_box = Boxes(ais_box)
                gt_box = Boxes(bbox)
                match_quality_matrix = pairwise_iou(gt_box, pred_box)
                matched_idxs, anchor_labels = anchor_matcher(match_quality_matrix)
                ais_box = transform.apply_boxes_torch(ais_box, original_size)
                ais_box = torch.as_tensor(ais_box, dtype=torch.float, device=device)
                if ais_box.shape[0] == 0:
                    empty_img += 1
                    continue
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=ais_box,
                    #masks=None,
                    masks=resized_segm,
                )
                if second_iter:
                    low_res_masks, iou_predictions = sam_model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    # second round
                    sec_sparse_embeddings, sec_dense_embeddings = sam_model.prompt_encoder(
                        points=None,
                        boxes=ais_box,
                        masks=low_res_masks,
                    )
                    sec_low_res_masks, iou_predictions = sam_model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sec_sparse_embeddings,
                        dense_prompt_embeddings=sec_dense_embeddings,
                        multimask_output=False,
                    )
                    upscaled_masks = sam_model.postprocess_masks(sec_low_res_masks, input_size, original_size).to(device)
                    pred_mask = upscaled_masks > mask_threshold
                else:
                    low_res_masks, iou_predictions = sam_model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_size).to(device)
                    pred_mask = upscaled_masks > mask_threshold
                if oracle:
                    fp_idxs = (anchor_labels == 0).nonzero(as_tuple=False).squeeze(1)
                    match_asegm = asegm[matched_idxs].clone()
                    match_asegm[fp_idxs] = pred_mask[fp_idxs].to(match_asegm.dtype)
                    pred_mask = match_asegm

                save_instance_result(img_id, pred_mask, ais_cls, ais_score, "")

                if pred_iou:
                    ais_score *= iou_predictions.squeeze(1)
                    save_instance_result(img_id, pred_mask, ais_cls, ais_score, "iou")
                
                if draw_output and i < 10:
                    vis(img_path[0], ais_box_copy, bbox, pred_mask, ais_mask, asegm)
                

        for th in result_list.keys():
            if th == "":
                p = os.path.join(result_save_root, result_save_path+'_{}'.format(ID)+'.json')
            else:
                p = os.path.join(result_save_root, result_save_path+'_{}'.format(ID)+'_{}_filter.json'.format(th))
                #print('num of filter out instances of threshold {}: {}'.format(th, filter_preds[th]))
            with open(p, 'w') as f:
                json.dump(result_list[th], f)
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
