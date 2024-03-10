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
        if dataset == 'kins':
            self.data_root = imgemb_root
            self.img_root = img_root
            self.anno_path = anno_path
            with open(self.anno_path) as f:
                anns = json.load(f)
                self.imgs_info = anns['images']
                self.anns_info = anns['annotations']
    def __getitem__(self, index):
        img_id = self.imgs_info[index]['id']
        img_name = self.imgs_info[index]['file_name']
        path = os.path.join(self.data_root, img_name.split('.png')[0]+'.pt')
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
        # amodal mask GT
        asegm = [anno["segmentation"] for anno in self.anns_info[str(img_id)]]
        asegm = np.stack([polys_to_mask(mask, h, w) for mask in asegm])
        asegm = torch.as_tensor(asegm, dtype=torch.float, device=device).unsqueeze(1)  
        # random point prompt
        point_torch = []
        # amodal bbox GT
        box_torch = []
        abbox = np.array([anno["bbox"] for anno in self.anns_info[str(img_id)]])
        box_torch = np.hstack((abbox[:, :2], abbox[:, :2] + abbox[:, 2:]))
        box_torch = torch.as_tensor(box_torch, dtype=torch.float, device=device)
        return img_emb, asegm, box_torch, point_torch, original_size, input_size, ais_data

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


vit_dict = {
    'vit_b':"/home/weientai18/SAM/SAM_ckpt/sam_vit_b_01ec64.pth", 
    'vit_h':"/home/weientai18/SAM/SAM_ckpt/sam_vit_h_4b8939.pth"
    }
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit_type = 'vit_h'
lr = 1e-4
dataset_name = 'kins'
img_root = '/home/weientai18/ais/data/datasets/KINS/train_imgs'
imgemb_root = '/work/weientai18/amodal_dataset/training_imgemb_h'
anno_path = '/home/weientai18/SAM/mod_instances_train.json'
tb_save_path = '/work/weientai18/amodal_dataset/checkpoint/runs'
ckpt_save_path = '/work/weientai18/amodal_dataset/checkpoint'
resume_ckpt = '/work/weientai18/amodal_dataset/checkpoint/model_20240309_044028_29'
resume_ep = 30
resume = True
train_val_ratio = [0.8, 0.2]
EPOCHS = 2000
anchor_matcher = Matcher(
        thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
    )
def vis(img_path, ais_box, matched_box, flag):
    print(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    matched_img = copy.deepcopy(img)
    ais_img = copy.deepcopy(img)
    matched_img = cv2.addWeighted(matched_img, 0.4, matched_img, 0, 0)
    ais_img = cv2.addWeighted(ais_img, 0.4, ais_img, 0, 0)
    color_match = (100, 0, 255)
    color_ais = (0, 255, 100)
    for i, box in enumerate(ais_box):
        matched = matched_box[i]
        ais_img = cv2.rectangle(ais_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_ais, 1)
        matched_img = cv2.rectangle(matched_img, (int(matched[0]), int(matched[1])), (int(matched[2]), int(matched[3])), color_match, 1)
    final_img = np.concatenate((matched_img, ais_img), axis=0)
    cv2.imwrite('/work/weientai18/ais_matched_{}.png'.format(flag), final_img)




def train_one_epoch(epoch_index, tb_writer, training_loader):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        data = [None if x == [] else x for x in data]
        image_embedding, asegm, bbox, point, original_size, input_size, ais_data = data
        with torch.no_grad():
            ais_data['image'] = torch.squeeze(ais_data['image'], 0).to(device)
            ais_data['height'] = ais_data['height'].item()
            ais_data['width'] = ais_data['width'].item()
            ais_data['image_id'] = ais_data['image_id'].item()
            
            original_size = [j.item() for j in original_size]
            input_size = [j.item() for j in input_size]
            asegm = torch.squeeze(asegm, 0)
            bbox = torch.squeeze(bbox, 0)
            output = aisformer([ais_data,])
            output = output[0]['instances']
            ais_box = output.pred_boxes.tensor
            ais_cls = output.pred_classes
            ais_score = output.scores
            pred_box = Boxes(ais_box)
            gt_box = Boxes(bbox)
            match_quality_matrix = pairwise_iou(gt_box, pred_box)
            matched_idxs, anchor_labels = anchor_matcher(match_quality_matrix)
            #idxs_2, labels_2 = anchor_matcher_2(match_quality_matrix)
            match_asegm = asegm[matched_idxs]
            #if i == 0:
            #    vis(img_path[0], ais_box, bbox[matched_idxs], 'train')
            ais_box = transform.apply_boxes_torch(ais_box, original_size)
            ais_box = torch.as_tensor(ais_box, dtype=torch.float, device=device)
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
        
        # Compute the loss and its gradients
        loss = loss_fn(upscaled_masks, match_asegm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    
    return last_loss


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
    if resume:
        sam_model.mask_decoder.load_state_dict(torch.load(resume_ckpt))
    sam_model.to(device)

    global transform
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    global optimizer 
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr = lr)
    
    global loss_fn
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Create datasets for training & validation
    dataset = AmodalDataset(dataset_name)
    #training_set, validation_set = random_split(dataset, train_val_ratio)

    # shuffle for training, not for validation
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    #training_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True)
    #validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(tb_save_path, 'fashion_trainer_{}'.format(timestamp)))
    epoch_number = 0
    if resume:
        epoch_number = resume_ep
    best_vloss = 1_000_000.

    # gradient tracking is on
    sam_model.mask_decoder.train(True)
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        avg_loss = train_one_epoch(epoch_number, writer, data_loader)
        #print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('LOSS train {}'.format(avg_loss))
        # Log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training Loss',
                        { 'Training' : avg_loss},
                        epoch_number + 1)
        #writer.add_scalars('Training vs. Validation Loss',
        #                { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if epoch_number % 10 == 9:
            model_path = os.path.join(ckpt_save_path, 'model_{}_{}'.format(timestamp, epoch_number))
            torch.save(sam_model.mask_decoder.state_dict(), model_path)
        
        epoch_number += 1
    


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.config_file='/work/weientai18/aisformer/aisformer_R_50_FPN_1x_amodal_kins_160000_resume/config.yaml'
    args.resume=False
    args.eval_only=True
    args.num_gpus=1
    args.num_machines=1
    args.machine_rank=0
    args.dist_url='tcp://127.0.0.1:64153'
    args.opts=['MODEL.WEIGHTS', '/work/weientai18/aisformer/aisformer_R_50_FPN_1x_amodal_kins_160000_resume/model_final.pth']
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
