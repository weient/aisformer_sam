from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import os
import re
import pandas as pd
ANN = {
    'cocoa': '/work/u6693411/amodal_dataset/cocoa/COCO_AMODAL_DATASET/COCO_amodal_val2014_with_classes.json',
    'cocoa_occ': '/work/u6693411/amodal_dataset/cocoa/COCO_AMODAL_DATASET/COCO_amodal_val2014_with_classes_occ.json',
    'cocoa_unocc': '/work/u6693411/amodal_dataset/cocoa/COCO_AMODAL_DATASET/COCO_amodal_val2014_with_classes_unocc.json',
    'kins': '/work/u6693411/amodal_dataset/kins/KITTI_AMODAL_DATASET/instances_val.json',
    'kins_occ': '/work/u6693411/amodal_dataset/kins/KITTI_AMODAL_DATASET/instances_val_occ.json', 
    'kins_unocc': '/work/u6693411/amodal_dataset/kins/KITTI_AMODAL_DATASET/instances_val_unocc.json'
}


def parse_model_info(filename):
    # Extract the base filename without the path
    base_name = os.path.basename(filename)
    # Use regex to find either "full" or "lora" followed by digits, and optionally another underscore and digits for lora rank
    match = re.search(r'(\d+|latest)_(?:(combined|all)_)?(full|lora)(\d+)(?:_(\d+))?_(box|random)_(random|amodal)(?:_([A-Za-z0-9._]+))?', base_name)
    
    if match:
        iteration = int(match.group(1)) if match.group(1) != 'latest' else 'latest'
        dataset = match.group(2) if match.group(2) else "pix2gestalt"
        encoder_type = match.group(3)
        encoder_block = int(match.group(4))
        lora_rank = int(match.group(5)) if match.group(5) else '-'
        prompt_type = match.group(6)+'_'+match.group(7)
        setting = match.group(8) if match.group(8) else '-'
        return {'Dataset':dataset, 'Encoder type':encoder_type, 'Encoder block':encoder_block, 'LoRA rank':lora_rank, 'Prompt type':prompt_type, 'Setting':setting, 'Iteration':iteration}
    else:
        return {}

def parse_model_info_2(filename):
    parts = filename.split('_')
    data = {
        'Dataset': parts[3],
        'Prompt type': parts[4],
        'Setting': parts[5],
    }
    if len(parts) == 10:
        data['Loss type'] = parts[6]
        data['Iteration'] = int(parts[9])
    elif len(parts) == 9: 
        data['Loss type'] = '-' 
        data['Iteration'] = int(parts[8])
    else:
        raise ValueError(f"Unexpected filename format: {filename}")
    
    return data

def dictionaries_to_excel(dict_list, save_path='result.xlsx', sort_by=None, ascending=True):
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(dict_list)
    if sort_by:
        if isinstance(sort_by, list) and all(col in df.columns for col in sort_by):
            df = df.sort_values(by=sort_by, ascending=ascending)
        elif isinstance(sort_by, str) and sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
        else:
            print(f"Warning: Invalid sort column(s). Data will not be sorted.")
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(save_path, engine='openpyxl')
    # Write the dataframe to an excel sheet
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file
    writer.close()
    print(f"Data has been written to {save_path}")

def coco_eval_mask(annFile, resFile, class_agnostic):
    annType = 'segm'
    cocoGt=COCO(annFile)
    cocoDt=cocoGt.loadRes(resFile)
    imgIds=sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    if class_agnostic:
        cocoEval.params.useCats = 0
    cocoEval.params.imgIds = imgIds  
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    result = {
        'mAP': cocoEval.stats[0],
        'AP50': cocoEval.stats[1],
        'AP75': cocoEval.stats[2],
        'APs': cocoEval.stats[3],
        'APm': cocoEval.stats[4],
        'APl': cocoEval.stats[5]
    }
    return result

def cal_result(root, file_name, dataset_name, class_agnostic):
    all_dir = set(os.listdir(root))
    dir_list = []
    for d in all_dir:
        
        info_dic = parse_model_info_2(d)
        resFile = os.path.join(root, d, file_name)
        annFile = ANN[dataset_name]
        result_dic = coco_eval_mask(annFile, resFile, class_agnostic)
        info_dic.update(result_dic)
        dir_list.append(info_dic)
    
    return dir_list

def main():
    dataset_name = 'kins_unocc'
    # TODO
    class_agnostic = True
    root = '/work/u6693411/ais_result_kins_df_rbbox'
    file_name = ['result_iou.json', 'result.json']

    for name in file_name:
        dir_list = cal_result(root, name, dataset_name, class_agnostic)
        save_path = f'final_{name.split(".json")[0]}.xlsx' if not class_agnostic else f'final_{name.split(".json")[0]}_nocls.xlsx'
        dictionaries_to_excel(dir_list,
            save_path=save_path,
            sort_by=['Dataset', 'Prompt type', 'Setting', 'Loss type', 'Iteration'],
            ascending=[True, True, True, True, True])

if __name__ == "__main__":
    main()