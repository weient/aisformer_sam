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
    # Updated regex pattern to capture the number after "full"
    print(filename)
    pattern = r"""
        model_
        (?P<Dataset>all|wococo|walt)_
        (?:(?P<Encoder_type>full)(?P<Encoder_block>\d{1})_)?
        (?P<Prompt_type>box)_
        (?P<box_type>random|amodal)_
        (?P<Setting>.*?)_
        (?:(?P<date>\d{8})_(?P<time>\d{6})_(?P<Iteration>\d+)|(?P<is_latest>latest))
    """
    
    match = re.match(pattern, filename, re.VERBOSE)
    
    if not match:
        print('!!filename do not match any format!!')
        return None  # Return None if the filename doesn't match the expected format
    
    result = match.groupdict()
    
    if result['is_latest']:
        result['Iteration'] = 'latest'
    else:
        result['Iteration'] = int(result['Iteration'])
    del result['is_latest']
    
    # Convert date and time to datetime object
    #result['datetime'] = datetime.strptime(f"{result['date']}_{result['time']}", "%Y%m%d_%H%M%S")
    del result['date']
    del result['time']
    
    # Handle the 'full' number
    if result['Encoder_block']:
        result['Encoder block'] = int(result['Encoder_block'])
    else:
        result['Encoder block'] = '-'
    del result['Encoder_block']
    result['Prompt type'] = f"{result['Prompt_type']}_{result['box_type']}"
    del result['Prompt_type']
    del result['box_type']

    result['Encoder type'] = result['Encoder_type']
    del result['Encoder_type']

    #result['is_latest'] = False
    
    return result
    
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
        info_dic = parse_model_info(d)
        #print(info_dic)
        resFile = os.path.join(root, d, file_name)
        annFile = ANN[dataset_name]
        result_dic = coco_eval_mask(annFile, resFile, class_agnostic)
        info_dic.update(result_dic)
        dir_list.append(info_dic)
    
    return dir_list

def main():
    dataset_name = 'kins_occ'
    # TODO
    class_agnostic = True
    root = '/work/u6693411/ais_result_kins_nv'
    file_name = ['result_iou.json', 'result.json']

    for name in file_name:
        dir_list = cal_result(root, name, dataset_name, class_agnostic)
        save_path = f'final_{name.split(".json")[0]}_kins.xlsx' if not class_agnostic else f'final_{name.split(".json")[0]}_nocls_kins.xlsx'
        dictionaries_to_excel(dir_list,
            save_path=save_path,
            sort_by=['Dataset', 'Encoder type', 'Encoder block', 'Prompt type', 'Setting', 'Iteration'],
            ascending=[True, True, True, True, True, True])

if __name__ == "__main__":
    main()