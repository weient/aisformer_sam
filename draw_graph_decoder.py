from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import os
import re



import pandas as pd
import matplotlib.pyplot as plt
import colorsys

def generate_distinct_colors(n):
    hue_partition = 1.0 / (n + 1)
    colors = []
    for value in range(n):
        hue = value * hue_partition
        saturation = 0.8 + (value % 2) * 0.1  # Alternating saturation
        lightness = 0.6 + (value % 2) * 0.1   # Alternating lightness
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors

def create_label(name_tuple):
    name_tuple = (str(part) for part in name_tuple)
    return "_".join([part for part in name_tuple if "-" not in str(part)])



def draw_graph(excel_path, fig_name_prefix='ais+effsam_mAP', versions_to_plot=[], noclass=True, eval_on='KINS'):

    # Read the Excel file
    df = pd.read_excel(excel_path)
    #df = pd.read_excel('/home/u6693411/ais/AISFormer/final_result_iou_nocls.xlsx')
    #fig_name_prefix = 'ais+effsam_mAP_b+p'
    #noclass = True

    #df = df[df['Iteration'] != 'latest']


    # for comparing different version
    if versions_to_plot != []:
        #versions_to_plot = ["full 5 - box_amodal -", "full 5 - box_amodal iou0.02", "full 5 - box_amodal iou0.08"]
        mask = df.apply(lambda row: f"{row['Encoder type']} {row['Encoder block']} {row['LoRA rank']} {row['Prompt type']} {row['Setting']}" in versions_to_plot, axis=1)
        df = df[mask]
    #df = df.rename(columns={"Prompt type": "Prompt_type"})
    #df = df.rename(columns={"Loss type": "Loss_type"})
    print(df.columns)
    # Group the data by Dataset
    dataset_groups = df.groupby(['Dataset', "Prompt type", 'Setting', 'Loss type'])
    n_colors = len(dataset_groups)
    color_palette = generate_distinct_colors(n_colors)
    plt.figure(figsize=(12, 6))
    for (dataset, dataset_df), color in zip(dataset_groups, color_palette):
        label = create_label(dataset)
        group_sorted = dataset_df.sort_values('Iteration')
        plt.plot(group_sorted['Iteration'], group_sorted['mAP'], marker='o', linestyle='-', label=label, color=color)
    
    # Customize the plot
    title = f'AIS+EffSAM mAP eval on {eval_on}'
    title = f'(class agnostic) {title}'if noclass else title
    plt.xlabel('Iteration')
    plt.ylabel('mAP')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{fig_name_prefix}_{eval_on}.png', dpi=300, bbox_inches='tight')
    
    # Close the figure to free up memory
    plt.close()

    print("All plots have been saved.")

if __name__ == "__main__":
    draw_graph(
        excel_path='/home/u6693411/ais/AISFormer/iou_class_agnostic.xlsx', 
        fig_name_prefix='ais+effsam_mAP', 
        versions_to_plot=[], 
        noclass=True,
        eval_on='KINS'
    )