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
    name_tuple = [str(part) for part in name_tuple]
    if 'True' in name_tuple:
        name_tuple.append('b+p')
    #name_tuple = (str(part) for part in name_tuple)
    name_tuple = [part for part in name_tuple if part != '-']
    name_tuple = [part for part in name_tuple if (part != 'False' and part != 'True')]
    label = "_".join(name_tuple)
    #label = f'{label}_b+p' if 'b+p' in name_tuple else label
    return label



def draw_graph(excel_path, fig_name_prefix='ais+effsam_mAP', versions_to_plot=[], noclass=True):

    # Read the Excel file
    df = pd.read_excel(excel_path)
    #df = pd.read_excel('/home/u6693411/ais/AISFormer/final_result_iou_nocls.xlsx')
    #fig_name_prefix = 'ais+effsam_mAP_b+p'
    #noclass = True

    df = df[df['Iteration'] != 'latest']


    # for comparing different version
    if versions_to_plot != []:
        #versions_to_plot = ["full 5 - box_amodal -", "full 5 - box_amodal iou0.02", "full 5 - box_amodal iou0.08"]
        mask = df.apply(lambda row: f"{row['Encoder type']} {row['Encoder block']} {row['LoRA rank']} {row['Prompt type']} {row['Setting']}" in versions_to_plot, axis=1)
        df = df[mask]


    # Group the data by Dataset
    dataset_groups = df.groupby('Dataset')

    for dataset, dataset_df in dataset_groups:
        # Create a new figure for each dataset
        plt.figure(figsize=(12, 6))
        
        # Group the data by unique combinations of Dataset, Encoder type, Encoder block, and LoRA rank
        groups = dataset_df.groupby(['Encoder type', 'Encoder block', 'LoRA rank', 'Prompt type', 'Setting', 'b+p'])    
        
        # Generate distinct colors for this dataset's groups
        n_colors = len(groups)
        color_palette = generate_distinct_colors(n_colors)
        
        # Plot each group with a unique color
        for (name, group), color in zip(groups, color_palette):
            label = create_label(name)
            group_sorted = group.sort_values('Iteration')
            #plt.plot(x_values, group_sorted['mAP'], marker='o', linestyle='-', label=label, color=color)
        
            plt.plot(group_sorted['Iteration'], group_sorted['mAP'], marker='o', linestyle='-', label=label, color=color)
    
        # Customize the plot
        title = f'AIS+EffSAM mAP trained on {dataset} Dataset eval on COCOA'
        title = f'(class agnostic) {title}'if noclass else title
        plt.xlabel('Iteration')
        plt.ylabel('mAP')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Adjust layout to prevent cutting off labels
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'{fig_name_prefix}_{dataset}.png', dpi=300, bbox_inches='tight')
        
        # Close the figure to free up memory
        plt.close()

    print("All plots have been saved.")

if __name__ == "__main__":
    draw_graph(
        excel_path='/home/u6693411/ais/AISFormer/final_result_iou_b+p.xlsx', 
        fig_name_prefix='ais+effsam_mAP_b+p', 
        versions_to_plot=["full 5 - random_random maxnum20", "full 5 - random_random -", "full 5 - random_random noboxaug"], 
        noclass=True
    )