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

# Read the Excel file
df = pd.read_excel('/home/u6693411/ais/AISFormer/final_result_iou.xlsx')


df = df[df['Iteration'] != 'latest']

# Group the data by Dataset
dataset_groups = df.groupby('Dataset')

for dataset, dataset_df in dataset_groups:
    # Create a new figure for each dataset
    plt.figure(figsize=(12, 6))
    
    # Group the data by unique combinations of Dataset, Encoder type, Encoder block, and LoRA rank
    groups = dataset_df.groupby(['Encoder type', 'Encoder block', 'LoRA rank', 'Prompt type', 'Setting'])    
    
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
    plt.xlabel('Iteration')
    plt.ylabel('mAP')
    plt.title(f'AIS+EffSAM mAP trained on {dataset} Dataset eval on COCOA')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'ais+cocoa_mAP_{dataset}.png', dpi=300, bbox_inches='tight')
    
    # Close the figure to free up memory
    plt.close()

print("All plots have been saved.")