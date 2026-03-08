#!/usr/bin/env python3
"""
Generate motivation diagram for the YOLOv8-Crack-Detection project.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#FEFEFE')

# ========== Helper function: draw rounded rectangle with text ==========
def draw_box(ax, x, y, w, h, texts, facecolor, edgecolor, fontsize=11, bold_lines=None):
    """Draw a rounded rectangle with multi-line text."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=facecolor,
                         edgecolor=edgecolor,
                         linewidth=2,
                         zorder=2)
    ax.add_patch(box)
    
    if bold_lines is None:
        bold_lines = []
    
    total = len(texts)
    line_h = h / (total + 1)
    for i, text in enumerate(texts):
        ty = y + h - line_h * (i + 1)
        weight = 'bold' if i in bold_lines else 'normal'
        fs = fontsize + 1 if i in bold_lines else fontsize
        ax.text(x + w / 2, ty, text,
                ha='center', va='center',
                fontsize=fs, fontweight=weight,
                color='#333333', zorder=3,
                fontfamily='DejaVu Sans')

def draw_arrow(ax, x1, y1, x2, y2, color='#555555', style='->', linestyle='-', lw=1.8):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style,
                            color=color,
                            linewidth=lw,
                            linestyle=linestyle,
                            mutation_scale=15,
                            zorder=1)
    ax.add_patch(arrow)

# ========== Layout parameters ==========
box_w = 4.2
box_h_top = 1.2
box_h_mid = 1.8
box_h_bot = 1.0

# X positions for 3 columns
col_x = [0.7, 5.9, 11.1]

# Y positions for 3 rows
row_y_top = 8.0      # Challenge row
row_y_mid = 4.8      # Solution row
row_y_bot = 1.5      # Goal row

# ========== Title ==========
ax.text(8, 9.6, 'Motivation: YOLOv8-based Robust Crack Detection',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color='#1A237E', fontfamily='DejaVu Sans')

# ========== CHALLENGE LAYER ==========
# Challenge 1
draw_box(ax, col_x[0], row_y_top, box_w, box_h_top,
         ['Challenge 1', 'Thin cracks & low contrast'],
         facecolor='#FFF8E1', edgecolor='#D4A017',
         fontsize=10, bold_lines=[0])

# Challenge 2
draw_box(ax, col_x[1], row_y_top, box_w, box_h_top,
         ['Challenge 2', 'Noise & blur degradation'],
         facecolor='#FFF8E1', edgecolor='#D4A017',
         fontsize=10, bold_lines=[0])

# Challenge 3
draw_box(ax, col_x[2], row_y_top, box_w, box_h_top,
         ['Challenge 3', 'Limited labeled data'],
         facecolor='#FFF8E1', edgecolor='#D4A017',
         fontsize=10, bold_lines=[0])

# ========== SOLUTION LAYER ==========
# Solution 1
draw_box(ax, col_x[0], row_y_mid, box_w, box_h_mid,
         ['RCBAM', 'Attention Module', '', 'Channel + Spatial attention', 'Residual connection'],
         facecolor='#E8F5E9', edgecolor='#2E7D32',
         fontsize=9, bold_lines=[0, 1])

# Solution 2
draw_box(ax, col_x[1], row_y_mid, box_w, box_h_mid,
         ['Blur & Noise', 'Data Augmentation', '', 'Gaussian blur + noise', 'Mixed degradation set'],
         facecolor='#E8F5E9', edgecolor='#2E7D32',
         fontsize=9, bold_lines=[0, 1])

# Solution 3
draw_box(ax, col_x[2], row_y_mid, box_w, box_h_mid,
         ['Multi-View', 'Self-supervised Loss', '', 'Dual-view cosine distance', 'Feature consistency'],
         facecolor='#E8F5E9', edgecolor='#2E7D32',
         fontsize=9, bold_lines=[0, 1])

# ========== GOAL LAYER ==========
goal_w = 6.0
goal_x = (16 - goal_w) / 2
draw_box(ax, goal_x, row_y_bot, goal_w, box_h_bot,
         ['Robust Crack Detection Model'],
         facecolor='#FFCCBC', edgecolor='#E64A19',
         fontsize=13, bold_lines=[0])

# ========== ARROWS: Challenge -> Solution ==========
for i in range(3):
    cx = col_x[i] + box_w / 2
    draw_arrow(ax, cx, row_y_top, cx, row_y_mid + box_h_mid, color='#555555')

# ========== ARROWS: Solution -> Goal ==========
# Left solution
draw_arrow(ax, col_x[0] + box_w / 2, row_y_mid, goal_x + goal_w * 0.2, row_y_bot + box_h_bot, color='#555555')
# Middle solution
draw_arrow(ax, col_x[1] + box_w / 2, row_y_mid, goal_x + goal_w * 0.5, row_y_bot + box_h_bot, color='#555555')
# Right solution
draw_arrow(ax, col_x[2] + box_w / 2, row_y_mid, goal_x + goal_w * 0.8, row_y_bot + box_h_bot, color='#555555')

# ========== DASHED ARROWS: Inter-solution interaction ==========
# S1 --> S3 (enhanced feature)
draw_arrow(ax, col_x[0] + box_w, row_y_mid + box_h_mid * 0.6,
           col_x[2], row_y_mid + box_h_mid * 0.6,
           color='#7B1FA2', linestyle='--', lw=1.5)
ax.text((col_x[0] + box_w + col_x[2]) / 2, row_y_mid + box_h_mid * 0.6 + 0.2,
        'Enhanced feature representation',
        ha='center', va='center', fontsize=8, color='#7B1FA2',
        fontstyle='italic', fontfamily='DejaVu Sans')

# S2 --> S3 (regularization)
draw_arrow(ax, col_x[1] + box_w, row_y_mid + box_h_mid * 0.3,
           col_x[2], row_y_mid + box_h_mid * 0.3,
           color='#7B1FA2', linestyle='--', lw=1.5)
ax.text((col_x[1] + box_w + col_x[2]) / 2, row_y_mid + box_h_mid * 0.3 + 0.2,
        'Regularization complement',
        ha='center', va='center', fontsize=8, color='#7B1FA2',
        fontstyle='italic', fontfamily='DejaVu Sans')

# ========== Layer labels ==========
ax.text(0.15, row_y_top + box_h_top / 2, 'Challenges',
        ha='center', va='center', fontsize=9, fontweight='bold',
        color='#D4A017', rotation=90, fontfamily='DejaVu Sans')
ax.text(0.15, row_y_mid + box_h_mid / 2, 'Solutions',
        ha='center', va='center', fontsize=9, fontweight='bold',
        color='#2E7D32', rotation=90, fontfamily='DejaVu Sans')
ax.text(0.15, row_y_bot + box_h_bot / 2, 'Goal',
        ha='center', va='center', fontsize=9, fontweight='bold',
        color='#E64A19', rotation=90, fontfamily='DejaVu Sans')

plt.tight_layout()
output_path = '/apdcephfs_fsgm/share_304156246/xmudongwang/codebase/zq/YOLOv8-Crack-Detection/motivation_diagram.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#FEFEFE')
print(f'Diagram saved to: {output_path}')
plt.close()
