"""
Redraw the method overview diagram (Figure 1) with color style aligned to Figure 2.
Color palette extracted from Figure 2:
  - Soft blue (Backbone region): #C8DCF0 / #D6E4F0
  - Soft green (Head region): #D8EFDB / #C8E6C9
  - Soft pink/rose: #F8E0E6 / #FADCE5
  - Soft purple/lavender: #E8D5F0 / #E1BEE7
  - Soft yellow/gold: #FFF3C4 / #FFF9C4
  - Soft orange: #FFE0B2
  - Deep text color: #333333
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm
import numpy as np

# ============ Font Setup (Noto Sans CJK) ============
CJK_FONT_PATH = '/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc'
CJK_FONT_BOLD_PATH = '/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc'
fm.fontManager.addfont(CJK_FONT_PATH)
fm.fontManager.addfont(CJK_FONT_BOLD_PATH)
CJK_FONT = fm.FontProperties(fname=CJK_FONT_PATH)
CJK_FONT_BOLD = fm.FontProperties(fname=CJK_FONT_BOLD_PATH)
CJK_FAMILY = CJK_FONT.get_name()
plt.rcParams['font.family'] = CJK_FAMILY
plt.rcParams['axes.unicode_minus'] = False

# ============ Color Palette (from Figure 2) ============
# Background regions
BG_BLUE = '#D6E4F0'       # like Backbone region
BG_GREEN = '#D8EFDB'      # like Head region
BG_PINK = '#FADCE5'       # like bottom modules

# Box fill colors
BOX_BLUE = '#BBDEFB'      # soft blue (like C2f blocks)
BOX_GREEN = '#C8E6C9'     # soft green (like ConvModule)
BOX_PURPLE = '#E1BEE7'    # soft purple/lavender (like RCBAM)
BOX_YELLOW = '#FFF9C4'    # soft yellow (like Concat)
BOX_ORANGE = '#FFE0B2'    # soft orange (like Upsample)
BOX_PINK = '#F8BBD0'      # soft pink
BOX_LIGHT_BLUE = '#E3F2FD'  # very light blue

# Border colors (slightly darker)
BORDER_BLUE = '#64B5F6'
BORDER_GREEN = '#81C784'
BORDER_PURPLE = '#BA68C8'
BORDER_YELLOW = '#FFD54F'
BORDER_ORANGE = '#FFB74D'
BORDER_PINK = '#F06292'

# Text
TEXT_DARK = '#333333'
TEXT_RED = '#D32F2F'
TEXT_GREEN_DARK = '#2E7D32'
ARROW_COLOR = '#555555'

# ============ Figure Setup ============
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')

# ============ Helper Functions ============
def draw_rounded_box(ax, x, y, w, h, facecolor, edgecolor, linewidth=1.5, alpha=1.0, zorder=2):
    """Draw a rounded rectangle."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_region_box(ax, x, y, w, h, facecolor, edgecolor, linewidth=1.5, alpha=0.4, zorder=0):
    """Draw a large background region box."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.2",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, lw=1.8, style='->', zorder=3):
    """Draw an arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=zorder)

def draw_circle(ax, x, y, radius=0.15, color='red', zorder=4):
    """Draw a filled circle (indicator dot)."""
    circle = plt.Circle((x, y), radius, color=color, zorder=zorder)
    ax.add_patch(circle)

def text_center(ax, x, y, text, fontsize=11, color=TEXT_DARK, weight='normal', zorder=5):
    """Draw centered text."""
    fp = CJK_FONT_BOLD if weight == 'bold' else CJK_FONT
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, weight=weight, zorder=zorder,
            fontproperties=fp)

# ============ Title Labels ============
# "三大挑战" title
draw_region_box(ax, 0.3, 0.5, 4.5, 8.2, BG_BLUE, BORDER_BLUE, linewidth=2, alpha=0.3)
text_center(ax, 2.55, 8.35, '三大挑战', fontsize=16, weight='bold', color='#1565C0')

# "三层协同改进" title
draw_region_box(ax, 5.8, 0.5, 5.2, 8.2, BG_GREEN, BORDER_GREEN, linewidth=2, alpha=0.3)
text_center(ax, 8.4, 8.35, '三层协同改进', fontsize=16, weight='bold', color=TEXT_GREEN_DARK)

# ============ Challenge Boxes (Left Side) ============
# Challenge 2: Image quality degradation (top)
ch2_x, ch2_y, ch2_w, ch2_h = 0.7, 6.0, 3.7, 1.8
draw_rounded_box(ax, ch2_x, ch2_y, ch2_w, ch2_h, BOX_YELLOW, BORDER_YELLOW, linewidth=2)
draw_circle(ax, ch2_x + 0.4, ch2_y + ch2_h - 0.35, 0.18, '#EF5350')
text_center(ax, ch2_x + ch2_w/2 + 0.15, ch2_y + ch2_h - 0.35, '挑战2', fontsize=12, weight='bold')
text_center(ax, ch2_x + ch2_w/2, ch2_y + ch2_h/2 - 0.1, '图像质量退化', fontsize=12, weight='bold')
text_center(ax, ch2_x + ch2_w/2, ch2_y + 0.35, '高斯噪声/运动模糊', fontsize=10)

# Challenge 1: Crack visual characteristics (middle)
ch1_x, ch1_y, ch1_w, ch1_h = 0.7, 3.4, 3.7, 2.0
draw_rounded_box(ax, ch1_x, ch1_y, ch1_w, ch1_h, BOX_YELLOW, BORDER_YELLOW, linewidth=2)
draw_circle(ax, ch1_x + 0.4, ch1_y + ch1_h - 0.35, 0.18, '#EF5350')
text_center(ax, ch1_x + ch1_w/2 + 0.15, ch1_y + ch1_h - 0.35, '挑战1', fontsize=12, weight='bold')
text_center(ax, ch1_x + ch1_w/2, ch1_y + ch1_h/2 - 0.05, '裂缝视觉特性极端', fontsize=12, weight='bold')
text_center(ax, ch1_x + ch1_w/2, ch1_y + 0.6, '细长/低对比度/像素占比', fontsize=10)
text_center(ax, ch1_x + ch1_w/2, ch1_y + 0.25, '<5%', fontsize=10)

# Challenge 3: Limited annotation (bottom)
ch3_x, ch3_y, ch3_w, ch3_h = 0.7, 1.0, 3.7, 1.8
draw_rounded_box(ax, ch3_x, ch3_y, ch3_w, ch3_h, BOX_YELLOW, BORDER_YELLOW, linewidth=2)
draw_circle(ax, ch3_x + 0.4, ch3_y + ch3_h - 0.35, 0.18, '#EF5350')
text_center(ax, ch3_x + ch3_w/2 + 0.15, ch3_y + ch3_h - 0.35, '挑战3', fontsize=12, weight='bold')
text_center(ax, ch3_x + ch3_w/2, ch3_y + ch3_h/2 - 0.1, '有限标注下', fontsize=12, weight='bold')
text_center(ax, ch3_x + ch3_w/2, ch3_y + 0.35, '过拟合与不稳定性', fontsize=10)

# ============ Solution Boxes (Middle) ============
# Tech 1: Data layer - Blur/Noise augmentation (top)
t1_x, t1_y, t1_w, t1_h = 6.3, 6.2, 4.2, 1.5
draw_rounded_box(ax, t1_x, t1_y, t1_w, t1_h, BOX_PURPLE, BORDER_PURPLE, linewidth=2)
draw_circle(ax, t1_x + 0.35, t1_y + t1_h - 0.35, 0.18, '#4CAF50')
text_center(ax, t1_x + t1_w/2 + 0.1, t1_y + t1_h - 0.35, '技术1·数据层', fontsize=12, weight='bold')
text_center(ax, t1_x + t1_w/2, t1_y + 0.4, '模糊加噪数据增强', fontsize=12, weight='bold')

# Tech 2: Feature layer - RCBAM attention (middle)
t2_x, t2_y, t2_w, t2_h = 6.3, 3.8, 4.2, 1.5
draw_rounded_box(ax, t2_x, t2_y, t2_w, t2_h, BOX_PURPLE, BORDER_PURPLE, linewidth=2)
draw_circle(ax, t2_x + 0.35, t2_y + t2_h - 0.35, 0.18, '#4CAF50')
text_center(ax, t2_x + t2_w/2 + 0.1, t2_y + t2_h - 0.35, '技术2·特征层', fontsize=12, weight='bold')
text_center(ax, t2_x + t2_w/2, t2_y + 0.4, 'RCBAM 注意力', fontsize=12, weight='bold')

# Tech 3: Optimization layer - Multi-View consistency loss (bottom)
t3_x, t3_y, t3_w, t3_h = 6.3, 1.3, 4.2, 1.5
draw_rounded_box(ax, t3_x, t3_y, t3_w, t3_h, BOX_PURPLE, BORDER_PURPLE, linewidth=2)
draw_circle(ax, t3_x + 0.35, t3_y + t3_h - 0.35, 0.18, '#4CAF50')
text_center(ax, t3_x + t3_w/2 + 0.1, t3_y + t3_h - 0.35, '技术3·优化层', fontsize=12, weight='bold')
text_center(ax, t3_x + t3_w/2, t3_y + 0.4, 'Multi-View 一致性损失', fontsize=12, weight='bold')

# ============ Output Box (Right Side) ============
out_x, out_y, out_w, out_h = 12.0, 3.5, 3.5, 2.2
draw_rounded_box(ax, out_x, out_y, out_w, out_h, BOX_ORANGE, BORDER_ORANGE, linewidth=2.5)
# YOLOv8 icon (small circle)
draw_circle(ax, out_x + 0.5, out_y + out_h/2 + 0.1, 0.22, '#EF5350')
text_center(ax, out_x + out_w/2 + 0.15, out_y + out_h/2 + 0.1, '改进的 YOLOv8', fontsize=13, weight='bold')
text_center(ax, out_x + out_w/2 + 0.15, out_y + out_h/2 - 0.4, '裂缝检测模型', fontsize=13, weight='bold')

# ============ Arrows: Challenges -> Solutions ============
# Arrow labels (between challenges and solutions)
arrow_label_x = 5.3

# Challenge 2 -> Tech 1
mid_y_top = ch2_y + ch2_h/2
draw_arrow(ax, ch2_x + ch2_w, mid_y_top, t1_x, t1_y + t1_h/2, color=ARROW_COLOR, lw=2)
text_center(ax, arrow_label_x, mid_y_top + 0.25, '缩小域偏移', fontsize=10, color=TEXT_RED, weight='bold')

# Challenge 1 -> Tech 2
mid_y_mid = ch1_y + ch1_h/2
draw_arrow(ax, ch1_x + ch1_w, mid_y_mid, t2_x, t2_y + t2_h/2, color=ARROW_COLOR, lw=2)
text_center(ax, arrow_label_x, mid_y_mid + 0.25, '增强判别特征', fontsize=10, color=TEXT_RED, weight='bold')

# Challenge 3 -> Tech 3
mid_y_bot = ch3_y + ch3_h/2
draw_arrow(ax, ch3_x + ch3_w, mid_y_bot, t3_x, t3_y + t3_h/2, color=ARROW_COLOR, lw=2)
text_center(ax, arrow_label_x, mid_y_bot + 0.25, '自监督正则化', fontsize=10, color=TEXT_RED, weight='bold')

# ============ Arrows: Solutions -> Output ============
# Tech 1 -> Output (curve up then right)
draw_arrow(ax, t1_x + t1_w, t1_y + t1_h/2, out_x, out_y + out_h - 0.3, color=ARROW_COLOR, lw=2)

# Tech 2 -> Output (straight right)
draw_arrow(ax, t2_x + t2_w, t2_y + t2_h/2, out_x, out_y + out_h/2, color=ARROW_COLOR, lw=2)

# Tech 3 -> Output (curve down then right)
draw_arrow(ax, t3_x + t3_w, t3_y + t3_h/2, out_x, out_y + 0.3, color=ARROW_COLOR, lw=2)

# ============ Save ============
plt.tight_layout()
plt.savefig('method_overview_v2.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('method_overview_v2.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Figure saved as method_overview_v2.png and method_overview_v2.pdf")
