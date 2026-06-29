"""
Combines 4 existing SHAP beeswarm PNGs into a single 2x2 grid figure using PIL.
No model loading, no SHAP recomputation — pure image stitching.

Layout:
  top-left    = NIFTY h=1      (a)
  top-right   = NIFTY h=5      (b)
  bottom-left = BANKNIFTY h=1  (c)
  bottom-right= BANKNIFTY h=5  (d)

Output: models/xai_combined/fig7_shap_beeswarm_combined.png
"""

import os
from PIL import Image, ImageDraw, ImageFont

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XAI_DIR    = os.path.join(BASE_DIR, 'models', 'xai')
OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'xai_combined')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCE_PATHS = [
    os.path.join(XAI_DIR, 'xai_nifty_h1',     'shap_xgb_nifty_h1_beeswarm.png'),
    os.path.join(XAI_DIR, 'xai_nifty_h5',     'shap_xgb_nifty_h5_beeswarm.png'),
    os.path.join(XAI_DIR, 'xai_banknifty_h1', 'shap_xgb_banknifty_h1_beeswarm.png'),
    os.path.join(XAI_DIR, 'xai_banknifty_h5', 'shap_xgb_banknifty_h5_beeswarm.png'),
]

LABELS = [
    '(a) NIFTY — Next-Day',
    '(b) NIFTY — Next-Week',
    '(c) BANKNIFTY — Next-Day',
    '(d) BANKNIFTY — Next-Week',
]

MARGIN  = 10
LABEL_H = 30
OUTPUT  = os.path.join(OUTPUT_DIR, 'fig7_shap_beeswarm_combined.png')


def load_font(size):
    for name in ('arialbd.ttf', 'arial.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def fit_to_cell(img, cell_w, cell_h):
    """Scale image to fit within cell preserving aspect ratio; white-pad to exact cell size."""
    img = img.convert('RGBA')
    img.thumbnail((cell_w, cell_h), Image.LANCZOS)
    cell = Image.new('RGBA', (cell_w, cell_h), (255, 255, 255, 255))
    x_off = (cell_w - img.width)  // 2
    y_off = (cell_h - img.height) // 2
    cell.paste(img, (x_off, y_off), img)
    return cell.convert('RGB')


if __name__ == '__main__':
    # Verify all source images exist
    for p in SOURCE_PATHS:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Source image not found: {p}')

    # Open all images and determine cell size from the largest dimensions
    images   = [Image.open(p) for p in SOURCE_PATHS]
    cell_w   = max(img.width  for img in images)
    cell_h   = max(img.height for img in images)
    print(f'Cell size (largest source): {cell_w}x{cell_h} px')

    # Canvas dimensions
    panel_h  = LABEL_H + cell_h
    canvas_w = 2 * cell_w  + 3 * MARGIN
    canvas_h = 2 * panel_h + 3 * MARGIN
    canvas   = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw     = ImageDraw.Draw(canvas)
    font     = load_font(14)

    # top-left, top-right, bottom-left, bottom-right
    positions = [
        (MARGIN,              MARGIN),
        (2 * MARGIN + cell_w, MARGIN),
        (MARGIN,              2 * MARGIN + panel_h),
        (2 * MARGIN + cell_w, 2 * MARGIN + panel_h),
    ]

    for img, label, (px, py) in zip(images, LABELS, positions):
        cell = fit_to_cell(img, cell_w, cell_h)

        # Label strip
        draw.rectangle([px, py, px + cell_w, py + LABEL_H - 2], fill=(235, 235, 235))
        draw.text((px + 8, py + 6), label, fill=(20, 20, 20), font=font)

        # Cell image
        canvas.paste(cell, (px, py + LABEL_H))

    canvas.save(OUTPUT, dpi=(150, 150))
    print(f'Saved -> {OUTPUT}')
    print(f'Canvas size: {canvas.width}x{canvas.height} px')
