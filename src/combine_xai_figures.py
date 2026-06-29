"""
Combines existing XAI PNG images into 2x2 grid figures using PIL.
No model loading, no SHAP/LIME/ELI5 recomputation — pure image stitching.

Output:
  models/xai_combined/fig7_shap_combined.png
  models/xai_combined/fig8_lime_combined.png
  models/xai_combined/fig9_eli5_combined.png
"""

import os
from PIL import Image, ImageDraw, ImageFont

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XAI_DIR    = os.path.join(BASE_DIR, 'models', 'xai')
OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'xai_combined')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CELL_W  = 800
CELL_H  = 550
LABEL_H = 28
MARGIN  = 10

LABELS = [
    '(a) NIFTY - Next-Day',
    '(b) NIFTY - Next-Week',
    '(c) BANKNIFTY - Next-Day',
    '(d) BANKNIFTY - Next-Week',
]

GRIDS = [
    {
        'name':   'fig7_shap_combined.png',
        'title':  'SHAP (XGBoost) Feature Importance',
        'paths': [
            os.path.join(XAI_DIR, 'xai_nifty_h1',     'shap_xgb_nifty_h1_bar.png'),
            os.path.join(XAI_DIR, 'xai_nifty_h5',     'shap_xgb_nifty_h5_bar.png'),
            os.path.join(XAI_DIR, 'xai_banknifty_h1', 'shap_xgb_banknifty_h1_bar.png'),
            os.path.join(XAI_DIR, 'xai_banknifty_h5', 'shap_xgb_banknifty_h5_bar.png'),
        ],
    },
    {
        'name':   'fig8_lime_combined.png',
        'title':  'LIME Local Explanations',
        'paths': [
            os.path.join(XAI_DIR, 'xai_nifty_h1',     'lime_nifty_h1.png'),
            os.path.join(XAI_DIR, 'xai_nifty_h5',     'lime_nifty_h5.png'),
            os.path.join(XAI_DIR, 'xai_banknifty_h1', 'lime_banknifty_h1.png'),
            os.path.join(XAI_DIR, 'xai_banknifty_h5', 'lime_banknifty_h5.png'),
        ],
    },
    {
        'name':   'fig9_eli5_combined.png',
        'title':  'ELI5 Feature Weights',
        'paths': [
            os.path.join(XAI_DIR, 'xai_nifty_h1',     'eli5_nifty_h1.png'),
            os.path.join(XAI_DIR, 'xai_nifty_h5',     'eli5_nifty_h5.png'),
            os.path.join(XAI_DIR, 'xai_banknifty_h1', 'eli5_banknifty_h1.png'),
            os.path.join(XAI_DIR, 'xai_banknifty_h5', 'eli5_banknifty_h5.png'),
        ],
    },
]


def load_font(size):
    for name in ('arialbd.ttf', 'arial.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def fit_image(img, cell_w, cell_h):
    """Resize image to fit within cell, preserving aspect ratio, white-pad to exact cell size."""
    img = img.convert('RGBA')
    img.thumbnail((cell_w, cell_h), Image.LANCZOS)
    canvas = Image.new('RGBA', (cell_w, cell_h), (255, 255, 255, 255))
    x_off  = (cell_w - img.width)  // 2
    y_off  = (cell_h - img.height) // 2
    canvas.paste(img, (x_off, y_off), img)
    return canvas.convert('RGB')


def make_grid(paths, labels, cell_w, cell_h, label_h, margin):
    """Build a 2x2 grid canvas and paste the 4 cells with labels."""
    font       = load_font(13)
    panel_h    = label_h + cell_h
    canvas_w   = 2 * cell_w  + 3 * margin
    canvas_h   = 2 * panel_h + 3 * margin
    canvas     = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw       = ImageDraw.Draw(canvas)

    positions = [
        (margin,              margin),               # top-left  → paths[0]
        (2 * margin + cell_w, margin),               # top-right → paths[1]
        (margin,              2 * margin + panel_h), # bot-left  → paths[2]
        (2 * margin + cell_w, 2 * margin + panel_h), # bot-right → paths[3]
    ]

    for idx, (path, label, (px, py)) in enumerate(zip(paths, labels, positions)):
        img  = Image.open(path)
        cell = fit_image(img, cell_w, cell_h)

        # Label above cell
        draw.rectangle([px, py, px + cell_w, py + label_h - 2], fill=(240, 240, 240))
        draw.text((px + 6, py + 5), label, fill=(30, 30, 30), font=font)

        # Paste cell image below label
        canvas.paste(cell, (px, py + label_h))

    return canvas


if __name__ == '__main__':
    for grid in GRIDS:
        print(f"Building {grid['name']} ...")
        for p in grid['paths']:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Source image not found: {p}")

        canvas  = make_grid(grid['paths'], LABELS, CELL_W, CELL_H, LABEL_H, MARGIN)
        out_path = os.path.join(OUTPUT_DIR, grid['name'])
        canvas.save(out_path, dpi=(150, 150))
        print(f"  Saved -> {out_path}  ({canvas.width}x{canvas.height} px)")

    print("\nDone. 3 combined figures written to:")
    print(f"  {OUTPUT_DIR}")
