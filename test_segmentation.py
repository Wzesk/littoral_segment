#!/usr/bin/env python3
"""Test YOLO26-seg model with unified weights on images from different sites.

Picks 5 normalized images from diverse geotools_sites (including non-periodic
coastlines) and runs YOLO26 segmentation, saving originals + masks side-by-side.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from seg_models.yolo26_seg import YOLO26Seg

GEOTOOLS = "/home/walter_littor_al/geotools_sites"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "test_results")

# 5 test images from different sites:
#   - Fenfushi, anhenunfushi: periodic/island (Maldives atolls)
#   - KakinadaAndraPradesh: non-periodic (Indian river delta coast)
#   - Puducherry: non-periodic (straight Indian coastline)
#   - korakas: non-periodic (Greek rocky coast)
TEST_IMAGES = [
    ("Fenfushi", "20211023T051901_20211023T053305_T43NBF_nir_x4.png"),
    ("anhenunfushi", "20210111T053719_20210111T053715_T43NBF_nir_x4.png"),
    ("KakinadaAndraPradesh", "20190312T045651_nir_x4.png"),
    ("Puducherry", "20240204T050011_nir_x4.png"),
    ("korakas", "20180814T090549_nir_x4.png"),
]


def create_comparison(img, mask, site_name, filename):
    """Create a side-by-side comparison image: original | mask | overlay."""
    w, h = img.size
    # Create overlay: original with red mask boundary
    overlay = img.convert("RGB").copy()
    mask_arr = np.array(mask)
    # Find mask edges via simple dilation diff
    from scipy.ndimage import binary_dilation
    edges = binary_dilation(mask_arr > 127, iterations=2) & ~(mask_arr > 127)
    overlay_arr = np.array(overlay)
    overlay_arr[edges, 0] = 255  # Red channel
    overlay_arr[edges, 1] = 0
    overlay_arr[edges, 2] = 0
    overlay = Image.fromarray(overlay_arr)

    # Side-by-side: original | mask | overlay
    canvas = Image.new("RGB", (w * 3 + 4, h + 20), (255, 255, 255))
    canvas.paste(img.convert("RGB"), (0, 20))
    canvas.paste(mask.convert("RGB"), (w + 2, 20))
    canvas.paste(overlay, (w * 2 + 4, 20))

    # Add title
    draw = ImageDraw.Draw(canvas)
    title = f"{site_name}: {filename}"
    draw.text((4, 2), title, fill=(0, 0, 0))

    return canvas


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model with unified weights
    params_dir = os.path.join(os.path.dirname(__file__), "seg_models", "yolo26_params")
    model = YOLO26Seg(folder=params_dir)
    # Force best.pt (unified fine-tuned weights) instead of default alphabetic pick
    model.weights_path = os.path.join(params_dir, "best.pt")
    model._model = None  # Reset so it reloads with correct weights
    print(f"Model loaded from: {model.weights_path}")
    print(f"Results will be saved to: {RESULTS_DIR}\n")

    for site_name, filename in TEST_IMAGES:
        img_path = os.path.join(GEOTOOLS, site_name, "NORMALIZED", filename)
        if not os.path.exists(img_path):
            print(f"  SKIP: {img_path} not found")
            continue

        img = Image.open(img_path)
        print(f"[{site_name}] {filename} ({img.size[0]}x{img.size[1]})")

        # Run segmentation (no contrast enhancement, no padding)
        mask = model.mask_from_img(img, padding=0, contrast=1.0)

        # Save individual results
        stem = f"{site_name}_{os.path.splitext(filename)[0]}"
        img.save(os.path.join(RESULTS_DIR, f"{stem}_original.png"))
        mask.save(os.path.join(RESULTS_DIR, f"{stem}_mask.png"))

        # Save comparison
        comparison = create_comparison(img, mask, site_name, filename)
        comparison.save(os.path.join(RESULTS_DIR, f"{stem}_comparison.png"))

        # Print mask stats
        mask_arr = np.array(mask)
        land_pct = (mask_arr > 127).sum() / mask_arr.size * 100
        print(f"  → land: {land_pct:.1f}%, water: {100 - land_pct:.1f}%")

    print(f"\nDone! Results saved to {RESULTS_DIR}/")
    print(f"Files: {len(os.listdir(RESULTS_DIR))}")


if __name__ == "__main__":
    main()

