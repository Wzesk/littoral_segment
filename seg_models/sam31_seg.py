from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu

# Default model: facebook/sam3 is the SAM 3.x release with full HuggingFace
# Transformers integration (Sam3Processor + Sam3Model).
# facebook/sam3.1 provides later checkpoints but has no Transformers bindings;
# to use those, download them from HuggingFace Hub and load via the Meta
# GitHub repo (github.com/facebookresearch/sam3), then set model_id to the
# local checkpoint path when those bindings become available.
DEFAULT_MODEL_ID = "facebook/sam3"

# PCS concept string passed to SAM 3's Promptable Concept Segmentation head.
# The model uses this text to detect all matching object instances across the image.
# Using 'landmass' instead of 'shoreline' to force the model to find the generalized
# convex outer boundary instead of tracing every inlet.
_PCS_CONCEPT = "landmass"


class SAM31Seg:
    """SAM 3 land/water segmentation using Promptable Concept Segmentation (PCS).

    Uses the HuggingFace ``Sam3Model`` + ``Sam3Processor`` API.  Key upgrades
    over SAM2:

    * **PCS text concept** — a text concept string is passed as the prompt,
      letting the model use semantic grounding to detect all coastal land
      instances simultaneously rather than relying solely on point prompts.
    * **Box hint from Otsu** — the bounding box of the largest Otsu-derived
      land component is provided as a positive visual prompt alongside the
      text concept, anchoring the detector to the correct spatial region.
    * **Temporal memory bank** — past-frame masks are stored and used to
      refine the Otsu-derived box hint on each subsequent frame, steering
      the model toward previously detected land extent and exploiting
      coastline continuity across Sentinel-2 stacks.
    * **Semantic segmentation output** — the model's ``semantic_seg`` output
      (merged binary land mask, shape ``[B, 1, H, W]``) is used as the
      primary prediction, avoiding the need to manually merge per-instance
      ``pred_masks``.

    Interface is identical to SAM2Seg so it can be swapped in without
    changes to calling code.

    Args:
        model_id: HuggingFace model identifier (default ``facebook/sam3``).
            After fine-tuning with ``seg-training/finetune_sam31.py``, set
            this to the output checkpoint path.
        use_temporal_memory: Maintain a rolling mask bank across the folder
            timeseries to refine box hints (default True).
        max_memory_frames: Maximum frames retained in the memory bank.
        text_concept: PCS concept string.  Change to e.g.
            ``"reef flat boundary"`` for atoll-specific deployments.
        inlet_span_meters: If provided, spans inlets up to this width by applying
            a morphological closing to the mask output. Assuming 10m/px resolution
            for Sentinel-2, kernel size is `inlet_span_meters / 10`.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        use_temporal_memory: bool = True,
        max_memory_frames: int = 5,
        text_concept: str = _PCS_CONCEPT,
        inlet_span_meters: Optional[int] = 50,
        prompt_boxes_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.use_temporal_memory = use_temporal_memory
        self.max_memory_frames = max_memory_frames
        self.text_concept = text_concept
        self.inlet_span_meters = inlet_span_meters
        # Directory containing AE bootstrap output; subdir per year holds
        # prompt_boxes.geojson with bbox_norm entries for SAM box hints.
        self.prompt_boxes_dir = prompt_boxes_dir

        self._model = None
        self._processor = None
        self._device: Optional[str] = None
        self._current_ae_box: Optional[Tuple[int, int, int, int]] = None

        # Rolling memory bank: each entry is a dict with key 'mask' (numpy
        # uint8 array, land=255) from a previous frame.
        self._memory_bank: List[dict] = []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return self._model

        from transformers import Sam3Processor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Sam3Processor.from_pretrained(self.model_id)

        # Prefer Sam3LiteTextModel (available in transformers≥5.6.0.dev0) for
        # the yonigozlan/sam3-litetext-s0 and any fine-tuned checkpoints derived
        # from it.  Fall back to Sam3Model for facebook/sam3 checkpoints.
        try:
            from transformers import Sam3LiteTextModel
            self._model = Sam3LiteTextModel.from_pretrained(self.model_id)
        except Exception:
            from transformers import Sam3Model
            self._model = Sam3Model.from_pretrained(self.model_id)

        self._model = self._model.to(self._device).eval()
        return self._model

    # ------------------------------------------------------------------
    # Temporal memory management
    # ------------------------------------------------------------------

    def reset_memory(self):
        """Clear the temporal memory bank.

        Call between unrelated sites or at the start of a new timeseries
        so that memory from a previous site does not bleed in.
        """
        self._memory_bank = []

    def _update_memory(self, mask_arr: np.ndarray):
        """Append the current frame's mask to the memory bank (FIFO eviction)."""
        self._memory_bank.append({"mask": mask_arr.copy()})
        if len(self._memory_bank) > self.max_memory_frames:
            self._memory_bank.pop(0)

    # ------------------------------------------------------------------
    # Otsu-guided prompt generation
    # ------------------------------------------------------------------

    def _otsu_land_mask(self, gray: np.ndarray, periodic: bool) -> np.ndarray:
        """Return a binary land mask via Otsu thresholding."""
        otsu = threshold_otsu(gray)
        land = gray > otsu
        flip_threshold = 0.5 if periodic else 0.8
        if land.sum() / land.size > flip_threshold:
            land = ~land
        return land

    def _land_bbox(self, land_mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounding box of the largest land component."""
        labeled, n = ndimage.label(land_mask)
        if n == 0:
            h, w = land_mask.shape
            return (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
        sizes = ndimage.sum(land_mask, labeled, range(1, n + 1))
        largest = labeled == (int(np.argmax(sizes)) + 1)
        rows = np.where(largest.any(axis=1))[0]
        cols = np.where(largest.any(axis=0))[0]
        # Add a small margin so the box is not pixel-tight.
        h, w = land_mask.shape
        y1 = max(0,     int(rows.min()) - 2)
        y2 = min(h - 1, int(rows.max()) + 2)
        x1 = max(0,     int(cols.min()) - 2)
        x2 = min(w - 1, int(cols.max()) + 2)
        return (x1, y1, x2, y2)

    def _load_ae_box(
        self, year: int, img_w: int, img_h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return AE-derived pixel-space box for this year, or None if unavailable."""
        if not self.prompt_boxes_dir:
            return None
        boxes_path = os.path.join(self.prompt_boxes_dir, str(year), "prompt_boxes.geojson")
        if not os.path.exists(boxes_path):
            return None
        try:
            import json
            with open(boxes_path) as fh:
                gj = json.load(fh)
            features = gj.get("features", [])
            # Prefer the largest land component; fall back to first feature
            box_feat = next(
                (f for f in features if f["properties"].get("is_largest")),
                features[0] if features else None,
            )
            if box_feat is None:
                return None
            x1n, y1n, x2n, y2n = box_feat["properties"]["bbox_norm"]
            return (
                int(x1n * img_w), int(y1n * img_h),
                int(x2n * img_w), int(y2n * img_h),
            )
        except Exception:
            return None

    def _memory_guided_bbox(
        self,
        gray: np.ndarray,
        periodic: bool,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return a box hint for SAM, in priority order:
        1. AE-derived box (set by mask_from_folder per image) — most accurate
        2. Blended Otsu + temporal memory box
        3. Pure Otsu box
        """
        # AE prompt box takes priority over all other sources
        if self._current_ae_box is not None:
            return self._current_ae_box

        otsu_land = self._otsu_land_mask(gray, periodic)
        otsu_box  = self._land_bbox(otsu_land)

        if not self.use_temporal_memory or not self._memory_bank:
            return otsu_box

        prev_mask = self._memory_bank[-1]["mask"]
        h, w = gray.shape
        prev_resized = np.array(
            Image.fromarray(prev_mask).resize((w, h), Image.NEAREST)
        )
        prev_land = prev_resized > 127
        if not prev_land.any():
            return otsu_box

        prev_box = self._land_bbox(prev_land)

        # Average the two boxes — smooth transitions between frames.
        x1 = (otsu_box[0] + prev_box[0]) // 2
        y1 = (otsu_box[1] + prev_box[1]) // 2
        x2 = (otsu_box[2] + prev_box[2]) // 2
        y2 = (otsu_box[3] + prev_box[3]) // 2
        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Core segmentation
    # ------------------------------------------------------------------

    def mask_from_img(self, pil_img: Image.Image, periodic: bool = True) -> Image.Image:
        """Generate a binary land mask from a single PIL Image.

        Uses SAM 3's Promptable Concept Segmentation with:
          - Text prompt: self.text_concept (e.g. "coastal water-land boundary")
          - Box prompt:  bounding box of the largest Otsu land component,
                         optionally tempered by the temporal memory bank.

        The model's ``semantic_seg`` output is used as the primary prediction
        (it already merges all detected instances into a single binary channel).

        Falls back to raw Otsu threshold if the model returns no useful mask.

        Args:
            pil_img: Input PIL Image (grayscale or RGB/3-channel NIR composite).
            periodic: True for island/atoll scenes; False for mainland scenes.

        Returns a PIL Image (mode 'L') with land=255, water=0.

        Side effect: on success, appends the mask to the temporal memory bank.
        """
        arr = np.array(pil_img)
        gray = arr.mean(axis=2).astype(np.uint8) if len(arr.shape) == 3 else arr.astype(np.uint8)

        orig_size = pil_img.size  # (W, H)
        model = self._load_model()

        bbox = self._memory_guided_bbox(gray, periodic)
        x1, y1, x2, y2 = bbox

        # SAM3 box prompt: xyxy pixel coordinates, shape [batch, num_boxes, 4].
        inputs = self._processor(
            images=pil_img,
            text=self.text_concept,
            input_boxes=[[[x1, y1, x2, y2]]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Prefer semantic_seg (already-merged binary map, [B, 1, H, W]).
        # Fall back to merging instance pred_masks if semantic_seg unavailable.
        mask_data = None

        if hasattr(outputs, "semantic_seg") and outputs.semantic_seg is not None:
            seg = outputs.semantic_seg[0, 0].cpu().numpy()  # (H, W) logits or probs
            mask_data = (seg > 0).astype(np.uint8) * 255
        elif hasattr(outputs, "pred_masks") and outputs.pred_masks is not None:
            probs = torch.sigmoid(outputs.pred_masks[0])  # (num_queries, H, W)
            if probs.shape[0] > 0:
                # Merge all instance masks → single binary land map.
                merged = probs.max(dim=0).values.cpu().numpy()
                mask_data = (merged > 0.5).astype(np.uint8) * 255

        if mask_data is not None:
            if self.inlet_span_meters:
                import cv2
                # Assuming ~10m/px resolution.
                k_size = int(self.inlet_span_meters / 10)
                if k_size > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                    mask_data = cv2.morphologyEx(mask_data, cv2.MORPH_CLOSE, kernel)

            mask_img = Image.fromarray(mask_data).resize(orig_size, Image.NEAREST)
            # Sanity check: if the result is almost entirely one class, it may
            # have inverted land/water.  Correct by comparing with Otsu.
            mask_arr_check = np.array(mask_img)
            land_pct = (mask_arr_check > 127).sum() / mask_arr_check.size
            flip_threshold = 0.5 if periodic else 0.8
            if land_pct > flip_threshold:
                mask_img = Image.fromarray((~(mask_arr_check > 127)).astype(np.uint8) * 255)
            self._update_memory(np.array(mask_img))
            return mask_img

        # Fallback: Otsu threshold.
        flip_threshold = 0.5 if periodic else 0.8
        otsu = threshold_otsu(gray)
        land_mask = gray > otsu
        if land_mask.sum() / land_mask.size > flip_threshold:
            land_mask = ~land_mask
        mask_img = Image.fromarray(land_mask.astype(np.uint8) * 255)
        return mask_img.resize(orig_size, Image.NEAREST)

    # ------------------------------------------------------------------
    # Folder-level inference
    # ------------------------------------------------------------------

    def mask_from_folder(
        self,
        folder: str,
        periodic: bool = True,
        selection_config: Optional[dict] = None,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """Walk *folder* for upsampled NIR images and save land masks.

        Mirrors the SAM2Seg interface: looks for files containing ``_x``
        followed by a digit, writes masks to a sibling ``MASK/`` directory
        with the ``_mask.png`` suffix.

        Files are sorted alphabetically before processing.  Sentinel-2
        filenames embed the acquisition date (``YYYYMMDD``), so alphabetical
        order is chronological — enabling the temporal memory bank to exploit
        coastline continuity across the timeseries.

        The memory bank is reset at the start of each call.

        Returns a list of saved mask paths.
        """
        self.last_qc_records: List[dict] = []
        masks: List[str] = []

        candidates = []
        for root, _dirs, filenames in os.walk(folder):
            for filename in sorted(filenames):
                if "_x" in filename and filename.split("_x")[1][0].isdigit():
                    candidates.append(os.path.join(root, filename))
        candidates.sort()

        self.reset_memory()

        for file_path in candidates:
            filename = os.path.basename(file_path)
            img = Image.open(file_path)
            # Load AE box hint for the image's year (no-op if no AE data available)
            year_str = filename[:4]
            if year_str.isdigit() and self.prompt_boxes_dir:
                w, h = img.size
                self._current_ae_box = self._load_ae_box(int(year_str), w, h)
            else:
                self._current_ae_box = None
            mask = self.mask_from_img(img, periodic=periodic)

            stem = os.path.basename(file_path).split("_x")[0]
            # Strip spectral-channel suffixes so the mask is named by date only.
            for _sfx in ("_nns", "_rgb", "_nir"):
                if stem.endswith(_sfx):
                    stem = stem[: -len(_sfx)]
                    break
            mask_filename = stem + "_mask.png"
            if output_dir:
                mask_path = os.path.join(output_dir, mask_filename)
            else:
                site_dir = os.path.dirname(os.path.dirname(file_path))
                mask_path = os.path.join(site_dir, "MASK", mask_filename)

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            mask.save(mask_path)
            masks.append(mask_path)
            self.last_qc_records.append(
                {
                    "image_name": filename,
                    "mask_name": os.path.basename(mask_path),
                    "mask_path": mask_path,
                    "candidate_count": 1,
                    "selected_index": 0,
                    "periodic": bool(periodic),
                    "memory_frames_used": len(self._memory_bank),
                }
            )

        return masks
