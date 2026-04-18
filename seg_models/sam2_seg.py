from ultralytics import SAM
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
import os


class SAM2Seg:
    """SAM2-based land/water segmentation using Otsu-guided auto-prompting.

    Uses Otsu thresholding on the NIR image to identify approximate land/water
    regions, then provides SAM2 with foreground (land) and background (water)
    point prompts derived from the thresholded mask.

    Interface mirrors YOLOV8 so the two can be swapped seamlessly.
    """

    def __init__(self, model_name='sam2.1_b.pt'):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = SAM(self.model_name)
        return self._model

    def group_contiguous_pixels(self, mask_in):
        """Keep only the largest connected component."""
        mask_array = np.array(mask_in)
        labeled_array, num_features = ndimage.label(mask_array)
        if num_features == 0:
            return mask_in
        sizes = ndimage.sum(mask_array, labeled_array, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        largest_component = labeled_array == largest_label
        return Image.fromarray(largest_component.astype(np.uint8) * 255)

    def _auto_prompts(self, gray, periodic=True):
        """Derive foreground/background point prompts from Otsu thresholding.

        In NIR imagery land is bright and water is dark. Otsu separates them.
        We pick the point deepest inside the largest land component as the
        foreground prompt, and the point farthest from any land as background.

        For non-periodic shorelines (periodic=False), additional foreground
        points are sampled at image-edge locations where Otsu detects land.
        This prevents SAM2 from rounding off the mask at image boundaries.

        Returns (points, labels) suitable for SAM2.
        """
        otsu = threshold_otsu(gray)
        land_mask = gray > otsu
        land_pct = land_mask.sum() / land_mask.size

        # For periodic (island) scenes land should be a small minority, so flip
        # when it exceeds 50%.  For non-periodic (mainland) scenes, land can
        # legitimately cover most of the image; only flip if the bright class
        # is overwhelmingly dominant (>80%), which likely means the threshold
        # captured water glint or cloud rather than true land.
        flip_threshold = 0.5 if periodic else 0.8
        if land_pct > flip_threshold:
            land_mask = ~land_mask

        # Largest connected component of land
        labeled, n = ndimage.label(land_mask)
        if n == 0:
            # Fallback: centre of image as foreground, corner as background
            h, w = gray.shape
            return [[w // 2, h // 2], [0, 0]], [1, 0]

        sizes = ndimage.sum(land_mask, labeled, range(1, n + 1))
        largest_label = int(np.argmax(sizes)) + 1
        largest = labeled == largest_label

        # Foreground: deepest point inside the land mass
        dist_inside = distance_transform_edt(largest)
        fg_y, fg_x = np.unravel_index(dist_inside.argmax(), dist_inside.shape)

        # Background: farthest point from any land
        dist_outside = distance_transform_edt(~land_mask)
        bg_y, bg_x = np.unravel_index(dist_outside.argmax(), dist_outside.shape)

        points = [[int(fg_x), int(fg_y)]]
        labels = [1]

        # For non-periodic shorelines, anchor the mask at image edges where
        # Otsu detects land.  Without these prompts SAM2 draws a closed
        # contour inside the image and rounds off all edge-touching corners.
        if not periodic:
            h, w = gray.shape
            edge_strips = [
                ("top",    land_mask[0, :],    lambda i: (0,   i)),
                ("bottom", land_mask[-1, :],   lambda i: (h-1, i)),
                ("left",   land_mask[:, 0],    lambda i: (i,   0)),
                ("right",  land_mask[:, -1],   lambda i: (i,   w-1)),
            ]
            for _name, strip, rc_fn in edge_strips:
                idx = np.where(strip)[0]
                if len(idx) == 0:
                    continue
                # Sample the midpoint of the land run on this edge
                mid = idx[len(idx) // 2]
                r, c = rc_fn(int(mid))
                points.append([c, r])
                labels.append(1)

        points.append([int(bg_x), int(bg_y)])
        labels.append(0)
        return points, labels

    def mask_from_img(self, pil_img, periodic=True):
        """Generate a binary land mask from a single PIL Image.

        Args:
            pil_img: Input PIL Image (grayscale or RGB).
            periodic: True for island/atoll scenes where the shoreline forms a
                closed loop within the image.  False for mainland scenes where
                the shoreline exits through image edges (non-periodic).

        Returns a PIL Image (mode 'L') with land=255, water=0.
        """
        arr = np.array(pil_img)
        if len(arr.shape) == 3:
            gray = arr.mean(axis=2).astype(np.uint8)
        else:
            gray = arr.astype(np.uint8)

        points, labels = self._auto_prompts(gray, periodic=periodic)

        model = self._load_model()

        # For non-periodic shorelines, reflect-pad the image before inference.
        # SAM2 draws closed contours; without padding it rounds off corners
        # where land exits the image boundary.  Reflecting the image makes the
        # land mass appear to continue naturally beyond the frame so SAM2
        # extends the mask fully to the edges.
        if not periodic:
            pad = max(pil_img.width, pil_img.height) // 4
            img_arr = np.array(pil_img)
            if len(img_arr.shape) == 2:
                padded_arr = np.pad(img_arr, pad, mode='reflect')
            else:
                padded_arr = np.pad(img_arr, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
            inf_img = Image.fromarray(padded_arr)
            # Shift prompts to account for the padding offset
            shifted_points = [[x + pad, y + pad] for x, y in points]
        else:
            inf_img = pil_img
            shifted_points = points
            pad = 0

        orig_size = pil_img.size  # (width, height)
        results = model(inf_img, points=shifted_points, labels=labels, verbose=False)

        if results and results[0].masks is not None and results[0].masks.data.shape[0] > 0:
            mask_data = results[0].masks.data[0].cpu().numpy()
            mask_bin = (mask_data > 0.5).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_bin).resize(inf_img.size, Image.NEAREST)
            if pad > 0:
                mask_arr = np.array(mask_img)
                mask_arr = mask_arr[pad:-pad, pad:-pad]
                mask_img = Image.fromarray(mask_arr)
            mask_img = mask_img.resize(orig_size, Image.NEAREST)
            return mask_img

        # SAM2 returned no mask — fall back to the Otsu threshold directly
        flip_threshold = 0.5 if periodic else 0.8
        otsu = threshold_otsu(gray)
        land_mask = gray > otsu
        if land_mask.sum() / land_mask.size > flip_threshold:
            land_mask = ~land_mask
        mask_img = Image.fromarray(land_mask.astype(np.uint8) * 255)
        return mask_img.resize(orig_size, Image.NEAREST)

    def mask_from_folder(self, folder, periodic=True, selection_config=None):
        """Walk *folder* for upsampled NIR images and save masks.

        Matches the YOLOV8 interface: looks for files containing ``_x``
        followed by a digit, writes masks to a sibling ``MASK/`` directory
        with the ``_mask.png`` suffix.

        Returns a list of saved mask paths.
        """
        self.last_qc_records = []
        masks = []
        for root, _dirs, filenames in os.walk(folder):
            for filename in filenames:
                if '_x' in filename and filename.split('_x')[1][0].isdigit():
                    file_path = os.path.join(root, filename)
                    img = Image.open(file_path)
                    mask = self.mask_from_img(img, periodic=periodic)

                    mask_path = file_path.replace('UP', 'MASK')
                    mask_path = mask_path.replace('NORMALIZED', 'MASK')
                    mask_path = mask_path.split('_x')[0] + '_mask.png'

                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                    mask.save(mask_path)
                    masks.append(mask_path)
                    self.last_qc_records.append({
                        "image_name": filename,
                        "mask_name": os.path.basename(mask_path),
                        "mask_path": mask_path,
                        "candidate_count": 1,
                        "selected_index": 0,
                        "periodic": bool(periodic),
                    })
        return masks
