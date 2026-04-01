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

    def _auto_prompts(self, gray):
        """Derive foreground/background point prompts from Otsu thresholding.

        In NIR imagery land is bright and water is dark. Otsu separates them.
        We pick the point deepest inside the largest land component as the
        foreground prompt, and the point farthest from any land as background.

        Returns (points, labels) suitable for SAM2.
        """
        otsu = threshold_otsu(gray)
        land_mask = gray > otsu
        land_pct = land_mask.sum() / land_mask.size

        # Ensure land is the minority class (bright in NIR)
        if land_pct > 0.5:
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

        points = [[int(fg_x), int(fg_y)], [int(bg_x), int(bg_y)]]
        labels = [1, 0]
        return points, labels

    def mask_from_img(self, pil_img):
        """Generate a binary land mask from a single PIL Image.

        Returns a PIL Image (mode 'L') with land=255, water=0.
        """
        arr = np.array(pil_img)
        if len(arr.shape) == 3:
            gray = arr.mean(axis=2).astype(np.uint8)
        else:
            gray = arr.astype(np.uint8)

        points, labels = self._auto_prompts(gray)

        model = self._load_model()
        results = model(pil_img, points=points, labels=labels, verbose=False)

        orig_size = pil_img.size  # (width, height)

        if results and results[0].masks is not None and results[0].masks.data.shape[0] > 0:
            mask_data = results[0].masks.data[0].cpu().numpy()
            mask_bin = (mask_data > 0.5).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_bin).resize(orig_size, Image.NEAREST)
            return mask_img

        # SAM2 returned no mask — fall back to the Otsu threshold directly
        otsu = threshold_otsu(gray)
        land_mask = gray > otsu
        if land_mask.sum() / land_mask.size > 0.5:
            land_mask = ~land_mask
        mask_img = Image.fromarray(land_mask.astype(np.uint8) * 255)
        return mask_img.resize(orig_size, Image.NEAREST)

    def mask_from_folder(self, folder):
        """Walk *folder* for upsampled NIR images and save masks.

        Matches the YOLOV8 interface: looks for files containing ``_x``
        followed by a digit, writes masks to a sibling ``MASK/`` directory
        with the ``_mask.png`` suffix.

        Returns a list of saved mask paths.
        """
        masks = []
        for root, _dirs, filenames in os.walk(folder):
            for filename in filenames:
                if '_x' in filename and filename.split('_x')[1][0].isdigit():
                    file_path = os.path.join(root, filename)
                    img = Image.open(file_path)
                    mask = self.mask_from_img(img)

                    mask_path = file_path.replace('UP', 'MASK')
                    mask_path = mask_path.replace('NORMALIZED', 'MASK')
                    mask_path = mask_path.split('_x')[0] + '_mask.png'

                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                    mask.save(mask_path)
                    masks.append(mask_path)
        return masks
