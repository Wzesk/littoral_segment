import torchvision.transforms as T
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage
import os


class YOLO26Seg:
    """YOLO26 instance segmentation for land/water masks.

    Expects a folder containing fine-tuned .pt weights (e.g. from Roboflow).
    Interface mirrors YOLOV8 / SAM2Seg for pipeline compatibility.
    """

    def __init__(self, folder='yolo26_params', model_name=None):
        self.folder = folder
        # Locate weights file
        weights_files = [f for f in os.listdir(self.folder) if f.endswith('.pt')]
        if not weights_files:
            raise ValueError(f"No .pt weights found in {self.folder}")
        self.weights_path = os.path.join(self.folder, weights_files[0])
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = YOLO(self.weights_path)
        return self._model

    def group_contiguous_pixels(self, mask_in):
        """Keep only the largest connected component."""
        mask_array = np.array(mask_in)
        labeled_array, num_features = ndimage.label(mask_array)
        if num_features == 0:
            return mask_in
        sizes = ndimage.sum(mask_array, labeled_array, range(num_features + 1))
        largest_label = np.argmax(sizes[1:]) + 1
        largest_mask = (labeled_array == largest_label).astype(np.uint8) * 255
        return Image.fromarray(largest_mask)

    def mask_from_img(self, pil_img, retina_masks=True, padding=256, contrast=2.0):
        """Generate binary land/water mask from a single image.

        Args:
            pil_img: Input PIL Image (grayscale or RGB).
            retina_masks: Use high-res masks (default True).
            padding: Pixels to pad to avoid bbox edge artifacts.
            contrast: Contrast enhancement factor (1.0 = no change).

        Returns:
            PIL Image mode 'L': land=255, water=0.
        """
        orig_size = pil_img.size  # (width, height)

        # Contrast enhancement
        if contrast != 1.0:
            pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)

        # Pad to prevent edge artifacts
        img_array = np.array(pil_img)
        if padding > 0:
            if len(img_array.shape) == 2:
                pad_value = int(np.mean(img_array[:, :10]))
                padded_array = np.pad(img_array, padding,
                                      mode='constant', constant_values=pad_value)
                padded_img = Image.fromarray(padded_array)
            else:
                pad_value = tuple(int(np.mean(img_array[:, :10, c])) for c in range(3))
                padded_array = np.pad(img_array,
                                      ((padding, padding), (padding, padding), (0, 0)),
                                      mode='constant', constant_values=0)
                padded_array[:padding, :] = pad_value
                padded_array[-padding:, :] = pad_value
                padded_array[:, :padding] = pad_value
                padded_array[:, -padding:] = pad_value
                padded_img = Image.fromarray(padded_array)
        else:
            padded_img = pil_img

        model = self._get_model()
        results = model(padded_img, retina_masks=retina_masks, verbose=False)

        if results[0].masks is not None and results[0].masks.data.shape[0] > 0:
            mask_tensor = results[0].masks.data[0]
            mask_img = T.ToPILImage()(mask_tensor)

            # Resize to padded dimensions then crop padding
            mask_img = mask_img.resize(padded_img.size)
            if padding > 0:
                mask_array = np.array(mask_img)
                mask_array = mask_array[padding:-padding, padding:-padding]
                mask_img = Image.fromarray(mask_array)

            # Keep largest connected component
            mask_img = self.group_contiguous_pixels(mask_img)
            mask_img = mask_img.resize(orig_size)
            return mask_img
        else:
            return Image.new('L', orig_size, 0)

    def mask_from_folder(self, folder):
        """Generate masks for all upsampled/normalized images in a folder.

        Mirrors YOLOV8.mask_from_folder() interface.
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
