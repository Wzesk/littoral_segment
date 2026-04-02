import torch
torch.backends.cudnn.enabled = False
import torchvision.transforms as T
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage
import os

from .shoreline_selection import select_best_mask


class YOLOV8:
  def __init__(self,folder='/yolo8_params', model_name='yolov8x-seg.pt'):
    self.folder = folder
    self.yaml = self.folder + "/data.yaml"
    # check the folder for any weights file with a .pt extension
    weights_files = [f for f in os.listdir(self.folder) if f.endswith('.pt')]
    if weights_files:
        self.weights_path = os.path.join(self.folder, weights_files[0])
    else:
        raise ValueError(f"No weights file found in {self.folder}")
    self.model_name = model_name
    
  # need to retrain at higher res to match size of island
  def train(self,epochs=100,imgsz=640,batch=8,mask_ratio=4,name='yolov8x_nir'):
    model = YOLO(self.model_name)
    results = model.train(
      data=self.yaml,
      imgsz=imgsz,
      epochs=epochs,
      batch=batch,
      seed=7,
      name=name,
      mask_ratio=mask_ratio
    )

  def predict(self,image,model):
    results = model(image)
    outDictOfClasses = {}
    return results

  def group_contiguous_pixels(self, mask_in):
      mask_array = np.array(mask_in)
      # Label connected components
      labeled_array, num_features = ndimage.label(mask_array)
      if num_features == 0:
          return Image.fromarray(np.zeros_like(mask_array, dtype=np.uint8))
      # Find the sizes of each component
      sizes = ndimage.sum(mask_array, labeled_array, range(num_features + 1))
      # Find the label of the largest component
      largest_component_label = np.argmax(sizes[1:]) + 1
      # Create a new mask with only the largest component
      largest_component_mask = labeled_array == largest_component_label
      largest_component_img = Image.fromarray(largest_component_mask.astype(np.uint8) * 255)
      return largest_component_img

  def mask_from_img(self, up_img, retina_masks=True, padding=256, return_qc=False):
    """
    Generate segmentation mask from image.
    
    Args:
        up_img: Input PIL Image
        retina_masks: Use high-res masks (default True)
        padding: Pixels to pad around image to avoid bounding box edge artifacts
    """
    orig_size = up_img.size  # (width, height)
    
    # Enhance contrast
    up_img = ImageEnhance.Contrast(up_img).enhance(2)
    
    # Add padding to avoid bounding box artifacts at edges
    if padding > 0:
      # Get the mean color for padding (or use black for grayscale)
      img_array = np.array(up_img)
      if len(img_array.shape) == 2:
        # Grayscale - pad with edge mean
        pad_value = int(np.mean(img_array[:, :10]))  # Use left edge mean
        padded_array = np.pad(img_array, padding, mode='constant', constant_values=pad_value)
        padded_img = Image.fromarray(padded_array)
      else:
        # RGB - pad each channel
        pad_value = tuple(int(np.mean(img_array[:, :10, c])) for c in range(3))
        padded_array = np.pad(img_array, ((padding, padding), (padding, padding), (0, 0)), 
                             mode='constant', constant_values=0)
        # Fill with pad_value
        padded_array[:padding, :] = pad_value
        padded_array[-padding:, :] = pad_value
        padded_array[:, :padding] = pad_value
        padded_array[:, -padding:] = pad_value
        padded_img = Image.fromarray(padded_array)
    else:
      padded_img = up_img
    
    # Run inference with retina_masks for high-res output
    model = YOLO(self.weights_path)
    std_results = model(padded_img, retina_masks=retina_masks)

    qc = {"candidate_count": 0, "selected_index": None}

    if std_results[0].masks is not None and std_results[0].masks.data.shape[0] > 0:
      result = std_results[0]
      mask_arrays = []
      confidences = []
      for idx in range(result.masks.data.shape[0]):
        mask_array = result.masks.data[idx].cpu().numpy()
        mask_arrays.append((mask_array > 0.5).astype(np.uint8) * 255)
        if result.boxes is not None and len(result.boxes) > idx:
          confidences.append(float(result.boxes.conf[idx].cpu().numpy()))
        else:
          confidences.append(0.0)

      selected_mask, qc = select_best_mask(
        mask_arrays,
        confidences=confidences,
        periodic=getattr(self, "_selection_periodic", True),
        config=getattr(self, "_selection_config", None),
      )
      mask_array = selected_mask
      mask_img = Image.fromarray(mask_array, mode='L')
      
      # Resize mask to padded image size
      padded_size = padded_img.size
      mask_img = mask_img.resize(padded_size, Image.NEAREST)
      
      # Crop padding from mask
      if padding > 0:
        mask_array = np.array(mask_img)
        mask_array = mask_array[padding:-padding, padding:-padding]
        mask_img = Image.fromarray(mask_array, mode='L')
      
      # Preserve raw model topology; shoreline extraction handles site-aware cleanup.
      mask_img = mask_img.resize(orig_size, Image.NEAREST)
      return (mask_img, qc) if return_qc else mask_img
    else:
      empty_mask = Image.new('L', orig_size, 0)
      return (empty_mask, qc) if return_qc else empty_mask
        
  def mask_from_folder(self,folder, periodic=True, selection_config=None):
    self._selection_periodic = periodic
    self._selection_config = selection_config or {}
    self.last_qc_records = []
    masks = []
    for root, directories, filenames in os.walk(folder):
      for filename in filenames:
        #if filename contains '_x' followed by a number
        if '_x' in filename and filename.split('_x')[1][0].isdigit():
            file_path = os.path.join(root,filename)
            img = Image.open(file_path)
            mask, qc = self.mask_from_img(img, return_qc=True)
            mask_path = file_path.replace('UP','MASK')
            mask_path = mask_path.replace('NORMALIZED','MASK')
            # split file name at _x and replace the rest with _mask.png
            mask_path = mask_path.split('_x')[0] + '_mask.png'

            # create directory if it doesn't exist
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)

            mask.save(mask_path)
            masks.append(mask_path)
            qc.update({
                "image_name": filename,
                "mask_name": os.path.basename(mask_path),
                "mask_path": mask_path,
            })
            self.last_qc_records.append(qc)
    return masks
