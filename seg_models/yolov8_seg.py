import torchvision.transforms as T
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage
import os


class YOLOV8:
  def __init__(self,folder='/yolo8_params', model_name='yolov8x-seg.pt'):
    self.folder = folder
    self.yaml = self.folder + "/data.yaml"
    self.weights_path = self.folder + '/best.pt'
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
      # Find the sizes of each component
      sizes = ndimage.sum(mask_array, labeled_array, range(num_features + 1))
      # Find the label of the largest component
      largest_component_label = np.argmax(sizes[1:]) + 1
      # Create a new mask with only the largest component
      largest_component_mask = labeled_array == largest_component_label
      largest_component_img = Image.fromarray(largest_component_mask.astype(np.uint8) * 255)
      return largest_component_img

  def mask_from_img(self,up_img):
    up_img = ImageEnhance.Contrast(up_img).enhance(2)
    #run inference
    std_results = self.predict(

      up_img,
      YOLO(self.weights_path)
    )

    if(std_results[0].masks != None):
      #get mask to save
      mask_img = T.ToPILImage()(std_results[0].masks.data[0])

      #clean mask
      single_isl = self.group_contiguous_pixels(mask_img)

      #resize single_isl to match the size of img_r
      single_isl = single_isl.resize(up_img.size)

      return single_isl
    else:
      empty_mask = Image.new('L', up_img.size, 0)
      return empty_mask
        
  def mask_from_folder(self,folder):
    masks = []
    for root, directories, filenames in os.walk(folder):
      for filename in filenames:
        if filename.endswith('_up.png'):
            file_path = os.path.join(root,filename)
            img = Image.open(file_path)
            mask = self.mask_from_img(img)
            mask_path = file_path.replace('_up.png','_mask.png')
            mask.save(mask_path)
            masks.append(mask_path)
    return masks