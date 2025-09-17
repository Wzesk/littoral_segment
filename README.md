# Littoral Segmentation

## Overview

The `littoral_segment` module provides advanced land/water segmentation for the Littoral shoreline analysis pipeline. Using state-of-the-art deep learning models, this module identifies and classifies shoreline boundaries in high-resolution satellite imagery, forming the foundation for accurate shoreline extraction.

## Pipeline Integration

This module serves as **Step 8** in the Littoral pipeline, processing super-resolved imagery after upsampling and before boundary extraction. It implements a standardized interface that allows for easy integration and future model updates.

### Pipeline Context
```
High Resolution Images → [Segmentation] → Binary Masks → Boundary Extraction → ...
```

### Interface
- **Input**: High-resolution, cloud-free satellite imagery (upsampled)
- **Output**: Binary masks delineating land/water boundaries
- **Function**: `YOLOV8.mask_from_folder()`
- **Technology**: YOLO v8 object detection and segmentation models

## Model Development

This repository migrates from the original segmentation notebook:
[Segmentation Colab Notebook](https://colab.research.google.com/drive/1piMHwigRpQZLXWatOZ-kSewfRdyazKnz?usp=drive_link)

### Model Assets
- **Weights**: [YOLOv8 Trained Weights](https://drive.google.com/file/d/1zjiUC86r6KXZ-_UNAFpi72lYb6tYCRIq/view?usp=sharing)
- **Configuration**: [YAML Configuration File](https://drive.google.com/file/d/1rWBJ7ApzF09d2nwRNNX-Y8sVxtlXPYmL/view?usp=sharing)

### Multi-Model Testing

We are actively testing and evaluating multiple ML models for segmentation to ensure optimal performance across diverse coastal environments:

**Current Models Under Evaluation**:
- **YOLO v8-v11**: Object detection with instance segmentation
- **SAM (Segment Anything Model)**: Foundation model for general segmentation
- **SAM2**: Enhanced version with improved temporal consistency
- **Prithvi**: Geospatial foundation model for Earth observation
- **Prithvi2**: Advanced version with multi-temporal capabilities

The goal is to create a site-adaptive system that selects the optimal model based on coastal characteristics and image conditions.

## Usage in Pipeline

```python
import sys
sys.path.append('/path/to/littoral_segment')
from seg_models.yolov8_seg import YOLOV8

# Initialize YOLO segmentation model
yolo_model = YOLOV8(folder='/path/to/yolo8_params')

# Process folder of upsampled images
input_folder = "/path/to/normalized/images"
mask_paths = yolo_model.mask_from_folder(input_folder)

print(f"Generated {len(mask_paths)} segmentation masks")
```

## Deployment Options

This repository is designed for flexible deployment across multiple platforms:

- **Container on Replicate**: Scalable cloud-based inference
- **Google Cloud Pipeline**: Integration with GCP infrastructure  
- **Athina Flow**: Workflow orchestration platform
- **Local Processing**: Direct Python execution

## Docker Deployment

### Building the Container
Build the Docker image using the following command:

```bash
docker build --build-arg PLATFORM=linux/amd64 -t yolov8-segmentation .
```

### Running Segmentation
To segment a test image, run the following command:

```bash
docker run --rm \
    -v $(pwd)/seg_models:/app/seg_models \
    -v $(pwd)/test_img.jpg:/app/test_img.jpg \
    -v $(pwd)/output:/app/output \
    yolov8-segmentation
```

The segmented output will be saved in the output directory as `segmented_output.png`.

## Credits and Attribution

### YOLO Citation
When using this segmentation module in research, please cite the YOLO papers:

**YOLOv8**:
```bibtex
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  url = {https://github.com/ultralytics/ultralytics},
  version = {8.0.0},
  year = {2023}
}
```

**Original YOLO**:
```bibtex
@misc{redmon2016yolo,
    title={You Only Look Once: Unified, Real-Time Object Detection},
    author={Joseph Redmon and Santosh Divvala and Ross Girshick and Ali Farhadi},
    year={2016},
    eprint={1506.02640},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

### Model Attribution
- **Ultralytics YOLO**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Segment Anything Model (SAM)**: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- **Prithvi**: [NASA-IMPACT/Prithvi](https://github.com/NASA-IMPACT/Prithvi)

## Contributors
This module and the larger project it is a part of has had numerous contributors, including:

**Core Development**: Walter Zesk, Tishya Chhabra, Leandra Tejedor, Philip Ndikum

**Project Leadership**: Sarah Dole, Skylar Tibbits, Peter Stempel

## Reference
This project draws extensive inspiration from the [CoastSat Project](https://github.com/kvos/CoastSat) described in detail here:

Vos K., Splinter K.D., Harley M.D., Simmons J.A., Turner I.L. (2019). CoastSat: a Google Earth Engine-enabled Python toolkit to extract shorelines from publicly available satellite imagery. Environmental Modelling and Software. 122, 104528. https://doi.org/10.1016/j.envsoft.2019.104528 (Open Access)
