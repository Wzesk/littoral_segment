# littoral_segment

repo for the segmentation step.  Migrating from this notebook:

https://colab.research.google.com/drive/1piMHwigRpQZLXWatOZ-kSewfRdyazKnz?usp=drive_link

Using these weights:
https://drive.google.com/file/d/1zjiUC86r6KXZ-_UNAFpi72lYb6tYCRIq/view?usp=sharing

and this yaml file:
https://drive.google.com/file/d/1rWBJ7ApzF09d2nwRNNX-Y8sVxtlXPYmL/view?usp=sharing


We are testing several different ML models for segmentation so this repo is expected to change and grow potentially allow for different models to be used based on the site.  

The models tested include YOLO v8-v11, SAM, SAM2, Prithvi, Prithvi2

The goal is for this repo to be accessible in different formats (container on replicate, in a google cloud pipeline, in an athina flow)


## Docker Updates

Build the Docker image using the following command:

`docker build --build-arg PLATFORM=linux/amd64 -t yolov8-segmentation .`

To segment a test image, run the following command:

`docker run --rm \
    -v $(pwd)/seg_models:/app/seg_models \
    -v $(pwd)/test_img.jpg:/app/test_img.jpg \
    -v $(pwd)/output:/app/output \
    yolov8-segmentation`

The segmented output will be saved in the output directory as segmented_output.png.

## Contributors
This module and the larger project it is a part of has had numerous contributors, including:

Walter Zesk, Tishya Chhabra, Leandra Tejedor, Philip Ndikum

Sarah Dole, Skylar Tibbits, Peter Stempel

## Reference
This project draws extensive inspiration from the [Coastsal Project](https://github.com/kvos/CoastSat) described in detail here:

Vos K., Splinter K.D., Harley M.D., Simmons J.A., Turner I.L. (2019). CoastSat: a Google Earth Engine-enabled Python toolkit to extract shorelines from publicly available satellite imagery. Environmental Modelling and Software. 122, 104528. https://doi.org/10.1016/j.envsoft.2019.104528 (Open Access)
