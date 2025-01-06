# littoral_segment

repo for the segmentation step.  Migrating from this notebook:

https://colab.research.google.com/drive/1piMHwigRpQZLXWatOZ-kSewfRdyazKnz?usp=drive_link

Using these weights:
https://drive.google.com/file/d/1zjiUC86r6KXZ-_UNAFpi72lYb6tYCRIq/view?usp=sharing

and this yaml file:
https://drive.google.com/file/d/1rWBJ7ApzF09d2nwRNNX-Y8sVxtlXPYmL/view?usp=sharing


We are testing several different ML models for segmentation so this repo is expected to change and grow potentially allow for different models to be used based on the site.  

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