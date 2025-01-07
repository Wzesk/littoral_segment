# littoral_segment

repo for the segmentation step.  Migrating from this notebook:

https://colab.research.google.com/drive/1piMHwigRpQZLXWatOZ-kSewfRdyazKnz?usp=drive_link

Using these weights:
https://drive.google.com/file/d/1zjiUC86r6KXZ-_UNAFpi72lYb6tYCRIq/view?usp=sharing

and this yaml file:
https://drive.google.com/file/d/1rWBJ7ApzF09d2nwRNNX-Y8sVxtlXPYmL/view?usp=sharing


We are testing several different ML models for segmentation so this repo is expected to change and grow potentially allow for different models to be used based on the site.  

The goal is for this repo to be accessible in different formats (container on replicate, in a google cloud pipeline, in an athina flow)


## Replicate Update

Push Cog to Replicate:

`cog login`
`cog push r8.im/<your-username>/<your-model-name>`

Full documentation: https://replicate.com/docs/guides/push-a-model