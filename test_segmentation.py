from PIL import Image
from seg_models.yolov8_seg import mask_from_img

# Load the test image
test_image_path = "test_img.jpg"
test_image = Image.open(test_image_path)

# Perform segmentation
segmented_mask = mask_from_img(test_image, folder="./yolo8_params")
print(segmented_mask)
# Save or display the output
segmented_mask.save("./output/segmented_output.png")

