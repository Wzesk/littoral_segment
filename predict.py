# predict.py
from cog import BasePredictor, Input, Path
from PIL import Image

from seg_models.yolov8_seg import mask_from_img

class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(
        self,
        image: Path = Input(description="Input image to segment"),
    ) -> Path:
        """
        Run segmentation on an input image and return the mask.
        """
        # Load image
        up_img = Image.open(image).convert("RGB")

        # Run segmentation; mask_from_img will call YOLO internally
        mask = mask_from_img(up_img, folder="./seg_models/yolo8_params")

        # Save the resulting mask
        output_path = "mask.png"
        mask.save(output_path)

        # Return mask as final output
        return Path(output_path)
