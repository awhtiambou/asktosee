import torch
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt


class SAMSegmenter:
    def __init__(self, model_type="vit_b", checkpoint_path="./../thirdparty/segment-anything/weights/sam_vit_b.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(self.device)
        self.predictor = SamPredictor(self.model)

    def set_image(self, image: Image.Image):
        """Sets the image for segmentation (expects PIL Image or NumPy RGB)."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.shape[-1] == 4:
            image = image[:, :, :3]  # remove alpha channel
        self.original_image = image
        self.predictor.set_image(image)

    def generate_all_masks(self):
        """Uses automatic mode to generate masks across the image."""
        height, width = self.original_image.shape[:2]
        input_points = self._generate_grid_points(width, height, grid_size=16)
        input_labels = np.ones(len(input_points))

        masks = []
        for point, label in zip(input_points, input_labels):
            mask, _, _ = self.predictor.predict(
                point_coords=np.array([point]),
                point_labels=np.array([label]),
                multimask_output=False
            )
            masks.append(mask[0])
        return masks  # list of binary (H, W) masks

    def _generate_grid_points(self, width, height, grid_size=16):
        """Generate a uniform grid of points across the image."""
        x_coords = np.linspace(0, width, grid_size, endpoint=False) + width / (grid_size * 2)
        y_coords = np.linspace(0, height, grid_size, endpoint=False) + height / (grid_size * 2)
        points = [(int(x), int(y)) for y in y_coords for x in x_coords]
        return points

def sam_wrapper_unit_test():
    image_path = "./../data/test/red_apple_image.jpg"
    image = Image.open(image_path).convert("RGB")

    # Initialize SAM
    segmenter = SAMSegmenter(
        model_type="vit_b",
        checkpoint_path="./../thirdparty/segment-anything/weights/sam_vit_b.pth"
    )

    # Set the image
    segmenter.set_image(image)

    # Generate masks
    masks = segmenter.generate_all_masks()
    print(f"Generated {len(masks)} masks")

    # Display first 4 masks
    plt.figure(figsize=(10, 6))
    for i in range(min(4, len(masks))):
        plt.subplot(2, 2, i + 1)
        plt.imshow(masks[i], cmap="gray")
        plt.title(f"Mask {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sam_wrapper_unit_test()
