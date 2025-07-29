import numpy as np
from PIL import Image
import torch
from models.clip_wrapper import CLIPEncoder
from models.sam_wrapper import SAMSegmenter
import matplotlib.pyplot as plt

class PromptSegmenter:
    def __init__(self, sam_ckpt_path="./thirdparty/segment-anything/weights/sam_vit_b.pth"):
        self.clip = CLIPEncoder()
        self.sam = SAMSegmenter(checkpoint_path=sam_ckpt_path)

    def run(self, image: Image.Image, prompt: str):
        """Main pipeline to segment object matching the prompt."""
        # Encode the text prompt
        text_feat = self.clip.encode_text(prompt)

        # Generate masks from image
        self.sam.set_image(image)
        masks = self.sam.generate_all_masks()

        # Evaluate masks
        best_mask = None
        best_score = -1
        best_crop = None

        for mask in masks:
            masked_crop = self.extract_masked_region(image, mask)
            img_feat = self.clip.encode_image(masked_crop)
            sim = torch.nn.functional.cosine_similarity(text_feat, img_feat).item()

            if sim > best_score:
                best_score = sim
                best_mask = mask
                best_crop = masked_crop

        return best_mask, best_score

    def extract_masked_region(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Extracts a square bounding box around the masked region and applies the mask."""
        np_image = np.array(image)
        mask = mask.astype(np.uint8) * 255

        # Find bounding box
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return image  # fallback

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # Crop and mask
        cropped_img = np_image[y1:y2+1, x1:x2+1]
        cropped_mask = mask[y1:y2+1, x1:x2+1]
        masked = np.where(cropped_mask[..., None] > 0, cropped_img, 0)

        return Image.fromarray(masked)

def prompt_segmenter_unit_test():
    image_path = "./../data/test/red_apple_image.jpg"
    image = Image.open(image_path).convert("RGB")
    segmenter = PromptSegmenter()

    mask, score = segmenter.run(image, "the red apple")
    print(f"Best match score: {score:.4f}")

    if mask is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(mask, cmap="gray")
        plt.title("Best Matching Mask")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print("No mask found.")

if __name__ == "__main__":
    prompt_segmenter_unit_test()