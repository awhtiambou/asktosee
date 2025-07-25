import torch
from thirdparty.CLIP import clip
from PIL import Image
import torchvision.transforms as transforms

class CLIPEncoder:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def encode_text(self, prompt: str) -> torch.Tensor:
        """Tokenizes and encodes a prompt into a normalized CLIP embedding."""
        tokens = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesses and encodes an image into a normalized CLIP embedding."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        return image_features / image_features.norm(dim=-1, keepdim=True)

def clip_encoder_unit_test():
    encoder = CLIPEncoder()
    text_feat = encoder.encode_text("a red apple")
    img = Image.open("./../data/test/red_apple_image.jpg")
    img_feat = encoder.encode_image(img)

    cos_sim = torch.nn.functional.cosine_similarity(text_feat, img_feat)
    print("Cosine similarity:", cos_sim.item())

if __name__ == "__main__":
    clip_encoder_unit_test()