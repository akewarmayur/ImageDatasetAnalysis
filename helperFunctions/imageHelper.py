from itertools import islice
import re
import torch
import clip
from PIL import Image


class ImageHelper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_top_prediction(self, image_path, list_of_prompts, no_of_top_predictions, model, preprocess) -> list:
        Highest3Predictions = []
        try:
            text = clip.tokenize(list_of_prompts).to(self.device)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                probs = probs.tolist()[0]
            vv = {}
            for i, j in enumerate(probs):
                vv[list_of_prompts[i]] = j
            maxx = {k: v for k, v in sorted(vv.items(), key=lambda item: item[1], reverse=True)}
            Highest3Predictions = list(islice(maxx.items(), no_of_top_predictions))
            print(f"{image_path} : {Highest3Predictions}")
        except:
            pass

        return Highest3Predictions

    def get_clip_prediction_in_batch(self, list_of_images, list_of_prompts, no_of_top_predictions, model, preprocess):
        predictions = {}
        for fi in list_of_images:
            Highest3Predictions = self.get_top_prediction(fi, list_of_prompts, no_of_top_predictions, model, preprocess)
            if len(Highest3Predictions) != 0:
                predictions[fi] = Highest3Predictions
        return predictions

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]
