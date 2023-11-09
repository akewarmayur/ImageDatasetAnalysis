import torch
from PIL import Image
from itertools import islice
import sys
import os
import argparse

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import subprocess
import glob
import pandas as pd
import cv2


# import class_map_config


class SAMInference:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def draw_bounding_box(self, image, bbox, color=(0, 255, 0), thickness=2):
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    def download_model(self, url, save_path):
        command = f'wget -O {save_path} {url}'
        subprocess.call(command, shell=True)

    def get_sam_model(self):
        if not os.path.exists("models"):
            os.makedirs("models")
        dir = os.listdir("models")
        if len(dir) == 0:
            self.download_model("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                                os.getcwd() + "/models/sam_vit_h_4b8939.pth")

    def initialize_models(self):
        self.get_sam_model()
        sam_checkpoint = os.getcwd() + "/models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator

    def calculate_normalized_coordinates(self, x1, y1, x2, y2, img):
        image_height, image_width = img.shape[:2]
        x_min_normalized = x1 / image_width
        y_min_normalized = y1 / image_height
        x_max_normalized = x2 / image_width
        y_max_normalized = y2 / image_height
        return x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized

    def getBox(self, SOURCE_IMAGE_PATH, mask_generator):
        ry = SOURCE_IMAGE_PATH.split("/")
        ry = ry[len(ry) - 1]
        z = str(ry).split(".")[0]
        if not os.path.exists(os.getcwd() + "/croppedBoxes"):
            os.makedirs(os.getcwd() + "/croppedBoxes")
        cropped_images = []
        try:
            print(SOURCE_IMAGE_PATH)
            image = cv2.imread(SOURCE_IMAGE_PATH)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image)
            for aa, m in enumerate(masks):
                bbox = tuple(m['bbox'])
                # x1, y1, x2, y2 = bbox
                x, y, width, height = bbox
                if height >= 50 and width >= 50:
                    cropped_image = image[y:y + height, x:x + width]
                    cv2.imwrite(os.getcwd() + "/croppedBoxes/" + str(z) + "_" + str(aa) + ".jpg", cropped_image)
                    cropped_images.append(os.getcwd() + "/croppedBoxes/" + str(z) + "_" + str(aa) + ".jpg")
        except Exception as e:
            print(e)

        return cropped_images

    def startProcess(self, list_of_images):
        mask_generator = self.initialize_models()
        res = {}
        for fi in list_of_images:
            try:
                SOURCE_IMAGE_PATH = fi
                name = SOURCE_IMAGE_PATH.split("/")
                cropped_images = self.getBox(SOURCE_IMAGE_PATH, mask_generator)
                res[fi] = cropped_images
            except Exception as e:
                print(e)
        return res


# if __name__ == '__main__':
#
#     folder_name = "/content/drive/MyDrive/d1/"
#     list_of_images = []
#     for fi in glob.glob(folder_name + "*"):
#         list_of_images.append(fi)
#     objT = SAMInference()
#     res = objT.startProcess(list_of_images)
#     print(res)
