from PIL import Image, ImageStat
import cv2
import imagehash
import os
import pandas as pd


class IP:
    def brightness(self, im_file):
        im = Image.open(im_file).convert('L')
        stat = ImageStat.Stat(im)
        aa = stat.mean[0]
        if aa > 30:
            return "bright"
        else:
            return "dark"

    def compute_hash(self, image_path):
        image = cv2.imread(image_path)
        resized = cv2.resize(image, (8, 8))  # Resize the image to a fixed size
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return imagehash.average_hash(Image.fromarray(gray))

    import cv2
    import os

    # Define a function to check if an image is low resolution
    def is_low_resolution(self, image_path, threshold_width, threshold_height):
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        return width <= threshold_width or height <= threshold_height

# obj = IP()
# Image Duplicay
# Specify the directory where your images are stored
# df = pd.DataFrame(columns=["duplicate images"])
# image_directory = '/home/manu/PycharmProjects/DatasetAnalysis/d1'
#
# # Create a dictionary to store image hashes and their corresponding file paths
# image_hashes = {}
#
# # Iterate through the image directory and compute hashes
# for filename in os.listdir(image_directory):
#     if filename.endswith(('.jpg', '.png', '.jpeg')):
#         image_path = os.path.join(image_directory, filename)
#         img_hash = obj.compute_hash(image_path)
#         if img_hash in image_hashes:
#             le = len(df)
#             df.loc[le] = [image_path + image_hashes[img_hash]]
#             print(f'Duplicate: {image_path} and {image_hashes[img_hash]}')
#         else:
#             image_hashes[img_hash] = image_path
#
# df.to_csv("ff.csv")

# df = pd.DataFrame(columns=["Low-Resolution Images", "width", "height"])
# # Image resolution
# # Specify the directory where your images are stored
# image_directory = '/home/manu/PycharmProjects/DatasetAnalysis/d1/'
#
# # Define resolution thresholds (e.g., 800x600 pixels)
# threshold_width = 800
# threshold_height = 600
#
# # Iterate through the image directory and check for low-resolution images
# for filename in os.listdir(image_directory):
#     if filename.endswith(('.jpg', '.png', '.jpeg')):
#         image_path = os.path.join(image_directory, filename)
#         if obj.is_low_resolution(image_path, threshold_width, threshold_height):
#             img = cv2.imread(image_path)
#             height, width, _ = img.shape
#             print(f'Low-Resolution Image: {image_path}, Resolution: {width}x{height}')
#             le = len(df)
#             df.loc[le] = [image_path, width, height]
#
# df.to_csv("ff.csv")
