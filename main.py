from helperFunctions.imageHelper import ImageHelper
from faceDetection.detectFaces import FaceDetection
import prompts.activities_contentPrompts as activityContentPrompts
import prompts.demographicPrompts as demographicPrompts
import prompts.peoplePrompts as peoplePrompts
import prompts.settingPrompts as settingPrompts
from imageProcessing.ip1 import IP
from objectAnalysis.detectObjects import SAMInference
from objectAnalysis.objectsList import ms_coco
import torch
import clip
import pandas as pd
import glob
import os
import cv2
import argparse
import re


class DatsetAnalysis:

    def __int__(self):
        pass

    def get_model(self):
        model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        return model, preprocess

    def process_predictions(self, list_of_images, prompts, threshold, model, preprocess):
        objHelp = ImageHelper()
        results = {}
        batch_predictions = objHelp.get_clip_prediction_in_batch(list_of_images, list(prompts.keys()), 3, model,
                                                                 preprocess)
        for key, Highest3Predictions in batch_predictions.items():
            try:
                c1 = Highest3Predictions[0][0]
                s1 = 100 * Highest3Predictions[0][1]
                if s1 >= threshold:
                    results[key] = prompts[c1]
                else:
                    results[key] = "others"
            except Exception as e:
                print(e)
        return results

    def process_predictionsDF(self, aa, prompts, threshold, model, preprocess):
        objHelp = ImageHelper()
        results = {}
        # aa = {11: ['a', 'b', 'c'], 22: ['r'], 33: ['x'], 44: ['pp', 'qq']}
        for k, v in aa.items():
            batch_predictions = objHelp.get_clip_prediction_in_batch(v, list(prompts.keys()), 3, model,
                                                                     preprocess)
            ss = []
            for key, Highest3Predictions in batch_predictions.items():
                try:
                    c1 = Highest3Predictions[0][0]
                    s1 = 100 * Highest3Predictions[0][1]
                    if s1 >= threshold:
                        ss.append(prompts[c1])
                    else:
                        ss.append("others")
                except Exception as e:
                    print(e)
            results[k] = ",".join(ss)
        return results

    def process_predictionsObjects(self, aa, model, preprocess, pr):
        objHelp = ImageHelper()
        ObjectResults = {}
        ObjectCountResults = {}
        # aa = {11: ['a', 'b', 'c'], 22: ['r'], 33: ['x'], 44: ['pp', 'qq']}
        prompts = ["a photo of " + str(i) for i in pr]
        for k, v in aa.items():
            batch_predictions = objHelp.get_clip_prediction_in_batch(v, prompts, 3, model,
                                                                     preprocess)
            ss = []
            for key, Highest3Predictions in batch_predictions.items():
                try:
                    c1 = Highest3Predictions[0][0]
                    msc1 = Highest3Predictions[0][0]
                    s1 = 100 * Highest3Predictions[0][1]
                    if s1 >= 70:
                        ss.append(c1)
                    else:
                        ss.append("others")
                except Exception as e:
                    print(e)
            my_list = [x for x in ss if x != "others"]
            zz = (list(set(my_list)))
            if len(zz) != 0:
                ObjectResults[k] = ",".join(zz)
                ObjectCountResults[k] = len(zz)
            else:
                ObjectResults[k] = "others"
                ObjectCountResults[k] = 0
        return ObjectResults, ObjectCountResults

    def settings(self, list_of_images, threshold, model, preprocess):
        # settings 1 [bar, bedroom]
        s1Prompts = settingPrompts.s1
        # settings 2 [lighting]
        s2Prompts = settingPrompts.s2
        # settings 3 [angle]
        s3Prompts = settingPrompts.s3

        s1Results = self.process_predictions(list_of_images, s1Prompts, threshold, model, preprocess)
        s2Results = self.process_predictions(list_of_images, s2Prompts, threshold, model, preprocess)
        s3Results = self.process_predictions(list_of_images, s3Prompts, threshold, model, preprocess)
        return s1Results, s2Results, s3Results

    def face_detection(self, list_of_images):
        obj = FaceDetection()
        df = obj.extractFaces(list_of_images)
        return df

    def activities_content(self, list_of_images, threshold, model, preprocess):
        results = self.process_predictions(list_of_images, activityContentPrompts.activities_content, threshold, model,
                                           preprocess)
        return results

    def objects_analysis(self, list_of_images, model, preprocess, list_of_objects):
        print("In Object Analysys")
        objT = SAMInference()
        res = objT.startProcess(list_of_images)
        ObjectResults, ObjectCountResults = self.process_predictionsObjects(res, model, preprocess, list_of_objects)
        return ObjectResults, ObjectCountResults

    def quality_black_bright(self, list_of_images):
        obj = IP()
        res = {}
        for i in list_of_images:
            st = obj.brightness(i)
            res[i] = st
        return res

    def demographics(self, people_count_results, threshold, model, preprocess):
        others = {}
        with_people = []
        for k, v in people_count_results.items():
            if v == "others":
                others[k] = "others"
            else:
                with_people.append(k)
        res = {}
        df = self.face_detection(with_people)
        # df = pd.read_csv("Results/Faces.csv")

        ss = list(people_count_results.keys())
        others = {}
        for s in ss:
            if s in df['FramesPath'].values:
                pass
            else:
                others[s] = "others"

        df1_grouped = df.groupby('FramesPath')
        for group_name, df_group in df1_grouped:
            res[group_name] = df_group['FacesPath'].tolist()

        ageResults = self.process_predictionsDF(res, demographicPrompts.age_prompts, threshold, model,
                                                preprocess)
        ageResults.update(others)
        raceResults = self.process_predictionsDF(res, demographicPrompts.race_prompts, threshold, model,
                                                 preprocess)
        raceResults.update(others)
        genderResults = self.process_predictionsDF(res, demographicPrompts.gender_prompts, threshold, model,
                                                   preprocess)
        genderResults.update(others)
        return ageResults, raceResults, genderResults

    def people_count(self, list_of_images, threshold, model, preprocess):
        results = self.process_predictions(list_of_images, peoplePrompts.people_count, threshold, model,
                                           preprocess)
        return results

    def qualityDuplicates(self, image_directory):
        obj = IP()
        df = pd.DataFrame(columns=["duplicate images"])
        image_hashes = {}
        for filename in os.listdir(image_directory):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(image_directory, filename)
                img_hash = obj.compute_hash(image_path)
                if img_hash in image_hashes:
                    le = len(df)
                    df.loc[le] = [image_path + image_hashes[img_hash]]
                    print(f'Duplicate: {image_path} and {image_hashes[img_hash]}')
                else:
                    image_hashes[img_hash] = image_path

        return df

    def qualityResolution(self, image_directory, threshold_width, threshold_height):
        obj = IP()
        df = pd.DataFrame(columns=["Low-Resolution Images", "width", "height"])
        for filename in os.listdir(image_directory):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(image_directory, filename)
                if obj.is_low_resolution(image_path, threshold_width, threshold_height):
                    img = cv2.imread(image_path)
                    height, width, _ = img.shape
                    print(f'Low-Resolution Image: {image_path}, Resolution: {width}x{height}')
                    le = len(df)
                    df.loc[le] = [image_path, width, height]

        return df

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def start_process(self, image_directory, threshold, objectsFile, threshold_width, threshold_height):
        res = pd.DataFrame(columns=["image_name", "settings1", "settings2", "settings3",
                                    "objects",
                                    "objects count",
                                    "people count",
                                    "activities_content",
                                    "age",
                                    "gender",
                                    "race",
                                    "quality"])
        model, preprocess = obj.get_model()
        if objectsFile != "":
            list_of_objects = []
            with open("objectFile.txt", 'r') as file:
                lines = file.readlines()
            for line in lines:
                list_of_objects.append(line.strip())
        else:
            list_of_objects = list(ms_coco.values())
        list_of_images = []
        for fi in glob.glob(image_directory + "*"):
            list_of_images.append(fi)

        list_of_images.sort(key=self.natural_keys)
        print(list_of_images)
        s1Results, s2Results, s3Results = self.settings(list_of_images, threshold, model, preprocess)
        activities_content_results = self.activities_content(list_of_images, threshold, model, preprocess)
        people_count_results = self.people_count(list_of_images, threshold, model, preprocess)
        ageResults, raceResults, genderResults = self.demographics(people_count_results, threshold, model, preprocess)
        qualityResults = self.quality_black_bright(list_of_images)
        ObjectResults, ObjectCountResults = self.objects_analysis(list_of_images, model, preprocess, list_of_objects)
        dfDuplicate = self.qualityDuplicates(image_directory)
        dfResolution = self.qualityResolution(image_directory, threshold_width, threshold_height)
        for k, v in s1Results.items():
            try:
                le = len(res)
                res.loc[le] = [v, s2Results[k], s3Results[k], ObjectResults[k], ObjectCountResults[k],
                               people_count_results[k],
                               activities_content_results[k],
                               ageResults[k],
                               genderResults[k], raceResults[k], qualityResults[k]]

                # res.loc[le] = [k, v, s2Results[k], s3Results[k], "ObjectResults[k]", "ObjectCountResults[k]",
                #                "people_count_results[k]",
                #                "activities_content_results[k]",
                #                ageResults[k],
                #                genderResults[k], raceResults[k], "qualityResults[k]"]
            except:
                pass

        res.to_csv("Results/DataAnalysis11.csv")
        dfDuplicate.to_csv("Results/DuplicateImages.csv")
        dfResolution.to_csv("Results/ImagesResolution.csv")


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--folder_name', action='store', type=str, required=True)
    my_parser.add_argument('--threshold', action='store', type=int, required=True)
    my_parser.add_argument('--objectsFile', action='store', type=str, required=True)
    my_parser.add_argument('--threshold_width', action='store', type=int, required=True)
    my_parser.add_argument('--threshold_height', action='store', type=int, required=True)
    args = my_parser.parse_args()
    folder_name = args.folder_name
    threshold = args.threshold
    objectsFile = args.objectsFile
    threshold_width = args.threshold_width
    threshold_height = args.threshold_height

    # folder_name = "data/"
    # threshold = 20
    # objectsFile = ""
    # threshold_width = 600
    # threshold_height = 400
    obj = DatsetAnalysis()
    obj.start_process(folder_name, threshold, objectsFile, threshold_width, threshold_height)

    # !python "/content/gdrive/MyDrive/DatasetAnalysis/main.py" --folder_name data --threshold 20 objectsFile "" --threshold_width 800 --threshold_height 600
