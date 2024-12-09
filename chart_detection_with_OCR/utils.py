import json
import os
import random
import boto3
import cv2
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import math
from PIL import Image
import fitz
from ultralytics import YOLO
import torch
import shutil
from chart_detection_with_OCR.blob_chart_detection_download import (
    chart_model_downloader,
)
import settings
import elastic_logging
from decimal import *

# Amazon Rekognition
client = boto3.client(
    "rekognition",
    region_name=settings.AWS_DETECT_TEXT_REGION,
    aws_access_key_id=settings.AWS_DETECT_TEXT_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_DETECT_TEXT_ACCESS_KEY,
)


class Chart_detection:
    def __init__(self, file_path, path_job_id=None) -> None:
        self.path_job_id = path_job_id
        self.pdf_path = file_path
        print(f"pdf path : {file_path}")

    def detect_charts(self):

        if self.path_job_id:
            image_dir = os.path.join(self.path_job_id, "pdf_images")
        else:
            image_dir = "pdf_images"
        try:
            os.makedirs(image_dir, exist_ok=True)
        except:
            pass

        if self.path_job_id:
            output_dir = os.path.join(self.path_job_id, "cropped_objects")
        else:
            output_dir = "cropped_objects"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except:
            pass

        chart_model_downloader()

        model = YOLO("model/best_model_chart_detect.pt")
        # model = YOLO(r"C:\Users\reddi\chart_parser\model\best_model_chart_detect.pt")

        cropped_objects = []

        pdf_document = fitz.open(self.pdf_path)

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # Convert to DPI 300 images
            page_image_path = os.path.join(image_dir, f"page_{page_num + 1}.png")
            pix.pil_save(page_image_path, format="png", dpi=(300, 300))
            page_image = Image.open(page_image_path)
            width, height = page_image.size

            results = model.predict(
                source=page_image_path,
                conf=0.80,
                save=True,
                save_txt=True,
                save_crop=True,
            )

            # croped_dir = 'runs/detect/predict/crops/chart'
            label_dir = "runs/detect/predict/labels/"

            if not os.path.exists(label_dir):
                print("No path or directory")
                pass
            else:
                for filename in os.listdir(label_dir):
                    print(filename)
                    if filename.endswith(".txt"):
                        with open(os.path.join(label_dir, filename), "r") as f:
                            img_name = filename.replace(".txt", ".png")
                            img_path = os.path.join(image_dir, img_name)
                            with Image.open(img_path) as img:
                                print(f"page width: {width}, page height: {height}")
                                for index, line in enumerate(f.readlines(), start=1):
                                    object_class, x_center_n, y_center_n, w_n, h_n = map(float, line.split())

                                    left_n = round(float(x_center_n - (w_n / 2)), 6)
                                    top_n = round(float(y_center_n - (h_n / 2)), 6)

                                    x_center, y_center, w, h = (
                                        x_center_n * width,
                                        y_center_n * height,
                                        w_n * width,
                                        h_n * height,
                                    )

                                    x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                                    x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

                                    cropped_img = img.crop((x1, y1, x2, y2))

                                    cropped_img_name = (
                                        f"{filename.replace('.txt', '')}_object_{int(object_class)}_{index}.png"
                                    )
                                    cropped_image_path = os.path.join(output_dir, cropped_img_name)
                                    cropped_img.save(cropped_image_path)

                                    cropped_objects.append(
                                        {
                                            "page_num": page_num,
                                            "image_path": cropped_image_path,
                                            "chart_width": math.ceil(w),
                                            "chart_height": math.ceil(h),
                                            "chart_x": int(x_center),
                                            "chart_y": int(y_center),
                                            "chart_left": x1,
                                            "chart_top": y1,
                                            "chart_width_norm": round(w_n, 6),
                                            "chart_height_norm": round(h_n, 6),
                                            "chart_left_norm": left_n,
                                            "chart_top_norm": top_n,
                                            "page_width": width,
                                            "page_height": height,
                                        }
                                    )
            shutil.rmtree("runs")
        result = pd.DataFrame(cropped_objects)
        return result
    
    def charts_by_grids(self):

        def assign_row_group(sub_df):
            threshold = 100
            sub_df = sub_df.sort_values(by='chart_y', ascending=True).reset_index(drop=True)
            group = 0
            sub_df['row_group'] = 0  # Initialize the row_group column
            
            # Iterate over rows and assign row groups based on threshold
            for i in range(1, len(sub_df)):
                if abs(sub_df.loc[i, 'chart_y'] - sub_df.loc[i - 1, 'chart_y']) > threshold:
                    group += 1
                sub_df.loc[i, 'row_group'] = group
            
            return sub_df
        obj_charts = self.detect_charts()
        df_grouped = obj_charts.groupby('page_num', group_keys=False).apply(assign_row_group)

        df_sorted = df_grouped.sort_values(by=['page_num', 'row_group', 'chart_x']).reset_index(drop=True)

        return df_sorted

    def clean_folder(self, folder_path):
        shutil.rmtree(folder_path)


class TextProcessor:
    def __init__(self, img_path: str, es_id:None, log_data:None) -> None:
        self.img_path = img_path
        self.textDetDF = self.detectTextAWS()
        self.textDetDF_copy = (
            self.textDetDF.copy()
        )  # Keep this copy to yourself and make all changes to textDetDF
        self.xlabels = self.find_xlabels(0.65, es_id, log_data)
        self.ylabels, self.y_labels_dataframe_cleaned = self.find_ylabels(
            es_id, log_data
        )
        self.y_labels_dataframe_original = self.find_y_dataframe()
        self.x_labels_dataframe = self.textDetDF_copy.loc[
            self.textDetDF["text"].isin(self.xlabels)
        ]

        self.presence_of_y = ""
        if self.y_labels_dataframe_cleaned.empty == False:
            self.presence_of_y = "yes"
            try:
                self.y_labels_dataframe_cleaned["clean_text"] = (
                    self.y_labels_dataframe_cleaned["text"].apply(
                        lambda x: float(self.clean_ylabels(x))
                    )
                )
            except:
                self.y_labels_dataframe_cleaned["clean_text"] = 0.0
            print("Y dataframe after clean value:")
            print(self.y_labels_dataframe_cleaned)
            self.axis_y_min = self.y_labels_dataframe_cleaned["clean_text"].apply(lambda x: float(x)).min()
            print(f"Y min value: {self.axis_y_min}")
            self.y_min_corr = self.y_labels_dataframe_cleaned.loc[
                self.y_labels_dataframe_cleaned["clean_text"] == self.axis_y_min,
                "y_val",
            ].tolist()[0]
            self.axis_y_max = self.y_labels_dataframe_cleaned["clean_text"].apply(lambda x: float(x)).max()
            self.y_axis_height = (
                self.y_labels_dataframe_cleaned["y_center"].max() - self.y_labels_dataframe_cleaned["y_center"].min()
            )
        else:
            self.presence_of_y = "no"

        #         #print(self.textDetDF)
        print(self.presence_of_y)

    def detectTextAWS(self):
        """Detect text from image using AWS rekognition
        Args:
            img_path (str): full path of image
        """
        image = cv2.imread(self.img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, im_buf = cv2.imencode("." + self.img_path.split(".")[-1], image)

        response = client.detect_text(Image={"Bytes": im_buf.tobytes()})
        textDetections = response["TextDetections"]
        textDetList = []
        for textDet in textDetections:
            if textDet["Type"] == "WORD" and textDet["Confidence"] >= 60:
                textDetList.append(
                    {
                        "text": textDet.get("DetectedText"),
                        "parentID": textDet.get("ParentId"),
                        "width": textDet.get("Geometry").get("BoundingBox").get("Width"),
                        "height": textDet.get("Geometry").get("BoundingBox").get("Height"),
                        "x_val": textDet.get("Geometry").get("BoundingBox").get("Left"),
                        "y_val": textDet.get("Geometry").get("BoundingBox").get("Top"),
                    }
                )
        textDetDF = pd.DataFrame(textDetList)
        textDetDF["x_center"] = textDetDF["x_val"] + textDetDF["width"] / 2
        textDetDF["y_center"] = textDetDF["y_val"] + textDetDF["height"] / 2
        textDetDF["x_right"] = textDetDF["x_val"] + textDetDF["width"]

        print(textDetDF)
        return textDetDF

    def find_xlabels(self, y_center_thresh, es_id, log_data):
        self.textDetDF["y_center"] = self.textDetDF["y_center"].apply(lambda x: round(x, 2))
        self.textDetDF["x_center"] = self.textDetDF["x_center"].apply(lambda x: round(x, 2))
        x_candidate_filter = self.textDetDF.loc[self.textDetDF["y_center"] > y_center_thresh]
        y_centers = x_candidate_filter["y_center"].tolist()

        axis_value = max(set(y_centers), key=y_centers.count, default=0)
        min_axis_value = round(axis_value - 0.01, 3)
        max_axis_value = round(axis_value + 0.03, 3)
        x_candidate_filter = x_candidate_filter.loc[
            (x_candidate_filter["y_center"] >= min_axis_value) & (x_candidate_filter["y_center"] <= max_axis_value)
        ].copy()
        self.x_candidate_filter = x_candidate_filter
        # self.textDetDF = self.textDetDF.drop(x_candidate_filter.index)
        x_labels = x_candidate_filter["text"].tolist()
        print("### X labels##")
        print(x_labels)
        log_data["details"].append({"x_labels": x_labels})
        elastic_logging.update_in_elasticsearch(es_id, log_data)
        return x_labels

    def find_y_dataframe(self, x_center_thresh: float = 0.33):
        self.textDetDF["y_center"] = self.textDetDF["y_center"].apply(
            lambda x: round(x, 2)
        )
        self.textDetDF["x_center"] = self.textDetDF["x_center"].apply(
            lambda x: round(x, 2)
        )
        self.textDetDF["x_right"] = self.textDetDF["x_right"].apply(
            lambda x: round(x, 2)
        )
        y_candidate_filter = self.textDetDF.loc[
            self.textDetDF["x_center"] < x_center_thresh
        ]

        # x_centers = y_candidate_filter["x_center"].tolist()
        # axis_value = max(set(x_centers), key=x_centers.count)
        x_rights = y_candidate_filter["x_right"].tolist()
        axis_value = max(set(x_rights), key=x_rights.count)
        min_axis_value = round(axis_value - 0.1, 3)
        max_axis_value = round(axis_value + 0.01, 3)
        y_candidate_filter = y_candidate_filter.loc[
            (y_candidate_filter["x_center"] >= min_axis_value) & (y_candidate_filter["x_center"] <= max_axis_value)
        ].copy()
        pattern = r"^(\$?\d+[a-zA-Z]+|\d+[a-zA-Z]+|\$?\d+)"
        y_candidate_filter = y_candidate_filter[
            y_candidate_filter["text"].str.contains(pattern, na=False)
        ]

        return y_candidate_filter

    def find_ylabels(self, es_id, log_data):
        y_filter = self.find_y_dataframe()
        print("Y filter after string manipulation: ")
        print(y_filter)

        y_filter["text"] = y_filter["text"].apply(self.clean_ylabels)
        y_filter["text"] = y_filter["text"].apply(self.convert_currency)

        BAD_CHARS = ["&", ";", "-", ":"]
        pat = "|".join(["({})".format(re.escape(c)) for c in BAD_CHARS])

        y_filter = y_filter[~y_filter["text"].str.contains(pat)]

        y_candidate_filter = y_filter
        print(y_candidate_filter)

        y_labels = []
        for val in y_candidate_filter["text"].tolist():
            try:
                float(self.clean_ylabels(val))
                y_labels.append(val)
            except Exception as e:
                print(e)
                continue
        print("### Y labels###")
        print(y_labels)
        log_data["details"].append({"y_labels": y_labels})
        elastic_logging.update_in_elasticsearch(es_id, log_data)
        return y_labels, y_candidate_filter

    @staticmethod
    def convert_currency(text: str):
        units = {
            "billion": 1000000000,
            "billions": 1000000000,
            "b": 1000000000,
            "bn": 1000000000,
            "million": 1000000,
            "millions": 1000000,
            "m": 1000000,
            "mm": 1000000,
            "mn": 1000000,
            "thousand": 1000,
            "thousands": 1000,
            "k": 1000,
            "hundred": 100,
            "hundreds": 100,
            "%": 0.01,
        }
        text = text.lower()
        for unit, multiplier in units.items():
            if unit in text:
                # Extract the number from the input_string
                number_part = text.split(unit)[0]
                if unit == "%":
                    try:
                        number_DECIMAL = Decimal(
                            number_part
                        )  # convert string to Decimal
                        number_DECIMAL_scaled_down = (
                            number_DECIMAL / 100
                        )  # Convert Decimal percentage to float by sacling down by 100
                        return str(float(number_DECIMAL_scaled_down))
                    except:
                        continue
                try:
                    number = float(number_part)
                    # Multiply the number with the corresponding multiplier
                    result = number * multiplier
                    return str(result)
                except ValueError:
                    # If the conversion to float fails, move to the next unit
                    continue

        # If no units are found, return the input string as-is
        return text

    @staticmethod
    def clean_ylabels(text: str):
        text = text.replace("$", "")
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("x", "")
        text = text.replace(" ", "")
        text = text.replace("-", "")
        text = text.replace("O", "0")
        text = text.replace("S", "")
        text = text.replace(",", "")
        return text

    def get_texts_not_x_and_y_labels(self):
        x_df = self.x_labels_dataframe
        y_df = self.y_labels_dataframe_original
        df_common = pd.merge(x_df, y_df, on="text", how="outer")
        rest_texts_df = self.textDetDF_copy[~self.textDetDF_copy["text"].isin(df_common["text"])]
        rest_texts_df = rest_texts_df.dropna(subset=["text"])
        # print("#### Rest of the text apart from x and y labels ####")
        # print(rest_texts_df)
        return rest_texts_df

    def get_actual_bar_height(self, y_bar: float, bar_height: float):
        """Convert calculated bar height to actual bar height

        Args:
            y_bar (float): y cordinates of bar
            bar_height (float): bar height in pixels
            image_height (float): Image height
        """

        img = cv2.imread(self.img_path)
        image_height, _, _ = img.shape

        y_bar = y_bar * image_height
        y_min_coordinate = self.y_min_corr * image_height
        y_axis_h = self.y_axis_height * image_height

        y_diff = self.axis_y_max - self.axis_y_min

        bar_height_new = y_min_coordinate - y_bar

        y_pred = (bar_height_new / y_axis_h) * (y_diff) + self.axis_y_min

        print(y_pred)

        return round(y_pred, 2)

    def merge_x_labels(self):
        df = self.x_labels_dataframe

        # Sort the DataFrame by x_center
        df = df.sort_values(by="x_center")

        def calculate_mother_id(group):
            threshold = 0.015  # Adjust the threshold as needed
            # mother_id = random.randint(1, 500)
            unique_mother_id_list = list(range(1, 1000))
            mother_id_list = list(range(1000, 1900))
            random.shuffle(unique_mother_id_list)
            random.shuffle(mother_id_list)
            prev_x_right = group["x_right"].iloc[0]
            # unique_mother_id = random.randint(1, 500)
            unique_mother_id = unique_mother_id_list[random.randint(1, 998)]
            mother_id = mother_id_list[random.randint(3, 897)]

            if len(group) == 2:
                diff = abs(group["x_val"].iloc[1] - prev_x_right)
                if diff < threshold:
                    unique_mother_id = group["parentID"].min()
                    group["motherID"] = unique_mother_id
                else:
                    for index, row in group.iterrows():
                        random.shuffle(unique_mother_id_list)
                        unique_mother_id = unique_mother_id_list[random.randint(1, 998)]
                        group.at[index, "motherID"] = unique_mother_id

            if len(group) > 2:
                for index, row in group.iterrows():
                    if index > group.index.min():
                        diff = abs(row["x_val"] - prev_x_right)
                        if diff < threshold:
                            group.at[index, "motherID"] = unique_mother_id  # group.at[index,'parentID']
                            continue
                        else:
                            mother_id += 1
                            group.at[index, "motherID"] = mother_id
                            continue
                    prev_x_right = row["x_right"]
            elif len(group) == 1:
                group["motherID"] = mother_id

            return group

        # Apply the function to create the 'motherID' column
        df = df.groupby("parentID").apply(calculate_mother_id)

        # If parentID not in group, assign parentID to motherID
        df["motherID"].fillna(df["parentID"], inplace=True)

        df_x_labels = (
            df.groupby("motherID")
            .agg(
                {
                    "text": " ".join,
                    "width": "sum",
                    "height": "mean",
                    "x_val": "mean",
                    "y_val": "mean",
                    "x_center": "mean",
                    "y_center": "mean",
                    "x_right": "mean",
                }
            )
            .sort_values("x_center")
        )

        df_x_labels = df_x_labels.reset_index()

        print("Merged X labels:")
        print(df_x_labels)

        return df_x_labels

    def show_x_textboxes(self):
        img = cv2.imread(self.img_path)
        img_h, img_w, _ = img.shape
        df_for_x = self.x_labels_dataframe

        for i, row_x in df_for_x.iterrows():
            box_width = int(row_x["width"] * img_w)
            box_height = int(row_x["height"] * img_h)
            center_x = row_x["x_center"] * img_w
            center_y = row_x["y_center"] * img_h
            top_left = (int(center_x - box_width / 2), int(center_y - box_height / 2))
            bottom_right = (
                int(center_x + box_width / 2),
                int(center_y + box_height / 2),
            )
            img2 = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 1)

        # Add the text to the image
        plt.axis("off")
        plt.title("X labels: ")
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.savefig("x_y_labels/x_labels.png")

    def show_y_textboxes(self):
        img = cv2.imread(self.img_path)
        img_h, img_w, _ = img.shape
        df_for_y = self.y_labels_dataframe

        for i, row_y in df_for_y.iterrows():
            box_width = int(row_y["width"] * img_w)
            box_height = int(row_y["height"] * img_h)
            center_x = row_y["x_center"] * img_w
            center_y = row_y["y_center"] * img_h
            top_left = (int(center_x - box_width / 2), int(center_y - box_height / 2))
            bottom_right = (
                int(center_x + box_width / 2),
                int(center_y + box_height / 2),
            )
            img2 = cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 1)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title("Y labels: ")
        plt.savefig("x_y_labels/y_labels.png")



