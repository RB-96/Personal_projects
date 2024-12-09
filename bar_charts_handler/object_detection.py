from ultralytics import YOLO
from PIL import Image
import os
import cv2
import pandas as pd
import shutil
import matplotlib.pyplot as plt

"""
This class is used to generate the bounding box of legends and bars from the bar chart images. 
Class functions:
1. prediction(): Generate boundingboxes loading and using YOLOv8 model. The output of YOLOv8 is saved in runs folder.
2. bounding_box(): shape the bounding box values into a pandas dataframe reading the text files from runs/labels folder.
3. clean_folder(): after the bounding box dataframe is generated delete the folder using this function.
"""


class ObjectDetection:
    def __init__(self, img_path: str) -> None:
        self.img_path = img_path
        self.image = cv2.imread(img_path)
        self.H, self.W, _ = self.image.shape
        self.bounding_box_df = self.prediction()
        # print(self.bounding_box_df)

    def show_output(self):
        box_df = self.bounding_box_df
        img = cv2.imread(self.img_path)
        img_h, img_w, _ = img.shape
        for i, row in box_df.iterrows():
            if row["class"] == "0":  # for bars
                box_width = int(row["width"] * img_w)
                box_height = int(row["height"] * img_h)
                center_x = row["x_center"] * img_w
                center_y = row["y_center"] * img_h
                top_left = (
                    int(center_x - box_width / 2),
                    int(center_y - box_height / 2),
                )
                bottom_right = (
                    int(center_x + box_width / 2),
                    int(center_y + box_height / 2),
                )
                img2 = cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 3)
            elif row["class"] == "1":  # for legends
                box_width = int(row["width"] * img_w)
                box_height = int(row["height"] * img_h)
                center_x = row["x_center"] * img_w
                center_y = row["y_center"] * img_h
                top_left = (
                    int(center_x - box_width / 2),
                    int(center_y - box_height / 2),
                )
                bottom_right = (
                    int(center_x + box_width / 2),
                    int(center_y + box_height / 2),
                )
                img2 = cv2.rectangle(img, top_left, bottom_right, (32, 50, 1), 3)
        plt.axis("off")
        plt.title("Objects detected by YOLO:")
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        # os.makedirs("yolo_output", exist_ok=True)
            
        # plt.savefig("yolo_output/yolo_out.png")

    def prediction(self):
        """
        Input: chart image path
        Output: Bounding box of detected objects in AWS Rekognition format

        """
        # model = YOLO(r"C:\Users\reddi\chart_parser\model\bars_legends_best.pt")
        model = YOLO("model/bars_legends_best.pt")
        results = model.predict(source=self.image, conf=0.7, save=True, save_txt=True, batch=1)
        bb_df = self.bounding_box("runs/detect/predict/labels", self.W, self.H)
        self.clean_folder("runs")
        return bb_df

    def bounding_box(self, txt_path, width, height):
        """
        Input: text file having the prediction values from the model, image width and height
        Output: Dataframe containing all the coordinate details for bars and legends
        """
        df = pd.DataFrame(
            columns=[
                "class",
                "width",
                "height",
                "x_val",
                "y_val",
                "x_center",
                "y_center",
            ]
        )
        i = 0
        file_dir = os.listdir(txt_path)
        for files in file_dir:
            if files.endswith(".txt"):
                with open(os.path.join(txt_path, files), "r") as f:
                    lis = f.readlines()
                for item in lis:
                    li = item.split()
                    xc, yc, w, h = (
                        float(li[1]),
                        float(li[2]),
                        float(li[3]),
                        float(li[4]),
                    )
                    left = xc - (w / 2)
                    top = yc - (h / 2)
                    df.loc[i] = [li[0], w, h, left, top, xc, yc]
                    i += 1
        return df

    def clean_folder(self, folder_path):
        shutil.rmtree(folder_path)
