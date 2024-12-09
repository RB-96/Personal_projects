import pandas as pd
import numpy as np
from decimal import *
from app.server.utils.utility_functions import DataType, BarType, VerticalBarType

"""
This class is used to calculate bar values from the y labels value using object detection class bounding box.
CLass functions:
1. bar_value(): Calculating bar values for each of the bars.
2. map_bar_to_value(): Mapping each calculated values of bars to corresponding xlabels and saved into a new dataframe.
"""


class BarHeightGenerator:
    def __init__(self, object_bb, OCR_obj) -> None:
        self.MIN_PIX_DIFF = 0.05
        self.bar_bb = object_bb
        self.obj_bb = object_bb.bounding_box_df[
            object_bb.bounding_box_df["class"] == "0"
        ]  # object of ObjectDetection class
        self.ocr_df = OCR_obj  # object of TextProcessor class
        self.txt_bb = OCR_obj.merge_x_labels()
        self.presence_of_y_labels = OCR_obj.presence_of_y
        x_df = OCR_obj.x_labels_dataframe
        rest_of_texts = OCR_obj.textDetDF_copy[
            ~OCR_obj.textDetDF_copy["text"].isin(x_df["text"])
        ]
        pattern = r"^.*\d+.*[a-zA-Z].*|^\$?\d+.*[a-zA-Z].*|\d+"
        self.text_dataframe = rest_of_texts[
            rest_of_texts["text"].str.contains(pattern, na=False)
        ]
        self.text_dataframe = self.text_dataframe.sort_values("x_center")
        self.texts_from_bars_indicator = ""
        self.check_bar = ""

    @staticmethod
    def convert_currency_for_bars(text: str):
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
    def clean_bar_values(text: str):
        text = text.replace("$", "")
        text = text.replace(",", "")
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("S", "")
        return text
    
    def text_position_indicator(self, bar_x):
        processed_bars = set()
        new_rows = []
        texts_from_bars_indicator = ""
        text_dataframe = self.text_dataframe.copy()
        if self.check_bar != VerticalBarType.stk.value:
            
            for _, bar in bar_x.iterrows():
                if tuple(bar) in processed_bars:
                    continue
                y_range_inside = (bar['y_center'] - bar['height'] / 2, bar['y_center'] + bar['height'] / 2)

                inside_texts = text_dataframe[
                    (text_dataframe['x_center'] >= (bar['x_center'] - bar['width'] / 2)) &
                    (text_dataframe['x_center'] <= (bar['x_center'] + bar['width'] / 2)) &
                    (text_dataframe['y_center'] >= y_range_inside[0]) &
                    (text_dataframe['y_center'] <= y_range_inside[1])
                ]
                inside_texts['text'] = inside_texts['text'].apply(self.clean_bar_values)
                inside_texts['text'] = inside_texts['text'].apply(self.convert_currency_for_bars)
                inside_texts['text']=pd.to_numeric(inside_texts['text'], errors='coerce')
                print(inside_texts)
                # If there is an inside text, add it to the new DataFrame
                if not inside_texts.empty:
                    highest_text = inside_texts.sort_values(by='text', ascending=False).iloc[0]
                    new_rows.append(
                                    {
                                        "text": str(highest_text['text']),
                                        'x_text':bar['x_text'],
                                        'text_x_center':bar['text_x_center'], 
                                        'text_y_center':bar['text_y_center'], 
                                        'text_width': bar['text_width'],
                                        'text_height':bar['text_height'],
                                        'width': bar['width'], 
                                        'height':bar['height'], 
                                        'x_center':bar['x_center'], 
                                        'y_center':bar['y_center'], 
                                        'y_val':bar['y_val'],
                                        "parentID": int(highest_text['parentID'])
                                    }
                                    )
#                     new_rows.append({**bar, **inside_texts.iloc[0]})
                    processed_bars.add(tuple(bar))

                else:
                    y_range_above = bar['y_center'] + bar['height'] / 2

                    # Filter the text DataFrame for texts that are immediately above the bar
                    above_texts = text_dataframe[
                        (text_dataframe['x_center'] >= (bar['x_center'] - bar['width'] / 2)) &
                        (text_dataframe['x_center'] <= (bar['x_center'] + bar['width'] / 2)) &
                        (text_dataframe['y_center'] < y_range_above) #& (text_dataframe['y_center'] > y_range_above[1])
                    ]

                    print(above_texts)
                    above_texts['text'] = above_texts['text'].apply(self.clean_bar_values)
                    above_texts['text'] = above_texts['text'].apply(self.convert_currency_for_bars)
                    above_texts = above_texts.sort_values(by='y_center', ascending=False)
                    # If there is an above text, add it to the new DataFrame
                    if not above_texts.empty:
                        new_rows.append({**bar, **above_texts.iloc[0]})
                        processed_bars.add(tuple(bar))

        else:
            for _, bar in bar_x.iterrows():
                # If the bar is already processed, skip it
                if tuple(bar) in processed_bars:
                    continue

                # Calculate the vertical range inside the bar
                y_range_inside = (bar['y_center'] - bar['height'] / 2, bar['y_center'] + bar['height'] / 2)

                
                # Filter the text DataFrame for texts that are inside the bar
                inside_texts = text_dataframe[
                    (text_dataframe['x_center'] >= (bar['x_center'] - bar['width'] / 2)) &
                    (text_dataframe['x_center'] <= (bar['x_center'] + bar['width'] / 2)) &
                    (text_dataframe['y_center'] >= y_range_inside[0]) &
                    (text_dataframe['y_center'] <= y_range_inside[1])
                ]
                inside_texts['text'] = inside_texts['text'].apply(self.clean_bar_values)
                inside_texts['text'] = inside_texts['text'].apply(self.convert_currency_for_bars)
                inside_texts['text']=pd.to_numeric(inside_texts['text'], errors='coerce')
                print(inside_texts)
                # If there is an inside text, add it to the new DataFrame
                if not inside_texts.empty:
                    highest_text = inside_texts.sort_values(by='text', ascending=False).iloc[0]
                    new_rows.append(
                                    {
                                        "text": str(highest_text['text']),
                                        'x_text':bar['x_text'],
                                        'text_x_center':bar['text_x_center'], 
                                        'text_y_center':bar['text_y_center'], 
                                        'text_width': bar['text_width'],
                                        'text_height':bar['text_height'],
                                        'width': bar['width'], 
                                        'height':bar['height'], 
                                        'x_center':bar['x_center'], 
                                        'y_center':bar['y_center'], 
                                        'y_val':bar['y_val']
                                    }
                                    )
                    processed_bars.add(tuple(bar))
                    
        if len(processed_bars) >= len(bar_x) / 2:
            texts_from_bars_indicator = DataType.e.value
        else:
            texts_from_bars_indicator = DataType.c.value

        return texts_from_bars_indicator

    def check_bar_type(self):
        bars = self.map_bar_to_x_label()
        bars_copy = bars
        bars_copy = bars_copy.sort_values("x_center")
        bars_copy.reset_index(inplace=True)
        bars_copy.drop(columns=bars_copy.columns[0], axis=1)
        bars_copy["x_center"] = bars_copy["x_center"].apply(lambda x: np.round(x, 3))

        if bars_copy["x_text"].duplicated().any():
            for i in range(len(bars_copy) - 1):
                if (
                    abs(bars_copy.loc[i, "x_center"] - bars_copy.loc[i + 1, "x_center"])
                    <= 0.002
                    and bars_copy.loc[i, "x_text"] == bars_copy.loc[i + 1, "x_text"]
                ):
                    self.check_bar = VerticalBarType.stk.value
                elif (
                    abs(bars_copy.loc[i, "x_center"] - bars_copy.loc[i + 1, "x_center"])
                    > 0.002
                    and bars_copy.loc[i, "x_text"] == bars_copy.loc[i + 1, "x_text"]
                ):
                    self.check_bar = VerticalBarType.grp.value
                else:
                    self.check_bar = VerticalBarType.sim.value
                    # continue
        else:
            self.check_bar = VerticalBarType.sim.value
        print(self.check_bar)
        return self.check_bar
    
    def append_bar_value(self):
        def get_pix_diff(a1, a2):
            return abs(float(a1 - a2))
        image_height = 1
        bar_h = []
        bars = self.map_bar_to_x_label()
        self.check_bar = self.check_bar_type()
        print(self.check_bar)
        self.texts_from_bars_indicator = self.text_position_indicator(bars)
        print(self.texts_from_bars_indicator)
        
        if self.texts_from_bars_indicator == DataType.c.value:
            if self.presence_of_y_labels == "yes":
                for i, row in bars.iterrows():
                    print("Printing Y val")
                    print(row["y_val"])
                    bar_h.append(
                        {
                            "text": row["x_text"],
                            "text_x_center": row["x_center"],
                            "text_y_center": row["y_center"],
                            "text_width": row["width"],
                            "text_height": row["height"],
                            "width": row["width"],
                            "height": row["height"],
                            "x_center": row["x_center"],
                            "y_center": row["y_center"],
                            "bar_value": self.ocr_df.get_actual_bar_height(
                                row["y_val"], row["height"]
                            ),
                        }
                    )

                bar_h = pd.DataFrame(bar_h)
                self.texts_from_bars_indicator = DataType.c.value

            else:
                self.text_dataframe["status"] = "False"
                new_rows = []
                for _, bar in bars.iterrows():
                    min_dist = 10000
                    close_text = ""
                    for i, row in self.text_dataframe.iterrows():
                        min_distance = get_pix_diff(bar["x_center"], row["x_center"])
                        print(min_distance)
                        if min_distance <= min_dist and row["status"] == "False":
                            min_dist = min_distance
                            close_text = row["text"]
                            print(row["x_center"])
                            position = i
                    new_rows.append(
                        {
                            "bar_value": close_text,
                            "x_text": bar["x_text"],
                            "text_x_center": bar["text_x_center"],
                            "text_y_center": bar["text_y_center"],
                            "text_width": bar["text_width"],
                            "text_height": bar["text_height"],
                            "width": bar["width"],
                            "height": bar["height"],
                            "x_center": bar["x_center"],
                            "y_center": bar["y_center"],
                            "y_val": bar["y_val"],
                        }
                    )
                    self.text_dataframe.at[position, "status"] = "True"
                bar_h = pd.DataFrame(new_rows)
                bar_h["bar_value"] = bar_h["bar_value"].apply(self.clean_bar_values)
                bar_h["bar_value"] = bar_h["bar_value"].apply(
                    self.convert_currency_for_bars
                )
            bar_h = bar_h.rename(columns={"x_text": "text"})
            return (bar_h, self.texts_from_bars_indicator)
        elif self.texts_from_bars_indicator == DataType.e.value:
            return ("", self.texts_from_bars_indicator)     


    def map_bar_to_x_label(self):
        
        abs_diff = abs(
            self.obj_bb["x_center"].values[:, None] - self.txt_bb["x_center"].values
        )

        min_idx = abs_diff.argmin(axis=1)

        bar_x = self.obj_bb.copy()
        bar_x[
            ["x_text", "text_width", "text_height", "text_x_center", "text_y_center"]
        ] = self.txt_bb.loc[
            min_idx, ["text", "width", "height", "x_center", "y_center"]
        ].values

        # print(bar_x)
        return bar_x
