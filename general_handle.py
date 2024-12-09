"""
This code works for all verticl bar charts and segmented bar charts where legend is either in the right side or 
bottom side of chart.
stacked_data_response_formatter(): For generating the response of the stacked chart which produces three dataframes
one for the data and another twos for the coordinates of bars and legends. I have added the bar coordinates in the
response data now. 
"""

from bar_charts_handler.object_detection import ObjectDetection
from chart_detection_with_OCR.utils import TextProcessor
from bar_charts_handler.bar_to_text import BarHeightGenerator
from bar_charts_handler.legend_to_text import Legendmap
import cv2
import pandas as pd
import numpy as np
import uuid
from decimal import *
from app.server.utils.utility_functions import (
    DataType,
    ModelUsed,
    VerticalBarType,
    BarType,
    LegendPosition,
)


class Chart_handler:
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, img_path, es_id, log_data):
        self.img_path = img_path
        self.img = cv2.imread(self.img_path)
        self.textprocessor = TextProcessor(self.img_path, es_id, log_data)
        self.OD = ObjectDetection(img_path)
        self.bval = BarHeightGenerator(self.OD, self.textprocessor)
        self.legend = Legendmap(
            img_path, self.OD, self.textprocessor, self.bval, es_id, log_data
        )

    def simple_chart_subtype(self):
        df = self.OD.bounding_box_df[self.OD.bounding_box_df["class"] == "0"]
        class_type = ""
        # Update class_type based on the condition
        if (df["width"] > df["height"]).all():
            class_type = BarType.hor.value
        elif (df["width"] < df["height"]).sum() >= 2:
            class_type = BarType.ver.value
        else:
            class_type = BarType.und.value
        print(class_type)
        return class_type

    def extract_title(self, df_titles):
        df_titles_sorted = (
            df_titles.groupby(["parentID"])
            .agg(
                {
                    "text": " ".join,
                    "width": "sum",
                    "height": "mean",
                    "x_center": "mean",
                    "y_center": "mean",
                }
            )
            .sort_values(by="y_center")
        )
        threshold_y = 0.09
        threshold_x = 0.1

        df_titles_sorted["y_center_diff"] = df_titles_sorted["y_center"].diff().abs()
        df_titles_sorted["x_center_diff"] = df_titles_sorted["x_center"].diff().abs()

        filtered_rows = df_titles_sorted[
            (df_titles_sorted["y_center_diff"] <= threshold_y)
            | (df_titles_sorted["y_center_diff"].isna())
        ]
        filtered_rows_x = filtered_rows[
            (filtered_rows["x_center_diff"] <= threshold_x)
            | (df_titles_sorted["x_center_diff"].isna())
        ]

        title_ = " ".join(filtered_rows_x["text"])

        return title_

    def chart_values_extract(
        self, img_path, chart_h, chart_w, chart_l, chart_t, es_id, log_data
    ):
        self.initialize_parameters(img_path, es_id, log_data)
        self.chart_height = chart_h
        self.chart_width = chart_w
        self.chart_left = chart_l
        self.chart_top = chart_t

        (b_values, bar_value_type) = self.bval.append_bar_value()
        c_type = self.bval.check_bar

        if bar_value_type == DataType.e.value:
            return ("", c_type, "", "", bar_value_type)

        elif bar_value_type == DataType.c.value:
            bars = (
                b_values.groupby(["width", "height", "x_center", "y_center"])
                .agg(
                    {
                        "text": " ".join,
                        "bar_value": "first",
                        "text_width": "sum",
                        "text_height": "mean",
                        "text_x_center": "mean",
                        "text_y_center": "mean",
                    }
                )
                .sort_values("x_center")
            )
            bars = bars.reset_index()

            if c_type == VerticalBarType.stk.value:
                l_pos, c_type, final_df = self.stacked_chart(c_type)
                print(l_pos)
                print(c_type)
                print(final_df)
                return (l_pos, c_type, final_df, "", bar_value_type)
            elif c_type == VerticalBarType.grp.value:
                l_pos, c_type, final_df = self.grouped_chart(c_type)
                print(l_pos)
                print(c_type)
                print(final_df)
                return (l_pos, c_type, final_df, "", bar_value_type)
            elif c_type == VerticalBarType.sim.value:
                # Extract title
                df_remaining = self.textprocessor.get_texts_not_x_and_y_labels()
                df_titles = df_remaining
                chart_title = self.extract_title(df_titles)
                print(f"chart title picked: {chart_title}")

                sub_type = self.simple_chart_subtype()
                l_pos = LegendPosition.err.value
                if sub_type == BarType.ver.value:
                    c_type, final_df = self.simple_chart(c_type, bars)
                    print(l_pos)
                    print(c_type)
                    print(final_df)
                    return (l_pos, c_type, final_df, chart_title, bar_value_type)
                elif sub_type == BarType.hor.value or sub_type == BarType.und.value:
                    return (l_pos, sub_type, None, "", "")

    def stacked_chart(self, c_type):
        l_pos, final_df = self.legend.stacked_bar_values()
        return l_pos, c_type, final_df

    def grouped_chart(self, c_type):
        l_pos, final_df = self.legend.grouped_bar_values()
        return l_pos, c_type, final_df

    def simple_chart(self, c_type, bars):
        bars_final_df = bars.copy()
        bars_final_df["left"] = bars_final_df["x_center"] - bars_final_df["width"] / 2
        bars_final_df["top"] = bars_final_df["y_center"] - bars_final_df["height"] / 2
        bars_final_df["text_left"] = (
            bars_final_df["text_x_center"] - bars_final_df["text_width"] / 2
        )
        bars_final_df["text_top"] = (
            bars_final_df["text_y_center"] - bars_final_df["text_height"] / 2
        )

        bar_left_list = []
        bar_top_list = []
        text_left_list = []
        text_top_list = []

        for i in range(len(bars_final_df)):
            bars_final_df.at[i, "width"] = int(
                self.chart_width + bars_final_df.at[i, "width"] * self.chart_width
            )
            bars_final_df.at[i, "height"] = int(
                self.chart_height + bars_final_df.at[i, "height"] * self.chart_height
            )
            bars_final_df.at[i, "x_center"] = int(
                self.chart_width + bars_final_df.at[i, "x_center"] * self.chart_width
            )
            bars_final_df.at[i, "y_center"] = int(
                self.chart_height + bars_final_df.at[i, "y_center"] * self.chart_height
            )

            bar_left = int(
                bars_final_df.at[i, "x_center"] - bars_final_df.at[i, "width"] / 2
            )
            bar_top = int(
                bars_final_df.at[i, "y_center"] - bars_final_df.at[i, "height"] / 2
            )
            bar_top_list.append(bar_top)
            bar_left_list.append(bar_left)
            bars_final_df.at[i, "text_width"] = int(
                self.chart_width + bars_final_df.at[i, "text_width"] * self.chart_width
            )
            bars_final_df.at[i, "text_height"] = int(
                self.chart_height
                + bars_final_df.at[i, "text_height"] * self.chart_height
            )
            bars_final_df.at[i, "text_x_center"] = int(
                self.chart_width
                + bars_final_df.at[i, "text_x_center"] * self.chart_width
            )
            bars_final_df.at[i, "text_y_center"] = int(
                self.chart_height
                + bars_final_df.at[i, "text_y_center"] * self.chart_height
            )
            text_left = int(
                bars_final_df.at[i, "text_x_center"]
                - bars_final_df.at[i, "text_width"] / 2
            )
            text_top = int(
                bars_final_df.at[i, "text_y_center"]
                - bars_final_df.at[i, "text_height"] / 2
            )
            text_left_list.append(text_left)
            text_top_list.append(text_top)

        bars_final_df["top"] = bar_top_list
        bars_final_df["left"] = bar_left_list
        bars_final_df["text_left"] = text_left_list
        bars_final_df["text_top"] = text_top_list

        print(bars_final_df)

        return c_type, bars_final_df

    def stacked_and_grouped_data_response_formatter(self, df_test):
        def clean_coordinate_point_table(df1, df2, df3, df4):
            df = pd.concat([df1, df2, df3, df4], axis=1)
            duplicate_columns = df.columns[df.columns.duplicated()]
            for index, row in df.iterrows():
                for col_name in duplicate_columns:
                    df[col_name + "_combined"] = df.apply(
                        lambda row: list(row[col_name]), axis=1
                    )
            columns_to_keep = [col for col in df.columns if col.endswith("_combined")]
            df.drop(df.columns[~df.columns.isin(columns_to_keep)], axis=1, inplace=True)
            return df

        # Flatten the DataFrame
        df_test.columns = ["_".join(col) for col in df_test.columns]
        df_test.reset_index(drop=True, inplace=True)
        # final data-value dataframe
        final_df = df_test[
            ["bar_name_"]
            + [col for col in df_test.columns if col.startswith("stack_value_")]
        ]
        final_df.columns = final_df.columns.str.replace("stack_value_", "")
        final_df.columns = final_df.columns.str.replace("bar_name_", "bar_name")

        # legend and bars
        final_legend = df_test[
            [col for col in df_test.columns if col.startswith("legend_normalized_")]
        ]
        final_legend.columns = final_legend.columns.str.replace(
            "legend_normalized_", ""
        )
        final_bars = df_test[
            [col for col in df_test.columns if col.startswith("bar_normalized_")]
        ]
        final_bars.columns = final_bars.columns.str.replace("bar_normalized_", "")

        # texts ocr coordinate dataframe
        final_txt = df_test[
            [col for col in df_test.columns if col.startswith("text_normalized_")]
        ]
        final_txt.columns = final_txt.columns.str.replace("text_normalized_", "")

        bars_x = final_bars[
            [col for col in final_bars.columns if col.startswith("x_center")]
        ]
        bars_y = final_bars[
            [col for col in final_bars.columns if col.startswith("y_center")]
        ]
        bars_w = final_bars[
            [col for col in final_bars.columns if col.startswith("width")]
        ]
        bars_h = final_bars[
            [col for col in final_bars.columns if col.startswith("height")]
        ]
        bars_x.columns = bars_x.columns.str.replace("x_center_", "")
        bars_y.columns = bars_y.columns.str.replace("y_center_", "")
        bars_w.columns = bars_w.columns.str.replace("width_", "")
        bars_h.columns = bars_h.columns.str.replace("height_", "")
        bars_left = (bars_x - (bars_w / 2)).astype(int)
        bars_top = (bars_y - (bars_h / 2)).astype(int)

        txt_x = final_txt[
            [col for col in final_txt.columns if col.startswith("x_center")]
        ]
        txt_y = final_txt[
            [col for col in final_txt.columns if col.startswith("y_center")]
        ]
        txt_w = final_txt[[col for col in final_txt.columns if col.startswith("width")]]
        txt_h = final_txt[
            [col for col in final_txt.columns if col.startswith("height")]
        ]
        txt_x.columns = txt_x.columns.str.replace("x_center_", "")
        txt_y.columns = txt_y.columns.str.replace("y_center_", "")
        txt_w.columns = txt_w.columns.str.replace("width_", "")
        txt_h.columns = txt_h.columns.str.replace("height_", "")
        txt_left = (txt_x - (txt_w / 2)).astype(int)
        txt_top = (txt_y - (txt_h / 2)).astype(int)

        legend_x = final_legend[
            [col for col in final_legend.columns if col.startswith("x_center")]
        ]
        legend_y = final_legend[
            [col for col in final_legend.columns if col.startswith("y_center")]
        ]
        legend_w = final_legend[
            [col for col in final_legend.columns if col.startswith("width")]
        ]
        legend_h = final_legend[
            [col for col in final_legend.columns if col.startswith("height")]
        ]
        legend_x.columns = legend_x.columns.str.replace("x_center_", "")
        legend_y.columns = legend_y.columns.str.replace("y_center_", "")
        legend_w.columns = legend_w.columns.str.replace("width_", "")
        legend_h.columns = legend_h.columns.str.replace("height_", "")
        legend_h.drop_duplicates(inplace=True)
        legend_w.drop_duplicates(inplace=True)
        legend_x.drop_duplicates(inplace=True)
        legend_y.drop_duplicates(inplace=True)
        legend_left = (legend_x - (legend_w / 2)).astype(int)
        legend_top = (legend_y - (legend_h / 2)).astype(int)

        df_bars = clean_coordinate_point_table(bars_w, bars_h, bars_top, bars_left)
        df_legends = clean_coordinate_point_table(
            legend_w, legend_h, legend_top, legend_left
        )
        df_texts = clean_coordinate_point_table(txt_w, txt_h, txt_top, txt_left)

        return final_df, df_bars, df_legends, df_texts

    def format_decimal(self, value):
        decimal_value = Decimal(value)
        normalized_value = decimal_value.normalize()
        return normalized_value

    def extract_number_from_text(self, text):
        try:
            float(text)
            return [float(text)]
        except ValueError:
            return []

    def create_headers(self, ocr_result):
        num_col = ocr_result.shape[1]
        headers = []
        for i in range(0, num_col - 1):
            header_dict = {
                "version": "Actual",
                "headerText": "Ended June 30, 2022",
                "dateLabel": "2022-06-30",
                "periodType": "SemiAnnual",
                "currency": "USD",
                "units": "Thousands",
                "headerType": "dateLabel",
            }
            headers.append(header_dict)

        return headers

    def get_cell_dict(self, row, col, text, width, height, left, top, page):
        cell_dict = {
            "cell_id": str(uuid.uuid4()),
            "row": row,
            "col": col,
            "row_span": 1,
            "col_span": 1,
            "score": 0.91002023,
            "page": page,
            "text": str(text),
            "Width": int(width),
            "Height": int(height),
            "Left": int(left),
            "Top": int(top),
            "number_in_text": self.extract_number_from_text(text),
        }
        return cell_dict

    def create_tabular_data_simple(
        self, ocr_result, coordinates, page, width, height, left, top
    ):
        print(ocr_result)

        cords_simple = pd.DataFrame(
            {
                "texts": coordinates[
                    ["text_width", "text_height", "text_top", "text_left"]
                ].values.tolist(),
                "bars": coordinates[["width", "height", "top", "left"]].values.tolist(),
            }
        )
        cords_simple = cords_simple.T
        cords_simple.insert(0, "new_column", [[0, 0, 0, 0], [0, 0, 0, 0]])

        cells = []
        row_ = 1
        col_ = 1

        for (idx, rows), (inx, cords) in zip(
            ocr_result.iterrows(), cords_simple.iterrows()
        ):
            col_ = 1
            for (_, value), (_, cord_item) in zip(rows.items(), cords.items()):
                width1 = cord_item[0]
                height1 = cord_item[1]
                top1 = cord_item[2] + top
                left1 = cord_item[3] + left
                cells.append(
                    self.get_cell_dict(
                        row_, col_, value, width, height, left, top, page
                    )
                )
                col_ += 1
            row_ += 1
        return cells

    def create_tabular_data_stacked_and_grouped(
        self,
        ocr_result,
        bars_coordinates,
        text_coordinates,
        legends_coordinates,
        page,
        width,
        height,
        left,
        top,
    ):

        print(ocr_result)
        selected_row = None
        for i in text_coordinates.index:
            if not any(0 in sublist for sublist in text_coordinates.loc[i]):
                selected_row = i
                break
        bars_coordinates = bars_coordinates.T

        cells = []
        row_ = 1
        col_ = 1
        cells.append(
            self.get_cell_dict(row_, col_, "Attributes", width, height, left, top, page)
        )

        col_ = 2

        for colmn in ocr_result.columns:
            cells.append(
                self.get_cell_dict(row_, col_, colmn, width, height, left, top, page)
            )
            col_ += 1
        row_ = 2
        for (idx, rows), (inx, cords) in zip(
            ocr_result.iterrows(), bars_coordinates.iterrows()
        ):
            col_ = 1
            cells.append(
                self.get_cell_dict(row_, col_, idx, width, height, left, top, page)
            )
            col_ += 1
            for (_, value), (_, cord_item) in zip(rows.items(), cords.items()):
                width1 = cord_item[0]
                height1 = cord_item[1]
                top1 = cord_item[2] + top
                left1 = cord_item[3] + left
                cells.append(
                    self.get_cell_dict(
                        row_, col_, value, width, height, left, top, page
                    )
                )
                col_ += 1
            row_ += 1

        return cells

    def create_tabular_data_stacked_and_grouped_no_legend(
        self, ocr_result, page, width, height, left, top
    ):
        print(ocr_result)
        cells = []
        row_ = 1
        col_ = 1

        for idx, rows in ocr_result.iterrows():
            col_ = 1
            for _, value in rows.items():
                cells.append(
                    self.get_cell_dict(
                        row_, col_, value, width, height, left, top, page
                    )
                )
                col_ += 1
            row_ += 1
        return cells

    def gpt_create_tabular_data(self, ocr_result, page, width, height, left, top):
        print(ocr_result)

        cells = []
        row_ = 0
        col_ = 1
        for idx, rows in ocr_result.iterrows():
            col_ = 1
            cells.append(
                self.get_cell_dict(row_, col_, idx, width, height, left, top, page)
            )
            col_ += 1
            for _, value in rows.items():
                cells.append(
                    self.get_cell_dict(
                        row_, col_, value, width, height, left, top, page
                    )
                )
                col_ += 1
            row_ += 1
        return cells

    def gpt_create_tabular_data_pie(self, ocr_result, page, width, height, left, top):
        print("Chart Data: ")
        print(ocr_result)

        cells = []
        row_ = 1
        col_ = 1
        for colmn, _ in ocr_result.iloc[0].items():
            cell_val = colmn
            cells.append(
                self.get_cell_dict(row_, col_, cell_val, width, height, left, top, page)
            )
            col_ += 1
        row_ = 2
        for i, row in ocr_result.iterrows():
            col_ = 1
            for _, value in row.items():
                cells.append(
                    self.get_cell_dict(
                        row_, col_, value, width, height, left, top, page
                    )
                )
                col_ += 1
            row_ += 1

        return cells

    def chart_reponse(self, final_charts, file_name):
        tab_data = {
            "result": [],
        }
        res = 0
        for page_no, group in final_charts.groupby("page_no"):
            tab_data["result"].append(
                {"input": file_name, "prediction": [], "page": page_no}
            )
            pred = 0
            for i, rows_ in group.iterrows():
                c_width = rows_["chart_width"]
                c_height = rows_["chart_height"]
                c_left = rows_["chart_left"]
                c_top = rows_["chart_top"]

                if rows_["chart_class"] == "BarGraph":

                    model_used = ""
                    ocr_result = rows_["data"]
                    if rows_["c_type"] == VerticalBarType.sim.value:

                        if rows_["data_type"] == DataType.c.value:

                            tab_data["result"][res]["prediction"].append(
                                {
                                    "table_id": str(uuid.uuid4()),
                                    "page_no": page_no,
                                    "Width": rows_["chart_width"],
                                    "Height": rows_["chart_height"],
                                    "Left": rows_["chart_left"],
                                    "Top": rows_["chart_top"],
                                    "width_norm": rows_["chart_width_norm"],
                                    "height_norm": rows_["chart_height_norm"],
                                    "x_left_norm": rows_["chart_left_norm"],
                                    "y_top_norm": rows_["chart_top_norm"],
                                    "cells": [],
                                }
                            )
                            chart_title = rows_["chart_title"]
                            bars_df = ocr_result.loc[:, ["text", "bar_value"]]
                            coordinate_df = ocr_result.loc[
                                :,
                                [
                                    "width",
                                    "height",
                                    "top",
                                    "left",
                                    "text_width",
                                    "text_height",
                                    "text_top",
                                    "text_left",
                                ],
                            ]
                            missing_values = [
                                np.nan,
                                None,
                                "NA",
                                "N/A",
                                "NaN",
                                "nan",
                                " ",
                            ]
                            bars_df["bar_value"].replace(
                                missing_values, np.nan, inplace=True
                            )
                            bars_df["bar_value"].fillna(0, inplace=True)
                            bars_df = bars_df.T
                            bars_df.insert(0, "new_column", ["Attributes", chart_title])
                            
                            print(bars_df)

                            cells = self.create_tabular_data_simple(
                                bars_df,
                                coordinate_df,
                                page_no,
                                c_width,
                                c_height,
                                c_left,
                                c_top,
                            )

                            try:
                                tab_data["result"][res]["prediction"][pred][
                                    "cells"
                                ] = cells
                            except:
                                pass

                            tab_data["result"][res]["prediction"][pred][
                                "chart_type"
                            ] = rows_["c_type"]
                            tab_data["result"][res]["prediction"][pred][
                                "model_used"
                            ] = ModelUsed.inh.value
                            tab_data["result"][res]["prediction"][pred][
                                "data_type"
                            ] = DataType.c.value

                        elif rows_["data_type"] == DataType.e.value:
                            tab_data["result"][res]["prediction"].append(
                                {
                                    "table_id": str(uuid.uuid4()),
                                    "page_no": page_no,
                                    "Width": rows_["chart_width"],
                                    "Height": rows_["chart_height"],
                                    "Left": rows_["chart_left"],
                                    "Top": rows_["chart_top"],
                                    "width_norm": rows_["chart_width_norm"],
                                    "height_norm": rows_["chart_height_norm"],
                                    "x_left_norm": rows_["chart_left_norm"],
                                    "y_top_norm": rows_["chart_top_norm"],
                                    "cells": [],
                                }
                            )
                            ocr_result = rows_["data"]
                            ocr_result = ocr_result.reset_index()
                            ocr_df = ocr_result.T
                            cells = self.gpt_create_tabular_data(
                                ocr_df, page_no, c_width, c_height, c_left, c_top
                            )
                            try:
                                tab_data["result"][res]["prediction"][pred][
                                    "cells"
                                ] = cells
                            except KeyError as e:
                                print(e)
                                pass
                            tab_data["result"][res]["prediction"][pred][
                                "chart_type"
                            ] = rows_["c_type"]
                            tab_data["result"][res]["prediction"][pred][
                                "model_used"
                            ] = ModelUsed.llm.value
                            tab_data["result"][res]["prediction"][pred][
                                "data_type"
                            ] = DataType.e.value

                    elif (
                        rows_["c_type"] == BarType.hor.value
                        or rows_["c_type"] == BarType.und.value
                    ):

                        tab_data["result"][res]["prediction"].append(
                            {
                                "table_id": str(uuid.uuid4()),
                                "page_no": page_no,
                                "Width": rows_["chart_width"],
                                "Height": rows_["chart_height"],
                                "Left": rows_["chart_left"],
                                "Top": rows_["chart_top"],
                                "width_norm": rows_["chart_width_norm"],
                                "height_norm": rows_["chart_height_norm"],
                                "x_left_norm": rows_["chart_left_norm"],
                                "y_top_norm": rows_["chart_top_norm"],
                                "cells": [],
                            }
                        )
                        ocr_result = rows_["data"]
                        ocr_result = ocr_result.reset_index()
                        ocr_df = ocr_result.T
                        cells = self.gpt_create_tabular_data(
                            ocr_df, page_no, c_width, c_height, c_left, c_top
                        )
                        try:
                            tab_data["result"][res]["prediction"][pred]["cells"] = cells
                        except KeyError as e:
                            print(e)
                            pass
                        tab_data["result"][res]["prediction"][pred]["chart_type"] = (
                            rows_["c_type"]
                        )
                        tab_data["result"][res]["prediction"][pred][
                            "model_used"
                        ] = ModelUsed.llm.value

                    elif (
                        rows_["c_type"] == VerticalBarType.stk.value
                        or rows_["c_type"] == VerticalBarType.grp.value
                    ):

                        if rows_["data_type"] == DataType.c.value:

                            if rows_["l_pos"] != LegendPosition.err.value:
                                tab_data["result"][res]["prediction"].append(
                                    {
                                        "table_id": str(uuid.uuid4()),
                                        "page_no": page_no,
                                        "Width": rows_["chart_width"],
                                        "Height": rows_["chart_height"],
                                        "Left": rows_["chart_left"],
                                        "Top": rows_["chart_top"],
                                        "width_norm": rows_["chart_width_norm"],
                                        "height_norm": rows_["chart_height_norm"],
                                        "x_left_norm": rows_["chart_left_norm"],
                                        "y_top_norm": rows_["chart_top_norm"],
                                        "cells": [],
                                    }
                                )
                                try:
                                    (
                                        final_data,
                                        bars_coordinates,
                                        legends_coordinates,
                                        text_coordinates,
                                    ) = self.stacked_and_grouped_data_response_formatter(
                                        ocr_result
                                    )
                                    final_data.set_index(keys="bar_name", inplace=True)
                                    final_data = final_data.T
                                    print(final_data)

                                except KeyError as e:
                                    print(e)
                                    pass

                                cells = self.create_tabular_data_stacked_and_grouped(
                                    final_data,
                                    bars_coordinates,
                                    text_coordinates,
                                    legends_coordinates,
                                    page_no,
                                    c_width,
                                    c_height,
                                    c_left,
                                    c_top,
                                )
                                try:
                                    tab_data["result"][res]["prediction"][pred][
                                        "cells"
                                    ] = cells
                                except:
                                    pass
                                tab_data["result"][res]["prediction"][pred][
                                    "chart_type"
                                ] = rows_["c_type"]
                                print("cells created in response")
                                tab_data["result"][res]["prediction"][pred][
                                    "model_used"
                                ] = ModelUsed.inh.value
                                tab_data["result"][res]["prediction"][pred][
                                    "data_type"
                                ] = DataType.c.value

                            else:
                                tab_data["result"][res]["prediction"].append(
                                    {
                                        "table_id": str(uuid.uuid4()),
                                        "page_no": page_no,
                                        "Width": rows_["chart_width"],
                                        "Height": rows_["chart_height"],
                                        "Left": rows_["chart_left"],
                                        "Top": rows_["chart_top"],
                                        "width_norm": rows_["chart_width_norm"],
                                        "height_norm": rows_["chart_height_norm"],
                                        "x_left_norm": rows_["chart_left_norm"],
                                        "y_top_norm": rows_["chart_top_norm"],
                                        "cells": [],
                                    }
                                )
                                no_legend_data = ocr_result[["text", "stack_value"]].T
                                no_legend_data.insert(
                                    0, "new_column", ["Attributes", "-"]
                                )
                                cells = self.create_tabular_data_stacked_and_grouped_no_legend(
                                    no_legend_data,
                                    page_no,
                                    c_width,
                                    c_height,
                                    c_left,
                                    c_top,
                                )
                                try:
                                    tab_data["result"][res]["prediction"][pred][
                                        "cells"
                                    ] = cells
                                except:
                                    pass

                                tab_data["result"][res]["prediction"][pred][
                                    "chart_type"
                                ] = rows_["c_type"]
                                tab_data["result"][res]["prediction"][pred][
                                    "model_used"
                                ] = ModelUsed.inh.value
                                tab_data["result"][res]["prediction"][pred][
                                    "data_type"
                                ] = DataType.c.value

                        elif rows_["data_type"] == DataType.e.value:
                            tab_data["result"][res]["prediction"].append(
                                {
                                    "table_id": str(uuid.uuid4()),
                                    "page_no": page_no,
                                    "Width": rows_["chart_width"],
                                    "Height": rows_["chart_height"],
                                    "Left": rows_["chart_left"],
                                    "Top": rows_["chart_top"],
                                    "width_norm": rows_["chart_width_norm"],
                                    "height_norm": rows_["chart_height_norm"],
                                    "x_left_norm": rows_["chart_left_norm"],
                                    "y_top_norm": rows_["chart_top_norm"],
                                    "cells": [],
                                }
                            )
                            ocr_result = rows_["data"]
                            ocr_result = ocr_result.reset_index()
                            ocr_df = ocr_result.T
                            cells = self.gpt_create_tabular_data(
                                ocr_df, page_no, c_width, c_height, c_left, c_top
                            )
                            try:
                                tab_data["result"][res]["prediction"][pred][
                                    "cells"
                                ] = cells
                            except KeyError as e:
                                print(e)
                                pass
                            tab_data["result"][res]["prediction"][pred][
                                "chart_type"
                            ] = rows_["c_type"]
                            tab_data["result"][res]["prediction"][pred][
                                "model_used"
                            ] = ModelUsed.llm.value
                            tab_data["result"][res]["prediction"][pred][
                                "data_type"
                            ] = DataType.e.value

                else:
                    if rows_["chart_class"] != "PieChart":

                        tab_data["result"][res]["prediction"].append(
                            {
                                "table_id": str(uuid.uuid4()),
                                "page_no": page_no,
                                "Width": rows_["chart_width"],
                                "Height": rows_["chart_height"],
                                "Left": rows_["chart_left"],
                                "Top": rows_["chart_top"],
                                "width_norm": rows_["chart_width_norm"],
                                "height_norm": rows_["chart_height_norm"],
                                "x_left_norm": rows_["chart_left_norm"],
                                "y_top_norm": rows_["chart_top_norm"],
                                "cells": [],
                            }
                        )
                        ocr_result = rows_["data"]
                        ocr_result = ocr_result.reset_index()
                        ocr_df = ocr_result.T
                        cells = self.gpt_create_tabular_data(
                            ocr_df, page_no, c_width, c_height, c_left, c_top
                        )
                        try:
                            tab_data["result"][res]["prediction"][pred]["cells"] = cells

                        except:
                            pass

                        tab_data["result"][res]["prediction"][pred]["chart_type"] = (
                            rows_["chart_class"]
                        )
                        tab_data["result"][res]["prediction"][pred][
                            "model_used"
                        ] = ModelUsed.llm.value

                    elif rows_["chart_class"] == "PieChart":
                        tab_data["result"][res]["prediction"].append(
                            {
                                "table_id": str(uuid.uuid4()),
                                "page_no": page_no,
                                "Width": rows_["chart_width"],
                                "Height": rows_["chart_height"],
                                "Left": rows_["chart_left"],
                                "Top": rows_["chart_top"],
                                "width_norm": rows_["chart_width_norm"],
                                "height_norm": rows_["chart_height_norm"],
                                "x_left_norm": rows_["chart_left_norm"],
                                "y_top_norm": rows_["chart_top_norm"],
                                "cells": [],
                            }
                        )
                        ocr_result = rows_["data"]
                        ocr_result = ocr_result.reset_index()
                        ocr_df = ocr_result.T

                        cells = self.gpt_create_tabular_data_pie(
                            ocr_result, page_no, c_width, c_height, c_left, c_top
                        )
                        try:
                            tab_data["result"][res]["prediction"][pred]["cells"] = cells

                        except:
                            pass
                        tab_data["result"][res]["prediction"][pred]["chart_type"] = (
                            rows_["chart_class"]
                        )
                        tab_data["result"][res]["prediction"][pred][
                            "model_used"
                        ] = ModelUsed.llm.value

                pred += 1

            res += 1

        comp_data = {
            "tabular_data": tab_data,
        }

        return tab_data
