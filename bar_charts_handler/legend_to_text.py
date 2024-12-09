import pandas as pd
import numpy as np
import cv2
import sys
import elastic_logging
from app.server.utils.utility_functions import LegendPosition, DataType

"""
This class represents the legends and the corresponding texts beside the legend.
Class fucntions:
1. check_xandy_labels(): This is used to check if there is any common value passed to both x and y label dataframe.
Situation like this happens if there is some common term in the (0,0) position. 
2. legend_generator(): generates a dataframe that only contains legends and axis titles from the textDetDF.
3. legend_to_text(): append corresponding texts to the legend by considering min_distance of text box.
4. legend_alignment(): this fucntion gives us the alignment of legends i.e. if all the legends positioned horizontally
or vertically.
5. legend_roi_color() & bar_roi_color(): these functions creates dataframe for legends and bars and stores color values
for each bounding box repectively.
6. legend_to_bar_map(): mapping legend_color_list and bar_color_list to create the final dataframe with bars and corresponding
legends and the value calculated for each stack.
7. stacked_bar_values(): calculating the values for each of the stack and add to the final dataframe. 
"""


class Legendmap:
    def __init__(self, image_path, object_bb, ocr_bb, bval, es_id, log_data) -> None:
        """
        initiating with image path, Objects of classes ObjectDetection and TextProcessor, bars with values mapped
        """
        self.ocr_df = ocr_bb.textDetDF_copy
        self.y_df = ocr_bb.find_y_dataframe()
        self.x_df = ocr_bb.x_labels_dataframe
        self.obj_bb = object_bb.bounding_box_df
        self.image_path = image_path
        self.bar_df, self.value_indicator = bval.append_bar_value()
        self.legend_data = ocr_bb.get_texts_not_x_and_y_labels()
        self.es_id = es_id
        self.log_data = log_data

    def legend_alignment(self):
        legend_array = self.obj_bb.loc[self.obj_bb["class"] == "1"].to_numpy()
        x_dist = 0
        y_dist = 0
        for i in range(0, len(legend_array)):
            if i + 1 > len(legend_array) - 1:
                break
            else:
                x_dist += abs(float(legend_array[i][5] - legend_array[i + 1][5]))
                y_dist += abs(float(legend_array[i][6] - legend_array[i + 1][6]))
        try:
            if (x_dist / y_dist) > 1:
                return LegendPosition.hor.value
            elif (x_dist / y_dist) < 1:
                return LegendPosition.ver.value
        except:
            return LegendPosition.err.value

    def legend_to_text(self):
        def get_pix_diff(a1, a2):
            return abs(float(a1 - a2))

        potential_legend = []
        legend_data = self.legend_data
        print("Legend_data")
        print(legend_data)
        legends = legend_data.groupby(["parentID"]).agg(
            {
                "text": " ".join,
                "width": "sum",
                "height": "mean",
                "x_val": "mean",
                "y_val": "mean",
                "x_center": "mean",
                "y_center": "mean",
            }
        )

        legend_position = self.legend_alignment()
        if legend_position == LegendPosition.err.value:
            print("Legend can not be extracted")
            # return self.bar_df
        pattern = "\$|\d+\.\d+|\d+\$"
        legends = legends[~legends["text"].str.contains(pattern)]
        position = 0

        legends["status"] = "False"
        if legend_position == LegendPosition.ver.value:
            for j, rows_j in self.obj_bb.iterrows():
                if rows_j["class"] == "1":
                    MIN_VAL = 10000
                    close_legend = None
                    for i, rows in legends.iterrows():
                        if (
                            rows_j["x_center"] < rows["x_center"]
                            and rows["status"] == "False"
                        ):
                            min_dist = get_pix_diff(
                                rows["y_center"], rows_j["y_center"]
                            )
                            if min_dist <= MIN_VAL:
                                MIN_VAL = min_dist

                                close_legend = rows["text"]
                                close_legend_x = rows["x_center"]
                                close_legend_y = rows["y_center"]
                                position = i
                    if close_legend != None:
                        potential_legend.append(
                            {
                                "legend_name": close_legend,
                                "legend_x_center": close_legend_x,
                                "legend_y_center": close_legend_y,
                                "width": rows_j["width"],
                                "height": rows_j["height"],
                                "x_val": rows_j["x_val"],
                                "y_val": rows_j["y_val"],
                                "x_center": rows_j["x_center"],
                                "y_center": rows_j["y_center"],
                            }
                        )
                        legends.at[position, "status"] = "True"

        elif legend_position == LegendPosition.hor.value:
            for j, rows_j in self.obj_bb.iterrows():
                if rows_j["class"] == "1":
                    MIN_VAL_x = 10000
                    MIN_VAL_y = 0.025
                    close_legend = None
                    for i, rows in legends.iterrows():
                        if (
                            rows["x_center"] > rows_j["x_center"]
                            and rows["status"] == "False"
                        ):
                            min_dist_x = get_pix_diff(
                                rows["x_center"], rows_j["x_center"]
                            )
                            min_dist_y = get_pix_diff(
                                rows["y_center"], rows_j["y_center"]
                            )

                            if min_dist_x < MIN_VAL_x and min_dist_y < MIN_VAL_y:
                                MIN_VAL_x = min_dist_x
                                #                                 MIN_VAL_y = min_dist_y
                                close_legend = rows["text"]
                                close_legend_x = rows["x_center"]
                                close_legend_y = rows["y_center"]
                                position = i
                                print(f"Legend name: {close_legend}")

                    if close_legend != None:
                        potential_legend.append(
                            {
                                "legend_name": close_legend,
                                "legend_x_center": close_legend_x,
                                "legend_y_center": close_legend_y,
                                "width": rows_j["width"],
                                "height": rows_j["height"],
                                "x_val": rows_j["x_val"],
                                "y_val": rows_j["y_val"],
                                "x_center": rows_j["x_center"],
                                "y_center": rows_j["y_center"],
                            }
                        )
                        legends.at[position, "status"] = "True"
        print("Potential legends in the chart: ")

        potential_legend = pd.DataFrame(potential_legend)
        print(potential_legend)
        self.log_data["details"].append(
            {"potential legends": potential_legend["legend_name"]}
        )
        elastic_logging.update_in_elasticsearch(self.es_id, self.log_data)
        return (legend_position, potential_legend)

    def legend_roi_color(self, image, legend_df, height, width):
        legend_color_list = []
        for i_l, row_l in legend_df.iterrows():
            w, h, xc, yc = (
                row_l["width"],
                row_l["height"],
                row_l["x_center"],
                row_l["y_center"],
            )
            xc *= width
            yc *= height
            w *= width
            h *= height
            # top left
            l_x1, l_y1 = int(xc - w / 2), int(yc - h / 2)
            l_x2, l_y2 = int(xc + w / 2), int(yc + h / 2)
            roi_l = image[l_y1:l_y2, l_x1:l_x2]
            average_color_l = np.mean(roi_l, axis=(0, 1))
            legend_color_list.append(
                {
                    "legend_name": row_l["legend_name"],
                    "legend_x_center": row_l["legend_x_center"],
                    "legend_y_center": row_l["legend_y_center"],
                    "legend_nomalized_x_center": int(xc),
                    "legend_nomalized_y_center": int(yc),
                    "legend_normalized_width": int(w),
                    "legend_normalized_height": int(h),
                    "average_color": np.round(average_color_l, 5),
                }
            )

        return pd.DataFrame(legend_color_list)

    def bar_roi_color(self, image, bar_df, height, width):
        bar_color_list = []
        for i_b, row_b in bar_df.iterrows():
            w_b, h_b, xc_b, yc_b = (
                row_b["width"],
                row_b["height"],
                row_b["x_center"],
                row_b["y_center"],
            )

            w_t, h_t, xc_t, yc_t = (
                row_b["text_width"],
                row_b["text_height"],
                row_b["text_x_center"],
                row_b["text_y_center"],
            )

            xc_t *= width
            yc_t *= height
            w_t *= width
            h_t *= height

            xc_b *= width
            yc_b *= height
            w_b *= width
            h_b *= height
            # top left
            b_x1, b_y1 = int(xc_b - w_b / 2), int(yc_b - h_b / 2)
            b_x2, b_y2 = int(xc_b + w_b / 2), int(yc_b + h_b / 2)
            roi_b = image[b_y1:b_y2, b_x1:b_x2]
            average_color_b = np.mean(roi_b, axis=(0, 1))
            bar_color_list.append(
                {
                    "bar_name": row_b["text"],
                    "bar_width": row_b["width"],
                    "bar_height": row_b["height"],
                    "bar_x_center": row_b["x_center"],
                    "bar_y_center": row_b["y_center"],
                    "text_normalized_width": int(w_t),
                    "text_normalized_height": int(h_t),
                    "text_normalized_x_center": int(xc_t),
                    "text_normalized_y_center": int(yc_t),
                    "bar_normalized_x_center": int(xc_b),
                    "bar_normalized_y_center": int(yc_b),
                    "bar_normalized_width": int(w_b),
                    "bar_normalized_height": int(h_b),
                    "bar_value": row_b["bar_value"],
                    "average_color": np.round(average_color_b, 5),
                }
            )
        return pd.DataFrame(bar_color_list)

    def legend_to_bar_map(self, color_similarity_threshold=40):
        bar_legend = []
        image = cv2.imread(self.image_path)
        height, width, _ = image.shape
        l_pos, legend_df = self.legend_to_text()
        if l_pos == LegendPosition.err.value:
            print("Legend can not be extracted")
            legend_counts = (self.obj_bb["class"] == "1").sum()
            print(f"Only {legend_counts} legends found")
            return (l_pos, self.bar_df)
        bar_df = self.bar_df.sort_values(["x_center"]).reset_index(inplace=False)
        legend_color_list = self.legend_roi_color(image, legend_df, height, width)
        print("Legend color list:")
        print(legend_color_list)
        bar_color_list = self.bar_roi_color(image, bar_df, height, width)
        print("Bar color list:")
        print(bar_color_list)
        for _, rows_b in bar_color_list.iterrows():
            min_diff_color = 1000
            closest_legend = None
            for _, rows in legend_color_list.iterrows():
                diff_rgb = np.round(
                    np.sqrt(
                        np.sum((rows["average_color"] - rows_b["average_color"]) ** 2)
                    ),
                    5,
                )
                if diff_rgb < min_diff_color:
                    min_diff_color = diff_rgb
                    closest_legend = rows["legend_name"]
                    close_legend_x = rows["legend_x_center"]
                    close_legend_y = rows["legend_y_center"]
                    closed_legend_x_norm = rows["legend_nomalized_x_center"]
                    closed_legend_y_norm = rows["legend_nomalized_y_center"]
                    closed_legend_w = rows["legend_normalized_width"]
                    closed_legend_h = rows["legend_normalized_height"]
            if closest_legend != None:
                bar_legend.append(
                    {
                        "legend_name": closest_legend,
                        "legend_x_center": close_legend_x,
                        "legend_y_center": close_legend_y,
                        "legend_normalized_x_center": closed_legend_x_norm,
                        "legend_normalized_y_center": closed_legend_y_norm,
                        "legend_normalized_width": closed_legend_w,
                        "legend_normalized_height": closed_legend_h,
                        "bar_name": rows_b["bar_name"],
                        "bar_width": rows_b["bar_width"],
                        "bar_height": rows_b["bar_height"],
                        "bar_x_center": rows_b["bar_x_center"],
                        "bar_y_center": rows_b["bar_y_center"],
                        "bar_normalized_x_center": rows_b["bar_normalized_x_center"],
                        "bar_normalized_y_center": rows_b["bar_normalized_y_center"],
                        "bar_normalized_width": rows_b["bar_normalized_width"],
                        "bar_normalized_height": rows_b["bar_normalized_height"],
                        "text_normalized_x_center": rows_b["text_normalized_x_center"],
                        "text_normalized_y_center": rows_b["text_normalized_y_center"],
                        "text_normalized_width": rows_b["text_normalized_width"],
                        "text_normalized_height": rows_b["text_normalized_height"],
                        "bar_value": rows_b["bar_value"],
                    }
                )
        bar_legend_df = pd.DataFrame(bar_legend)
        print(bar_legend_df)
        return (l_pos, bar_legend_df)

    def stacked_bar_values(self):
        def subtract_values(group):
            group["stack_value"] = np.round(
                group["bar_value"] - group["bar_value"].shift(), 3
            )
            return group

        l_pos, bar_legend_map = self.legend_to_bar_map()
        if l_pos == LegendPosition.err.value:
            bar_df = bar_legend_map.sort_values(["x_center"])
            if self.value_indicator == DataType.c.value:
                bar_df_sorted = bar_df.sort_values(["y_center"], ascending=False)
                bar_df_sorted["bar_value"] = bar_df_sorted["bar_value"].astype(float)
                group_df = bar_df_sorted.groupby("text")
                bar_df_subtracted = group_df.apply(subtract_values)
                df_subtracted1 = bar_df_subtracted.sort_values("x_center")
                df_subtracted1["stack_value"] = df_subtracted1["stack_value"].fillna(
                    df_subtracted1["bar_value"]
                )

                final_df = df_subtracted1

            elif self.value_indicator == DataType.e.value:
                bar_df_sorted = (
                    bar_df.groupby("text")
                    .apply(lambda x: x.sort_values("y_center"))
                    .reset_index(drop=True)
                )
                bar_df_sorted = bar_df_sorted.sort_values(["x_center"], ascending=True)
                final_df = bar_df_sorted.rename(columns={"bar_value": "stack_value"})

            print("Without legend data:")
            print(final_df)
            return (l_pos, final_df)
        try:
            bar_legend_map = bar_legend_map.sort_values("bar_x_center")
            df_sorted = bar_legend_map.sort_values("bar_y_center", ascending=False)

            df_sorted["bar_value"] = df_sorted["bar_value"].astype(float)

            if self.value_indicator == DataType.c.value:
                group = df_sorted.groupby("bar_name")
                df_subtracted = group.apply(subtract_values)
                df_subtracted1 = df_subtracted.sort_values("bar_x_center")
                df_subtracted1["stack_value"] = df_subtracted1["stack_value"].fillna(
                    df_subtracted1["bar_value"]
                )

                final_df = df_subtracted1

            elif self.value_indicator == DataType.e.value:
                final_df = df_sorted
                final_df = final_df.rename(columns={"bar_value": "stack_value"})

            if l_pos == LegendPosition.ver.value:
                df_sorted = final_df.sort_values(
                    by=["bar_x_center", "legend_y_center"], ascending=[True, True]
                )
                final_df = (
                    df_sorted.groupby("bar_name")
                    .apply(
                        lambda x: x.sort_values(
                            by=["bar_x_center", "legend_y_center"],
                            ascending=[True, True],
                        )
                    )
                    .reset_index(drop=True)
                )
                # final_df = final_df.sort_values(
                #     ["bar_x_center", "legend_y_center"], ascending=[True, True]
                # )
                final_df["legend_name"] = pd.Categorical(
                    final_df["legend_name"],
                    categories=final_df.sort_values("legend_y_center", ascending=True)[
                        "legend_name"
                    ].unique(),
                    ordered=True,
                )

                # Sort 'bar_name' based on 'bar_x_center'
                final_df["bar_name"] = pd.Categorical(
                    final_df["bar_name"],
                    categories=final_df.sort_values("bar_x_center")[
                        "bar_name"
                    ].unique(),
                    ordered=True,
                )
            elif l_pos == LegendPosition.hor.value:
                df_sorted = final_df.sort_values(
                    by=["bar_x_center", "legend_y_center"], ascending=[True, False]
                )
                final_df = (
                    df_sorted.groupby("bar_name")
                    .apply(
                        lambda x: x.sort_values(
                            by=["bar_x_center", "legend_y_center"],
                            ascending=[True, False],
                        )
                    )
                    .reset_index(drop=True)
                )
                final_df = final_df.sort_values(
                    ["bar_x_center", "legend_x_center"], ascending=[True, False]
                )

            # final_df = final_df.rename(columns={"bar_name": "x_label"})
            df_test = final_df.pivot_table(
                index="bar_name",
                columns="legend_name",
                values=[
                    "stack_value",
                    "legend_normalized_x_center",
                    "legend_normalized_y_center",
                    "legend_normalized_width",
                    "legend_normalized_height",
                    "bar_normalized_x_center",
                    "bar_normalized_y_center",
                    "bar_normalized_width",
                    "bar_normalized_height",
                    "bar_normalized_height",
                    "text_normalized_x_center",
                    "text_normalized_y_center",
                    "text_normalized_width",
                    "text_normalized_height",
                ],
                aggfunc="first",
                sort=False,
                fill_value=0,
            )
            df_test = df_test.reset_index()
        except KeyError as e:
            print(e)
            l_pos = LegendPosition.err.value
            bar_df = self.bar_df.sort_values(["x_center"])
            if self.value_indicator == DataType.c.value:
                bar_df_sorted = bar_df.sort_values(["y_center"], ascending=False)
                bar_df_sorted["bar_value"] = bar_df_sorted["bar_value"].astype(float)
                group_df = bar_df_sorted.groupby("text")
                bar_df_subtracted = group_df.apply(subtract_values)
                df_subtracted1 = bar_df_subtracted.sort_values("x_center")
                df_subtracted1["stack_value"] = df_subtracted1["stack_value"].fillna(
                    df_subtracted1["bar_value"]
                )

                final_df = df_subtracted1

            elif self.value_indicator == DataType.e.value:
                bar_df_sorted = (
                    bar_df.groupby("text")
                    .apply(lambda x: x.sort_values("y_center"))
                    .reset_index(drop=True)
                )
                bar_df_sorted = bar_df_sorted.sort_values(["x_center"], ascending=True)
                final_df = bar_df_sorted.rename(columns={"bar_value": "stack_value"})

            print("Without legend data:")
            print(final_df)
            return (l_pos, final_df)
        print(df_test)
        return l_pos, df_test

    def grouped_bar_values(self):
        l_pos, bar_legend_map = self.legend_to_bar_map()

        if l_pos == LegendPosition.err.value:
            bar_df = self.bar_df.sort_values(["x_center"])
            bar_df = bar_df.rename(columns={"bar_value": "stack_value"})
            return l_pos, bar_df
        try:
            final_table = bar_legend_map
            if l_pos == LegendPosition.ver.value:
                final_table = final_table.sort_values(
                    ["legend_y_center", "bar_x_center"]
                )
                final_table["legend_name"] = pd.Categorical(
                    final_table["legend_name"],
                    categories=final_table.sort_values("legend_y_center")[
                        "legend_name"
                    ].unique(),
                    ordered=True,
                )

                # Sort 'bar_name' based on 'bar_x_center'
                final_table["bar_name"] = pd.Categorical(
                    final_table["bar_name"],
                    categories=final_table.sort_values("bar_x_center")[
                        "bar_name"
                    ].unique(),
                    ordered=True,
                )

            elif l_pos == LegendPosition.hor.value:
                final_table = final_table.sort_values(
                    ["legend_x_center", "bar_x_center"], ascending=True
                )

            final_table = final_table.rename(columns={"bar_value": "stack_value"})

            df_test = final_table.pivot_table(
                index="bar_name",
                columns="legend_name",
                values=[
                    "stack_value",
                    "legend_normalized_x_center",
                    "legend_normalized_y_center",
                    "legend_normalized_width",
                    "legend_normalized_height",
                    "bar_normalized_x_center",
                    "bar_normalized_y_center",
                    "bar_normalized_width",
                    "bar_normalized_height",
                    "text_normalized_x_center",
                    "text_normalized_y_center",
                    "text_normalized_width",
                    "text_normalized_height",
                ],
                aggfunc="first",
                sort=False,
                fill_value=0,
            )
            df_test = df_test.reset_index()

        except KeyError as e:
            print(e)
            l_pos = LegendPosition.err.value
            bar_df = self.bar_df.sort_values(["x_center"])
            bar_df = bar_df.rename(columns={"bar_value": "stack_value"})
            return l_pos, bar_df
        print(df_test)
        return l_pos, df_test
