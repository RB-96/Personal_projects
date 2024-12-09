"""
This code works for all verticl bar charts and segmented bar charts where legend is either in the right side or 
bottom side of chart.
stacked_data_response_formatter(): For generating the response of the stacked chart which produces three dataframes
one for the data and another twos for the coordinates of bars and legends. I have added the bar coordinates in the
response data now. 
"""
import cv2
import pandas as pd
import numpy as np

class Chart_handler:
    def __init__(self)->None:
        super().__init__()
        
    def extract_number_from_text(self, text):
        try:
            float(text)
            return [float(text)]
        except ValueError:
            return []
        
    def get_cell_dict(self, row, col, text, width, height, left, top, page):
        cell_dict = {
            "id": "89494a6c-8b3f-4c63-b022-8c0b57a67022",
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
        
    def create_tabular_data_simple(self, ocr_result, page):
        print("Chart Data: ")
        print(ocr_result)
        
        cells = []
        row_ = 1
        col_ = 1
        for idx, rows in ocr_result.iterrows():
            col_ = 1
            cells.append(self.get_cell_dict(row_, col_, idx, 0, 0, 0, 0, page))
            col_+=1
            for _,value in rows.items():
                cells.append(self.get_cell_dict(row_, col_, value, 0, 0, 0, 0, page))
                col_+=1
            row_+=1
        return cells
    
    def create_tabular_data_pie(self, ocr_result, page):
        print("Chart Data: ")
        print(ocr_result)
        
        cells = []
        row_ = 1
        col_ = 1
        for colmn, _ in ocr_result.iloc[0].items():
            cell_val = colmn
            cells.append(self.get_cell_dict(row_, col_, cell_val, 0, 0, 0, 0, page))
            col_ += 1
        row_ = 2
        for i, row in ocr_result.iterrows():
            col_ = 1
            for _, value in row.items():
                cells.append(self.get_cell_dict(row_, col_, value, 0, 0, 0, 0, page))
                col_+=1
            row_+=1
            
        return cells

        
    def chart_reponse(self, final_charts, file_name):
        tab_data = {
            "result": [],
        }
        res=0
        for page_no, group in final_charts.groupby("page_no"):
            tab_data["result"].append(
                {"input": file_name,
                "prediction": [],
                "page": page_no}
            )
            pred = 0
            for i, rows_ in group.iterrows():
                if rows_["chart_class"] != "PieChart":
                    tab_data["result"][res]["prediction"].append(
                            {
                                "id": "aae80e34-533c-4712-b03f-199147eb7b4d",
                                "page_no": page_no,
                                "Width": rows_["chart_width"],
                                "Height": rows_["chart_height"],
                                "Left": rows_["chart_left"],
                                "Top": rows_["chart_top"],
                                # "headers": [],
                                "cells": [],
                            }
                    )
                    ocr_result = rows_["data"]
                    ocr_df = ocr_result.T
                    cells = self.create_tabular_data_simple(ocr_df, page_no)
                    
                else:
                    tab_data["result"][res]["prediction"].append(
                            {
                                "id": "aae80e34-533c-4712-b03f-199147eb7b4d",
                                "page_no": page_no,
                                "Width": rows_["chart_width"],
                                "Height": rows_["chart_height"],
                                "Left": rows_["chart_left"],
                                "Top": rows_["chart_top"],
                                # "headers": [],
                                "cells": [],
                            }
                    )
                    ocr_result = rows_["data"]
                    
                    cells = self.create_tabular_data_pie(ocr_result, page_no)
                    
                    
                    
                tab_data["result"][res]["prediction"][pred]["cells"] = cells
                    
                    
                    
                pred+=1
                    
            res+=1
                
                    
        comp_data = {
                "tabular_data": tab_data,
            }
        return tab_data
            
    