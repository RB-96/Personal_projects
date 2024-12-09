import time
import pandas as pd
from app.server.services.chart_llm_pipeline import LlmPipeline
from bar_charts_handler.bar_legend_model.bar_legend_model_downloader import (
    model_downloader,
)
from chart_classification.classification_model_download import (
    classification_model_downloader,
)
from chart_detection_with_OCR.utils import Chart_detection
import chart_classification.classification_type as cls_type

from app.server.utils.utility_functions import ModelUsed, DataType, BarType
from general_handle import Chart_handler
import elastic_logging
from base import BaseChartPasrer

model_downloader()
classification_model_downloader()


class ChartParsing(BaseChartPasrer):
    def __init__(self, file_name: str, file_path: str, path_job_id: None, es_id: None, log_data: None) -> None:
        self.file_path = file_path
        self.file_name = file_name
        self.path_job_id = path_job_id
        self.es_id = es_id
        self.log_data = log_data
        self.c_detect = Chart_detection(self.file_path, self.path_job_id)
        self.chart_handler = Chart_handler()
        self.class_obj = cls_type.ChartClass()

        # LLM model selection
        self.llm_model_used = "claude"
        self.llm_model = LlmPipeline(model=self.llm_model_used)

    def parse_pdf(self):

        final_charts = []
        model_used = None
        c_type = None
        title = ""
        chart_type = None

        t_start = time.time()
        obj_charts = self.c_detect.charts_by_grids()
        self.log_data["# charts detected"] = len(obj_charts)
        elastic_logging.update_in_elasticsearch(self.es_id, self.log_data)

        for i, rows in obj_charts.iterrows():
            img_path = rows["image_path"]
            pg_no = rows["page_num"]
            chart_width = rows["chart_width"]
            chart_height = rows["chart_height"]
            chart_left = rows["chart_left"]
            chart_top = rows["chart_top"]
            page_width = rows["page_width"]
            page_height = rows["page_height"]
            chart_width_norm = rows["chart_width_norm"]
            chart_height_norm = rows["chart_height_norm"]
            chart_left_norm = rows["chart_left_norm"]
            chart_top_norm = rows["chart_top_norm"]

            chart_type = self.class_obj.classify_charts(img_path)

            if chart_type == "BarGraph":
                try:
                    c_type = chart_type
                    (l_pos, c_type, df, title, data_present) = self.chart_handler.chart_values_extract(
                        img_path, chart_height, chart_width, chart_left, chart_top, self.es_id, self.log_data
                    )
                    if data_present == DataType.c.value:
                        model_used = ModelUsed.inh.value
                        print(f"Chart_width = {chart_width}\nChart_height = {chart_height}")
                        if df is not None:
                            if df.empty == False:
                                final_charts.append(
                                    {
                                        "chart_class": chart_type,
                                        "c_type": c_type,
                                        "chart_title": title,
                                        "l_pos": l_pos,
                                        "page_no": pg_no,
                                        "chart_width": chart_width,
                                        "chart_height": chart_height,
                                        "chart_left": chart_left,
                                        "chart_top": chart_top,
                                        "page_width": page_width,
                                        "page_height": page_height,
                                        "chart_width_norm": chart_width_norm,
                                        "chart_height_norm": chart_height_norm,
                                        "chart_left_norm": chart_left_norm,
                                        "chart_top_norm": chart_top_norm,
                                        "data": df,
                                        "data_type": data_present,
                                    }
                                )
                            self.log_data["extraction_time"] = f"Success {time.time() - t_start}"
                            self.log_data["details"].append(
                                {"chart_type": c_type, "page_no": pg_no, "model_used": model_used}
                            )
                            elastic_logging.update_in_elasticsearch(self.es_id, self.log_data)

                    elif data_present == DataType.e.value:
                        model_used = f"{ModelUsed.llm.value}-{self.llm_model_used}"

                        # image_df = self.gpt_obj.get_gpt_response(img_path)
                        image_df, llm_log_data = self.llm_model.llm_chart_parser_pipeline(img_path)
                        self.log_data["llm_log_data"] = llm_log_data
                        try:
                            if image_df is not None:
                                if image_df.empty == False:
                                    final_charts.append(
                                        {
                                            "chart_class": chart_type,
                                            "c_type": c_type,
                                            "page_no": pg_no,
                                            "chart_width": chart_width,
                                            "chart_height": chart_height,
                                            "chart_left": chart_left,
                                            "chart_top": chart_top,
                                            "page_width": page_width,
                                            "page_height": page_height,
                                            "chart_width_norm": chart_width_norm,
                                            "chart_height_norm": chart_height_norm,
                                            "chart_left_norm": chart_left_norm,
                                            "chart_top_norm": chart_top_norm,
                                            "data": image_df,
                                            "data_type": data_present,
                                        }
                                    )
                                    self.log_data["extraction_time"] = f"Success {time.time() - t_start}"
                                    self.log_data["details"].append(
                                        {"chart_type": c_type, "page_no": pg_no, "model_used": model_used}
                                    )
                                    elastic_logging.update_in_elasticsearch(self.es_id, self.log_data)
                        except KeyError as e:
                            print(e)
                            self.log_data["extraction_status"] = f"Failed due to {e}"

                    elif df is None and c_type == BarType.hor.value or c_type == BarType.und.value:
                        model_used = f"{ModelUsed.llm.value}-{self.llm_model_used}"

                        # image_df = self.gpt_obj.get_gpt_response(img_path)
                        image_df, llm_log_data = self.llm_model.llm_chart_parser_pipeline(img_path)
                        self.log_data["llm_log_data"] = llm_log_data

                        try:
                            if image_df is not None:
                                if image_df.empty == False:
                                    final_charts.append(
                                        {
                                            "chart_class": chart_type,
                                            "c_type": c_type,
                                            "page_no": pg_no,
                                            "chart_width": chart_width,
                                            "chart_height": chart_height,
                                            "chart_left": chart_left,
                                            "chart_top": chart_top,
                                            "page_width": page_width,
                                            "page_height": page_height,
                                            "chart_width_norm": chart_width_norm,
                                            "chart_height_norm": chart_height_norm,
                                            "chart_left_norm": chart_left_norm,
                                            "chart_top_norm": chart_top_norm,
                                            "data": image_df,
                                        }
                                    )
                                    self.log_data["extraction_time"] = f"Success {time.time() - t_start}"
                                    self.log_data["details"].append(
                                        {"chart_type": c_type, "page_no": pg_no, "model_used": model_used}
                                    )
                                    elastic_logging.update_in_elasticsearch(self.es_id, self.log_data)
                        except KeyError as e:
                            print(e)
                            self.log_data["extraction_status"] = f"Failed due to {e}"
                except KeyError as e:
                    print(e)
                    self.log_data["extraction_status"] = f"Failed due to {e}"

            else:
                model_used = f"{ModelUsed.llm.value}-{self.llm_model_used}"

                # image_df = self.gpt_obj.get_gpt_response(img_path)
                image_df, llm_log_data = self.llm_model.llm_chart_parser_pipeline(img_path)
                self.log_data["llm_log_data"] = llm_log_data

                try:
                    if image_df is not None:
                        if image_df.empty == False:
                            final_charts.append(
                                {
                                    "chart_class": chart_type,
                                    "chart_class": chart_type,
                                    "page_no": pg_no,
                                    "chart_width": chart_width,
                                    "chart_height": chart_height,
                                    "chart_left": chart_left,
                                    "chart_top": chart_top,
                                    "page_width": page_width,
                                    "page_height": page_height,
                                    "chart_width_norm": chart_width_norm,
                                    "chart_height_norm": chart_height_norm,
                                    "chart_left_norm": chart_left_norm,
                                    "chart_top_norm": chart_top_norm,
                                    "data": image_df,
                                }
                            )
                except KeyError as e:
                    print(e)
                    self.log_data["extraction_status"] = f"Failed due to {e}"
                self.log_data["extraction_time"] = f"Success {time.time() - t_start}"
                self.log_data["details"].append({"chart_type": chart_type, "page_no": pg_no, "model_used": model_used})
                elastic_logging.update_in_elasticsearch(self.es_id, self.log_data)
        final_charts = pd.DataFrame(final_charts)
        print(final_charts.columns)
        final_output = self.chart_handler.chart_reponse(final_charts, self.file_name)
        return final_output
