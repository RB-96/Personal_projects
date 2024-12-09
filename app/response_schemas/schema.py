HEALTH_CHECK_RESPONSE = {
    200: {"description": "Success", "content": {"application/json": {"example": {"message": "Alive..."}}}},
}

CHART_PARSER_RESPONSE = {
    200: {
        "description": "Success",
        "content": {
            "application/json": {
                "example": {
                    "mapping_data": {
                        "mapping_data": {
                            "complete_processed_data": {
                                "tabular_data": {
                                    "result": [
                                        {
                                            "input": "Sample.pdf",
                                            "prediction": [
                                                {
                                                    "table_id": "8a8fb46b-86da-435d-87b6-400cf8590224",
                                                    "page_no": 0,
                                                    "Width": 3046,
                                                    "Height": 1278,
                                                    "Left": 279,
                                                    "Top": 580,
                                                    "width_norm": 0.86822,
                                                    "height_norm": 0.514844,
                                                    "x_left_norm": 0.079707,
                                                    "y_top_norm": 0.233782,
                                                    "cells": [
                                                        {
                                                            "cell_id": "1d0cc2ac-cbf5-4ad7-8925-910528ff5d9a",
                                                            "row": 1,
                                                            "col": 1,
                                                            "row_span": 1,
                                                            "col_span": 1,
                                                            "score": 0.91002023,
                                                            "page": 0,
                                                            "text": "Month",
                                                            "Width": 3046,
                                                            "Height": 1278,
                                                            "Left": 279,
                                                            "Top": 580,
                                                            "number_in_text": [],
                                                            "currency": "null",
                                                            "units": "null",
                                                        },
                                                        {
                                                            "cell_id": "c1b3488a-56b0-44bb-898c-81be658f9b6d",
                                                            "row": 1,
                                                            "col": 2,
                                                            "row_span": 1,
                                                            "col_span": 1,
                                                            "score": 0.91002023,
                                                            "page": 0,
                                                            "text": "Mar-23",
                                                            "Width": 3046,
                                                            "Height": 1278,
                                                            "Left": 279,
                                                            "Top": 580,
                                                            "number_in_text": [],
                                                            "currency": "null",
                                                            "units": "null",
                                                        },
                                                    ],
                                                    "chart_type": "LineGraph",
                                                    "model_used": "Vision model",
                                                    "headers": [
                                                        {
                                                            "version": "Actual",
                                                            "headerText": "Mar-23",
                                                            "dateLabel": "2023-03-31",
                                                            "periodType": "Monthly",
                                                            "currency": "USD",
                                                            "units": "Millions",
                                                            "variance": False,
                                                            "headerType": "dateLabel",
                                                            "version_score": 80,
                                                            "currency_score": 80,
                                                            "periodType_score": 85,
                                                            "unit_score": 75,
                                                            "dateLabel_score": 90,
                                                            "variance_score": 80,
                                                        },
                                                        {
                                                            "version": "Actual",
                                                            "headerText": "Apr-23",
                                                            "dateLabel": "2023-04-30",
                                                            "periodType": "Monthly",
                                                            "currency": "USD",
                                                            "units": "Millions",
                                                            "variance": False,
                                                            "headerType": "dateLabel",
                                                            "version_score": 80,
                                                            "currency_score": 80,
                                                            "periodType_score": 85,
                                                            "unit_score": 75,
                                                            "dateLabel_score": 90,
                                                            "variance_score": 80,
                                                        },
                                                    ],
                                                    "table_type": "others",
                                                    "table_type_confidence_score": 0.46,
                                                }
                                            ],
                                            "page": 0,
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    "request_id": "1334",
                    "success": True,
                }
            }
        },
    },
}
