from openai import AzureOpenAI
import base64
from io import StringIO
import cv2
import pandas as pd
import numpy as np
import settings
import requests
from prompt.prompt_template import prompt_assist, prompt_user


class GptParsing:
    def __init__(self) -> None:
        self.client = AzureOpenAI(
            api_key=settings.AZURE_GPT_API_KEY,
            api_version=settings.AZURE_GPT_API_VERSION,
            base_url=f"{settings.AZURE_GPT_ENDPOINT}openai/deployments/{settings.AZURE_GPT_MODEL}/extensions",
        )

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
        text = str(text)
        text = text.lower()
        for unit, multiplier in units.items():
            if unit in text:
                number_part = text.split(unit)[0]
                try:
                    number = float(number_part)
                    result = number * multiplier
                    return str(result)
                except ValueError:
                    continue
        return text

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_gpt_response(self, image_path):
        base64_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model=settings.AZURE_GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt_assist,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_user,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=3000,
        )

        csv_data = response.choices[0].message.content
        cleaned_string = (
            csv_data.replace("```", "").replace("```", "").replace("csv", "")
        )
        cleaned_string = cleaned_string.replace("plaintext", "")
        try:
            df = pd.read_csv(StringIO(cleaned_string))
            missing_values = [np.nan, None, "NA", "N/A", "NaN", "nan", "NAN", " ", "-"]
            df.replace(missing_values, np.nan, inplace=True)
            df.fillna(0, inplace=True)
            df.columns = df.columns.str.replace('"', "")
            df.replace('"', "", regex=True, inplace=True)
            columns_to_apply = df.columns[1:]
            df[columns_to_apply] = df[columns_to_apply].applymap(
                self.convert_currency_for_bars
            )

            return df
        except:
            pass
