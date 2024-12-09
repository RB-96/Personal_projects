import base64
import json
import re
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI
import openai
import pandas as pd
from app.server.models.base_llm import BaseLlm
import settings
from langchain_core.messages import HumanMessage, SystemMessage
import boto3
import numpy as np


class LlmService(BaseLlm):
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0,
        max_tokens: int = 1024,
        input_token_cost: int = 0.003,
        output_token_cost: int = 0.015,
    ) -> None:
        super().__init__(
            input_token_cost=input_token_cost,
            output_token_cost=output_token_cost,
        )

        self.llm_model: AzureChatOpenAI = self.initialize_llm(
            temperature=temperature, max_tokens=max_tokens, model_name=model
        )

    def initialize_llm(self, temperature: float = 0, max_tokens: int = 1024, model_name: str = "gpt-4o"):
        if model_name == "gpt-4o":
            model = AzureChatOpenAI(
                api_key=settings.AZURE_GPT_4O_API_KEY,
                api_version=settings.AZURE_GPT_4O_API_VERSION,
                azure_endpoint=settings.AZURE_GPT_4O_ENDPOINT,
                azure_deployment=settings.AZURE_GPT_4O_DEPLOYMENT,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=60,
                max_retries=0,
            )
        elif model_name == "claude":
            bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_KEY,
                aws_secret_access_key=settings.AWS_SECRET_KEY,
            )
            model = ChatBedrock(
                client=bedrock_runtime,
                model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
                model_kwargs=dict(temperature=temperature, max_tokens=max_tokens),
                verbose=True,
            )
        else:
            raise ValueError(f"model {model} not supported")
        return model

    def get_response(self, image: base64, system_prompt: str = None, user_prompt: str = None):
        messages = [
            SystemMessage(
                content=system_prompt,
            ),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                    # {"type": "text", "text": user_prompt}
                ]
            ),
        ]
        try:
            response = self.llm_model.invoke(messages)
            return response
        except openai.InternalServerError as e:
            raise e

    @staticmethod
    def llm_response_to_json(response):
        try:
            pattern = re.compile(r"<json_output>(.*?)</json_output>", re.DOTALL)
            result = pattern.findall(response.content)[0]
        except IndexError:
            try:
                pattern = re.compile(r"```json(.*?)```", re.DOTALL)
                result = pattern.findall(response.content)[0]
            except IndexError:
                result = response.content
        return json.loads(result), response.response_metadata

    @staticmethod
    def postprocess_json_to_df(json_data, metadata):
        df = pd.DataFrame(json_data)
        

        missing_values = [np.nan, None, "NA", "N/A", "NaN", "nan", "NAN", " ", "-"]
        df.replace(missing_values, np.nan, inplace=True)
        df.fillna(0, inplace=True)

        df.set_index(df.columns[0], inplace=True)

        try:
            if "title" in metadata and metadata["title"]:
                title = metadata["title"]
                print(f"Metadata title: {title}")
                
                if "value" in df.columns:
                    df.rename(columns={"value": title}, inplace=True)
                else:
                    print("'value' column not found in DataFrame.")

            else:
                print("Metadata title is missing or empty.")
        except KeyError as e:
            print(f"KeyError during post-processing: {e}")
        except Exception as e:
            print(f"Unexpected error during post-processing: {e}")
            
        print(df)

        return df

