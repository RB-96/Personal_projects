from pathlib import Path
import time
from typing import Optional, Tuple, Union
import pandas as pd
from app.server.static import prompt_template
import numpy as np
from app.server.utils import llm_utils, image_utils


class LlmPipeline:
    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self.llm_model = llm_utils.LlmService(model=model, temperature=0, max_tokens=2048)

    def llm_chart_parser_pipeline(self, image: Optional[Union[Path, str, np.ndarray]]) -> Tuple[pd.DataFrame, dict]:
        """
        This function takes an image of a chart and uses a GPT-4O Large Language Model (LLM) to extract the data from the chart.

        Parameters
        ----------
        image: Optional[Union[Path, str, np.ndarray]]
            The image of the chart. Can be a Path to the image, a numpy array, or None.

        Returns
        -------
        A tuple of a pandas DataFrame and a dictionary containing logging data.
        The DataFrame contains the extracted data from the chart, and the dictionary contains data about the LLM's performance, such as the number of tokens used and the cost of the operation.
        """
        log_data = {}
        if isinstance(image, (Path, str)):
            base_64_image = image_utils.create_base64_image(image_utils.read_image_from_path(image))
        elif isinstance(image, np.ndarray):
            base_64_image = image_utils.create_base64_image(image)
        else:
            raise ValueError("image must be either a Path or a numpy array")

        start_time = time.time()
        if self.model == "gpt-4o":
            prompt = prompt_template.GPT_SYSTEM_PROMPT
        elif self.model == "claude":
            prompt = prompt_template.CLAUDE_SYSTEM_PROMPT
        response = self.llm_model.get_response(image=base_64_image, system_prompt=prompt)
        json_res, metadata = self.llm_model.llm_response_to_json(response)

        # get the token usage and token cost and log it
        if "usage" in metadata:
            token_usage = metadata["usage"]
        else:
            token_usage = metadata["token_usage"]
        input_tokens = token_usage["prompt_tokens"]
        output_tokens = token_usage["completion_tokens"]
        log_data["tokens"] = token_usage

        cost = self.llm_model.token_cost(input_tokens, output_tokens)
        log_data["token_cost"] = cost

        log_data["time_taken_by_llm"] = f"{round(time.time() - start_time,2)} s"

        # Post-process the data
        df = self.llm_model.postprocess_json_to_df(json_res["data"], json_res)
        return df, log_data
