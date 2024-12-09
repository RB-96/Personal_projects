from abc import ABC, abstractmethod
import base64


class BaseLlm(ABC):
    @abstractmethod
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0,
        max_tokens: int = 1024,
        input_token_cost: int = 0,
        output_token_cost: int = 0,
    ) -> None:
        """
        Initialize a BaseLlm object.

        Parameters
        ----------
        model_name : str
            The name of the LLM model to use. If None, the default model will be used.
        temperature : float
            The temperature to use for the model. If 0, the default temperature will be used.
        max_tokens : int
            The maximum number of tokens to generate for a single response. If 0, the default value will be used.
        input_token_cost : int
            The cost of input per 1000 tokens. If 0, the cost will be ignored.
        output_token_cost : int
            The cost of output per 1000 tokens. If 0, the cost will be ignored.
        """
        self.input_token_cost = input_token_cost
        self.output_token_cost = output_token_cost

    def token_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the total cost of generating a response, given the input and output token counts.

        Parameters
        ----------
        input_tokens : int
            The number of input tokens.
        output_tokens : int
            The number of output tokens.

        Returns
        -------
        float
            The total cost of generating a response, in the currency set by the LLM provider.
        """
        if not self.input_token_cost > 0 or not self.output_token_cost > 0:
            raise ValueError("input token cost or output token cost must be greater than 0")
        return (input_tokens / 1000) * self.input_token_cost + (output_tokens / 1000) * self.output_token_cost

    @abstractmethod
    def initialize_llm(self, model_name: str = None, temperature: float = 0, max_tokens: int = 1024):
        """
        Initialize the LLM model.

        Parameters
        ----------
        model_name : str
            The name of the LLM model to use. If None, the default model will be used.
        temperature : float
            The temperature to use for the model. If 0, the default temperature will be used.
        max_tokens : int
            The maximum number of tokens to generate for a single response. If 0, the default value will be used.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        NotImplementedError

    @abstractmethod
    def get_response(self, system_prompt: str = None, user_prompt: str = None, image: base64 = None):
        """
        Get a response from the LLM model.

        Parameters
        ----------
        system_prompt : str, optional
            The system prompt to send to the LLM model. If None, the default prompt will be used.
        user_prompt : str, optional
            The user prompt to send to the LLM model. If None, the default prompt will be used.
        image : base64, optional
            A base64 encoded image to send to the LLM model. If None, no image will be sent.

        Returns
        -------
        str
            The response from the LLM model.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        NotImplementedError
