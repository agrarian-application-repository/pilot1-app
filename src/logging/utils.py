import os
from typing import Union

from dotenv import load_dotenv


def get_wandb_api_key() -> str:
    """
    Load the W&B API key from the environment variables.

    This function loads environment variables from a `.env` file, retrieves the
    W&B (Weights & Biases) API key, and ensures it is found. If the key is not found,
    it raises a `ValueError`.

    Returns:
        str: The W&B API key.

    Raises:
        ValueError: If the W&B API key is not found in the environment variables.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve W&B API key from environment variables
    wandb_api_key: Union[str, None] = os.getenv("WANDB_API_KEY")

    # Ensure W&B API key is loaded
    if wandb_api_key is None:
        raise ValueError("W&B API key not found. Please check your .env file.")

    return wandb_api_key
