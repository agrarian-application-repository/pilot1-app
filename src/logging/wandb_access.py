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


def get_wandb_entity() -> str:
    """
    Load the W&B Entity from the environment variables.

    This function loads environment variables from a `.env` file, retrieves the
    W&B (Weights & Biases) Entity, and ensures it is found. If the Entity is not found,
    it raises a `ValueError`.

    Returns:
        str: The W&B Entity.

    Raises:
        ValueError: If the W&B API Entity is not found in the environment variables.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve W&B Entity from environment variables
    wandb_entity: Union[str, None] = os.getenv("WANDB_ENTITY")

    # Ensure W&B API key is loaded
    if wandb_entity is None:
        raise ValueError("W&B Entity not found. Please check your .env file.")

    return wandb_entity


def get_wandb_username() -> str:
    """
    Load the W&B Username from the environment variables.

    This function loads environment variables from a `.env` file, retrieves the
    W&B (Weights & Biases) Username, and ensures it is found. If the Username is not found,
    it raises a `ValueError`.

    Returns:
        str: The W&B Username.

    Raises:
        ValueError: If the W&B Username is not found in the environment variables.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve W&B Username from environment variables
    wandb_username: Union[str, None] = os.getenv("WANDB_USERNAME")

    # Ensure W&B Username is loaded
    if wandb_username is None:
        raise ValueError("W&B Username not found. Please check your .env file.")

    return wandb_username
