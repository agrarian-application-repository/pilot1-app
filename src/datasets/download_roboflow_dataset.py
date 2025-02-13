import argparse
import logging
import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow


def parse_arguments():
    """
    Parses command-line arguments for workspace, project, and version.

    Returns:
        argparse.Namespace: Parsed arguments with attributes `workspace`, `project`, and `version`.
    """
    parser = argparse.ArgumentParser(description="Download a dataset from Roboflow.")
    parser.add_argument("--workspace", type=str, required=True, help="The name of the workspace.")
    parser.add_argument("--project", type=str, required=True, help="The name of the project.")
    parser.add_argument("--version", type=int, required=True, help="The version number of the project.")
    parser.add_argument("--env_file", type=str, default=os.getcwd(), help="The path to the .env file containing the API key.")
    parser.add_argument("--save_location", type=str, default="./data", help="The path where to save the downloaded dataset")
    return parser.parse_args()


def terminate(message: str, exit_code: int = 1):
    """Logs an error message and exits the program."""
    logging.error(message)
    sys.exit(exit_code)


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Parse command-line arguments
    args = parse_arguments()
    logging.info(f"Parsed arguments: "
                 f"workspace='{args.workspace}', "
                 f"project='{args.project}', "
                 f"version={args.version}, "
                 f"save_location={args.save_location}, "
                 f"env_file={args.env_file}"
                 )

    # Load environment variables from .env if it exists
    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"Loaded environment variables from {env_path}")
    else:
        terminate(f"No ENV file found in working directory {env_path}. Ensure environment variables are set.")

    # Fetch the Roboflow API key
    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY", None)
    if roboflow_api_key is None:
        terminate("ROBOFLOW API key not found. Please check your .env file or environment settings.")

    try:
        # Initialize Roboflow with the API key
        rf = Roboflow(api_key=roboflow_api_key)
        logging.info("Initialized Roboflow API client.")

        # Access the specified workspace, project, and version
        workspace = rf.workspace(args.workspace)
        project = workspace.project(args.project)
        version = project.version(args.version)
        logging.info(f"Accessed workspace '{args.workspace}', project '{args.project}', version {args.version}.")

        # Download the dataset to the designated data directory
        download_location = Path(args.save_location)
        logging.info(f"Starting dataset download to '{download_location}'...")
        version.download("yolov11", location=download_location)
        logging.info(f"Dataset downloaded successfully to '{download_location}'.")

    except Exception as e:
        terminate(f"An error occurred during the dataset download process: {e}")


if __name__ == "__main__":
    main()
