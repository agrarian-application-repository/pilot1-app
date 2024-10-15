import argparse
import os

from dotenv import load_dotenv
from roboflow import Roboflow


def parse_arguments():
    """
    Parses command-line arguments for workspace, project, and version.
    """
    parser = argparse.ArgumentParser(description="Download a dataset from Roboflow.")
    parser.add_argument('--workspace', type=str, required=True, help='The name of the workspace.')
    parser.add_argument('--project', type=str, required=True, help='The name of the project.')
    parser.add_argument('--version', type=int, required=True, help='The version number of the project.')

    return parser.parse_args()


def main():
    # Load environment variables from .env
    load_dotenv()

    # Fetch the Roboflow API key
    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    if roboflow_api_key is None:
        raise ValueError("ROBOFLOW API key not found. Please check your .env file.")

    # Parse command-line arguments
    args = parse_arguments()

    # Initialize Roboflow with the API key
    rf = Roboflow(api_key=roboflow_api_key)

    # Access the specified workspace, project, and version
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    # Download the dataset to the data directory
    dataset = version.download("yolov11", location="./data")


if __name__ == "__main__":
    main()
