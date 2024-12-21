import argparse
import json
import logging

from config import logger  # Import the logger from config.py
from llm_client import LLMClient
from organizer import format_structure_output  # Import from organizer.py


def gather_user_feedback_and_improve(
    llm_client: LLMClient, proposed_structure: dict, analysis_results: list
):
    """Gathers user feedback on the proposed structure and refines it using the LLM."""
    logger.info("Proposed structure:")
    logger.info(format_structure_output(proposed_structure))  # Use format_structure_output

    while True:
        user_input = input(
            "Approve (a), Reject (r), or provide Feedback (f) on the structure: "
        ).lower()

        if user_input == "a":
            logger.info("Structure approved.")
            return proposed_structure
        elif user_input == "r":
            logger.info("Structure rejected. Exiting.")
            exit()
        elif user_input == "f":
            feedback = input("Please provide your feedback on the structure: ")
            logger.info(f"User feedback: {feedback}")
            proposed_structure = llm_client.propose_structure(analysis_results, feedback)
            logger.info("Revised structure based on feedback:")
            logger.info(format_structure_output(proposed_structure))
        else:
            logger.warning("Invalid input. Please enter 'a', 'r', or 'f'.")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Organize files using AI.")
    parser.add_argument(
        "target_folder", type=str, help="Path to the folder containing files to organize."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file (default: config.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory"
    )
    return parser.parse_args()