import asyncio
import logging
import os

import pandas as pd
from dotenv import load_dotenv

from src.data_handling.data_loader import DataLoader
from src.data_handling.database.async_database import AsyncDatabase
from src.data_handling.database.file_repo import FileRepository
from src.data_handling.features.completion_date_labler import CompletionDateLabler
from src.data_handling.features.generators.commit_history_feature_generator import CommitHistoryFeatureGenerator
from src.logging_config import setup_logging

list_of_projects = ['flairNLP/fundus', 'khoj-ai/khoj', 'vuejs/core', 'mozilla/addons-server',
                        'fastapi/fastapi', 'pallets/flask', 'keras-team/keras', 'tabler/tabler',
                        'google/material-design-lite', 'google/gson']

async def main():
    """
    Minimal script to load data, run only the necessary feature generation,
    and calculate target variable statistics.
    """
    setup_logging(level=logging.INFO)

    # --- CONFIGURE THIS ---
    # Load .env file from the parent directory (config/.env)
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    for PROJECT_NAME in list_of_projects:
        # Set your DB config
        AsyncDatabase.URI = "mongodb://localhost:27018"  # <--- SET YOUR MONGO_URI
        AsyncDatabase.DATABASE_NAME = "github_data"  # <--- SET YOUR DB_NAME
        # --- END CONFIGURATION ---

        if not AsyncDatabase.URI:
            logging.error("MONGO_URI not set. Please configure it in config/.env")
            return

        await AsyncDatabase.initialize()
        logging.info(f"Database initialized. Target project: {PROJECT_NAME}")

        file_repo = FileRepository(PROJECT_NAME)
        loader = DataLoader(file_repo)

        logging.info("Fetching raw data from database...")
        try:
            df = await loader.fetch_all_files()
            if df.empty:
                logging.error("No data loaded. Check project name and database connection.")
                return
            logging.info(f"Successfully loaded {len(df)} total commit-file rows.")
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            return

        # Filter out files with < 5 commits, as your main pipeline does
        df = df.groupby("path").filter(lambda g: len(g) >= 5)
        if df.empty:
            logging.error("No files remaining after < 5 commit filter.")
            return
        logging.info(f"{len(df)} rows remaining after 5-commit filter.")

        # 1. Run the *only* required feature generator
        logging.info("Running minimal feature generation (CommitHistoryFeatureGenerator)...")
        history_gen = CommitHistoryFeatureGenerator()
        df, _ = history_gen.generate(df)
        logging.info("Commit history features generated.")

        # 2. Run the completion labler
        logging.info("Running completion labeling...")
        labler = CompletionDateLabler()
        # The .label() method finds the completion_date
        df, _, _, _ = labler.label(df)
        # This method calculates the final target column
        df = labler.add_days_until_completion(df)
        logging.info("Labeling complete.")

        # 3. Calculate statistics
        logging.info("--- Calculating Target Variable Statistics ---")
        valid_targets = df["days_until_completion"].dropna()

        if valid_targets.empty:
            logging.error("No valid 'days_until_completion' targets found.")
            return

        median = valid_targets.median()
        q1 = valid_targets.quantile(0.25)
        q3 = valid_targets.quantile(0.75)
        iqr = q3 - q1

        print("\n" + "=" * 40)
        print(f"Statistics for: {PROJECT_NAME}")
        print(f"Total valid data points: {len(valid_targets)}")
        print(f"Median (Q2): {median:.2f} days")
        print(f"25th Percentile (Q1): {q1:.2f} days")
        print(f"75th Percentile (Q3): {q3:.2f} days")
        print(f"Interquartile Range (IQR): {iqr:.2f} days")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    # Ensure you have 'python-dotenv' installed: pip install python-dotenv
    asyncio.run(main())