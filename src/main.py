import asyncio
import datetime
import logging
import os
import platform
from pathlib import Path

from dotenv import load_dotenv

from src.data_handling.database.commit_repo import CommitRepository
from src.data_handling.service.commit_sync_service import CommitSyncService
from src.data_handling.service.file_history_service import FileHistoryService
from src.github.http_client import GitHubClient

import yaml

from src.data_handling.database.async_database import AsyncDatabase
from src.data_handling.database.file_repo import FileRepository
from src.data_handling.service.sync_orchestrator import SyncOrchestrator
from src.github.token_bucket import TokenBucket
from src.logging_config import setup_logging
from src.pipeline.ablation import AblationStudy
from src.pipeline.pipeline_feature_engineering import FeatureEngineeringPipeline
from src.pipeline.pipeline_model_training import ModelTrainingPipeline
from src.predictions.registry import get_model_class
from src.visualisations.model_plotting import ModelPlotter

def configure_event_loop():
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def load_app_config():
    base_dir = Path(__file__).parent
    config_path = base_dir / "../config/config.yml"
    try:
        load_dotenv(dotenv_path=base_dir / "../config/.env")
        config = yaml.safe_load(config_path.read_text())

        for project in config["projects"]:
            for model in project["models"]:
                model["class"] = get_model_class(model["class"])

        return config
    except FileNotFoundError:
        logging.error(f"CRITICAL: Config file not found at {config_path}")
        raise

async def run_data_fetching(project, db: AsyncDatabase, config: dict, token_bucket: TokenBucket = None):
    project_name = project['name']
    get_newest = project.get('get_newest', True)

    client_cfg = config.get('github', {}).get('client', {})
    auth_token = os.path.expandvars(config['github']['token'])

    if get_newest:
        http_client = GitHubClient(auth_token,
                                   token_bucket,
                                   concurrency=client_cfg.get('concurrency', 100),
                                   timeout=client_cfg.get('timeout_seconds', 300))

        commit_repo = CommitRepository(project_name, db)
        file_repo = FileRepository(project_name, db)
        commit_service = CommitSyncService(http_client, project_name, commit_repo)
        file_service = FileHistoryService(http_client, project_name, file_repo, commit_service)

        orchestrator = SyncOrchestrator(project_name, http_client, commit_service, file_service)
        logging.info(f"Starting processing for project: {project_name}")
        await orchestrator.run()
        logging.debug("Finished calling synchronised orchestrator")
    else:
        logging.info("Skipping fetch of new commit history")

async def run_model_training(project, db: AsyncDatabase, config: dict):
    labelling_cfg = config.get('labelling', {})

    project_name = project['name']
    models = project.get('models', [])
    source_directory = project.get('source_directory', "src")
    is_ablation_study = project.get('ablation', False)

    logging.info(f"--- Starting model training for: {project_name} ---")

    try:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_output_dir = os.path.join("runs", project_name, timestamp)
        os.makedirs(run_output_dir, exist_ok=True)

        images_dir = os.path.join(run_output_dir, "images")
        logging.info("images_dir: {}".format(images_dir))
        models_dir = os.path.join(run_output_dir, "models")

        master_results_path = os.path.join(run_output_dir, "results.csv")

        file_repo = FileRepository(project_name, db)
        plotter = ModelPlotter(project_name, images_dir=images_dir)

        feature_pipeline = FeatureEngineeringPipeline(file_repo,
                                                      plotter,
                                                      source_directory,
                                                      labelling_config=labelling_cfg)

        if not is_ablation_study:
            training_pipe = ModelTrainingPipeline(project_name, models, feature_pipeline, images_dir, models_dir,
                                                  master_results_path, timestamp)
            await training_pipe.run()
        else:
            ablation_study = AblationStudy(
                project_name,
                file_repo,
                plotter,
                images_dir,
                models_dir,
                source_directory,
                timestamp,
                master_results_path
            )

            await ablation_study.run(models=models)
            logging.info(f"Ablation study for {project_name} complete. Results saved to {master_results_path}")


    except Exception as e:
        logging.exception(f"Critical unexpected error processing {project_name}: {e}")
    finally:
        logging.info('Project {} finished!'.format(project_name))


async def main():
    configure_event_loop()
    setup_logging()
    config = load_app_config()

    shared_token_bucket = TokenBucket()
    db = AsyncDatabase(config['mongo']['uri'], config['mongo']['database'])

    tasks = [run_data_fetching(project, db, config, shared_token_bucket) for project in config["projects"]]
    await asyncio.gather(*tasks)

    for project in config["projects"]:
        await run_model_training(project, db, config)

if __name__ == '__main__':
    asyncio.run(main())
