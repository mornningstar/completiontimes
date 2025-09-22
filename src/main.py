import asyncio
import datetime
import logging
import os
import platform

import pandas as pd
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

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

setup_logging()

with open("../config/config.yml", "r") as f:
    config = yaml.safe_load(f)

for project in config["projects"]:
    for model in project["models"]:
        model["class"] = get_model_class(model["class"])

async def process_project(project, token_bucket: TokenBucket = None):
    project_name = project['name']
    models = project.get('models', [])
    get_newest = project.get('get_newest', True)
    source_directory = project.get('source_directory', "src")

    is_ablation_study = project.get('ablation', False)

    auth_token = os.path.expandvars(config['github']['token'])

    try:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_output_dir = os.path.join("runs", project_name, timestamp)
        os.makedirs(run_output_dir, exist_ok=True)

        images_dir = os.path.join(run_output_dir, "images")
        logging.info("images_dir: {}".format(images_dir))
        models_dir = os.path.join(run_output_dir, "models")
        
        if get_newest:
            orchestrator = SyncOrchestrator(auth_token, project_name, token_bucket)
            logging.info(f"Starting processing for project: {project_name}")
            await orchestrator.run()
            logging.debug("Finished calling synchronised orchestrator")
        else:
            logging.info("Skipping fetch of new commit history")

        file_repo = FileRepository(project_name)
        plotter = ModelPlotter(project_name, images_dir=images_dir)
        AsyncDatabase.URI = config['mongo']['uri']
        AsyncDatabase.DATABASE_NAME = config['mongo']['database']
        await AsyncDatabase.initialize()

        feature_pipeline = FeatureEngineeringPipeline(file_repo, plotter, source_directory)

        if not is_ablation_study:
            training_pipe = ModelTrainingPipeline(project_name, models, feature_pipeline, images_dir, models_dir)
            await training_pipe.run()

        else:
            ablation_study = AblationStudy(
                project_name,
                file_repo,
                plotter,
                images_dir,
                models_dir,
                source_directory,
                timestamp
            )

            ablation_results = await ablation_study.run(models=models)

            if ablation_results:
                results_df = pd.DataFrame(ablation_results)
                results_path = os.path.join(run_output_dir, "results.csv")
                results_df.to_csv(results_path, index=False)
                logging.info(f"Ablation study results saved to {results_path}")

    except Exception:
        logging.exception('Error while processing project {}'.format(project_name))
    finally:
        logging.info('Project {} finished!'.format(project_name))


async def main():
    shared_token_bucket = TokenBucket()
    tasks = [process_project(project, shared_token_bucket) for project in config["projects"]]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
