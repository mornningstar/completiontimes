import asyncio
import datetime
import logging
import os
import platform

from config.config import CONFIG
from config.projects import PROJECTS
from src.data_handling.database.async_database import AsyncDatabase
from src.data_handling.database.file_repo import FileRepository
from src.data_handling.features.file_feature_engineering import FileFeatureEngineer
from src.data_handling.service.sync_orchestrator import SyncOrchestrator
from src.github.token_bucket import TokenBucket
from src.predictions.file_model_trainer import FileModelTrainer
from src.visualisations.model_plotting import ModelPlotter

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('pymongo').setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Adjust the level as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

async def process_project(project, token_bucket: TokenBucket = None):
    project_name = project['name']
    models = project.get('models', [])
    flags = {cfg.get("use_categorical", False) for cfg in models}
    get_newest = project.get('get_newest', True)
    source_directory = project.get('source_directory', "src")

    auth_token = CONFIG[0]['github_access_token']

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
        await AsyncDatabase.initialize()

        features_by_flag = {}

        for flag in flags:
            engineer = FileFeatureEngineer(file_repo, plotter, use_categorical=flag)
            features = await engineer.run(source_directory=source_directory)
            logging.debug(f"Length of file_features: {len(features)}")
            features_by_flag[flag] = features
            logging.debug(f"Finished feature engineering for project: {project_name}")

        for model in models:
            model_class = model["class"]
            flag = model.get("use_categorical", False)
            features_to_use = features_by_flag[flag]

            logging.info(f"Training {model_class.__name__} (use_categorical={flag})")

            file_model_trainer = FileModelTrainer(project_name, model_class, images_dir=images_dir, output_dir=models_dir)
            file_model_trainer.train_and_evaluate(features_to_use)
            file_model_trainer.predict_unlabeled_files(features_to_use)

    except Exception:
        logging.exception('Error while processing project {}'.format(project_name))
    finally:
        logging.info('Project {} finished!'.format(project_name))


async def main():
    shared_token_bucket = TokenBucket()
    tasks = [process_project(project, shared_token_bucket) for project in PROJECTS]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
