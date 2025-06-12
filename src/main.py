import asyncio
import datetime
import logging
import os
import platform

import pandas as pd

from config.config import CONFIG
from config.projects import PROJECTS
from src.data_handling.database.async_database import AsyncDatabase
from src.data_handling.database.file_repo import FileRepository
from src.data_handling.features.feature_engineer_runner import FeatureEngineerRunner
from src.data_handling.features.regression_feature_eng import RegressionFeatureEngineering
from src.data_handling.features.survival_feature_engineer import SurvivalFeatureEngineer
from src.data_handling.service.sync_orchestrator import SyncOrchestrator
from src.github.token_bucket import TokenBucket
from src.predictions.training.regression_model_trainer import RegressionModelTrainer
from src.predictions.training.survival_model_trainer import SurvivalModelTrainer
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

ENGINEER_BY_TYPE = {
    "regression": RegressionFeatureEngineering,
    "survival":   SurvivalFeatureEngineer,
}

TRAINER_BY_TYPE = {
    "regression": RegressionModelTrainer,
    "survival":   SurvivalModelTrainer,
}

async def process_project(project, token_bucket: TokenBucket = None):
    project_name = project['name']
    models = project.get('models', [])
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

        features_cache: dict[tuple[type, bool], pd.DataFrame] = {}
        
        for model_cfg in models:
            model_cls = model_cfg["class"]
            flag = model_cfg.get("use_categorical", False)
            feature_type = model_cfg.get("feature_type", "regression")
            eng_cls = ENGINEER_BY_TYPE[feature_type]

            cache_key = (eng_cls, flag)

            if cache_key not in features_cache:
                engineer = eng_cls(file_repo, plotter, use_categorical=flag)
                runner = FeatureEngineerRunner(engineer)
                engineered_df = await runner.run(source_directory=source_directory)
                features_cache[cache_key] = engineered_df
                logging.info(f"Computed features with {eng_cls.__name__} (use_categorical={flag}) "
                             f"– rows={len(engineered_df)}")

            features_to_use = features_cache[cache_key]

            trainer_cls = TRAINER_BY_TYPE[feature_type]

            trainer = trainer_cls(project_name, model_cls, images_dir=images_dir, output_dir=models_dir)
            trainer.train_and_evaluate(features_to_use)
            trainer.predict_unlabeled_files(features_to_use, latest_only=True)

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
