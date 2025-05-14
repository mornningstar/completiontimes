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

async def process_project(project):
    project_name = project['name']
    models = project.get('models', [])

    #api_connection = APIConnectionAsync.create(project_name)
    token = CONFIG[0]['github_access_token']
    orchestrator = SyncOrchestrator(token, project_name)

    try:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_output_dir = os.path.join("runs", project_name, timestamp)
        os.makedirs(run_output_dir, exist_ok=True)

        images_dir = os.path.join(run_output_dir, "images")
        logging.info("images_dir: {}".format(images_dir))
        models_dir = os.path.join(run_output_dir, "models")

        logging.info(f"Starting processing for project: {project_name}")
        await orchestrator.run()

        file_repo = FileRepository(project_name)
        plotter = ModelPlotter(project_name, images_dir=images_dir)
        engineer = FileFeatureEngineer(file_repo, plotter, threshold=0.05, consecutive_days=14)
        
        logging.info(f"Running feature engineering for project: {project_name}")
        file_features = await engineer.run()
        logging.info(f"Finished feature engineering for project: {project_name}")

        for model in models:
            logging.info(f"Using {model}")
            file_model_trainer = FileModelTrainer(project_name, model, images_dir=images_dir, output_dir=models_dir)
            file_model_trainer.train_and_evaluate(file_features)
            file_model_trainer.predict_unlabeled_files(file_features)

    except Exception:
        logging.exception('Error while processing project {}'.format(project_name))
    finally:
        logging.info('Project {} finished!'.format(project_name))


async def main():
    tasks = [process_project(project) for project in PROJECTS]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
