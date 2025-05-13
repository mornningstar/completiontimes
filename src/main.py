import asyncio
import datetime
import logging
import os
import platform

from config.projects import PROJECTS
from src.data_handling.database.api_connection_async import APIConnectionAsync
from src.data_handling.features.file_feature_engineering import FileFeatureEngineer
from src.predictions.file_model_trainer import FileModelTrainer
from src.visualisations.file_visualiser import FileVisualiser

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

async def process_file_visualiser(api_connection, project_name, file_path, commit_visualiser, models, target,
                                        all_file_features, project, cluster_combined_df=None):
    if file_path:
        logging.info(f"Processing file: {file_path}, target: {target} in project: {project_name}")
    else:
        logging.info(f"Processing cluster, target: {target} in project: {project_name}")

    file_visualiser = FileVisualiser(
        api_connection,
        project_name,
        file_path,  # No specific file_path for cluster mode
        commit_visualiser,
        models,
        [target],
        all_file_features,
        cluster_combined_df=cluster_combined_df,
        )

    try:
        await file_visualiser.run()

        horizon = project['file_modeling_tasks'][target]['horizon']
        threshold = project['file_modeling_tasks'][target]['threshold']
        consecutive_days = project['file_modeling_tasks'][target]['consecutive_days']

        completion_date = await file_visualiser.predict_completion(target, horizon, threshold, consecutive_days)

        if completion_date:
            logging.info(f"Predicted completion date for {file_path}: {completion_date}")
        else:
            logging.info(f"No completion detected for {file_path} within {horizon} days.")

    except Exception as e:
        logging.error(f"Error processing file/cluster for target {target}. The error: {e}", exc_info=True)


async def process_project(project):
    project_name = project['name']
    models = project.get('models', [])

    api_connection = await APIConnectionAsync.create(project_name)

    try:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_output_dir = os.path.join(project_name, "runs", timestamp)
        os.makedirs(run_output_dir, exist_ok=True)

        images_dir = os.path.join(run_output_dir, "images")
        logging.info("images_dir: {}".format(images_dir))
        models_dir = os.path.join(run_output_dir, "models")

        logging.info(f"Starting processing for project: {project_name}")
        await api_connection.populate_db()

        feature_engineer = FileFeatureEngineer(api_connection, project_name, threshold=0.05, consecutive_days=14,
                                                images_dir=images_dir)
        file_features = await feature_engineer.run()

        for model in models:
            logging.info(f"Using {model.__class__.__name__}")
            file_model_trainer = FileModelTrainer(project_name, model, images_dir=images_dir, output_dir=models_dir)
            file_model_trainer.train_and_evaluate(file_features)
            result = file_model_trainer.predict_unlabeled_files(file_features)

    except Exception:
        logging.exception('Error while processing project {}'.format(project_name))
    finally:
        logging.info('Project {} finished!'.format(project_name))
        await api_connection.http_client.clos


async def main():
    tasks = [process_project(project) for project in PROJECTS]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
