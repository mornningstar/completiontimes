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
from src.data_handling.features.base_feature_engineer import ALL_FEATURE_GROUPS
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

    is_ablation_study = project.get('ablation', False)

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

        if not is_ablation_study:
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

        else:
            ablation_results = []

            ablation_configs = [{"name": "ALL", "include": ALL_FEATURE_GROUPS}]
            for feature in ALL_FEATURE_GROUPS:
                ablation_configs.append({
                    "name": f"all_except_{feature}",
                    "include": [f for f in ALL_FEATURE_GROUPS if f != feature]
                })

            for ablation in ablation_configs:
                logging.info("Running ablation study: {}".format(ablation["name"]))

                for model_cfg in models:
                    model_cls = model_cfg["class"]
                    flag = model_cfg.get("use_categorical", False)
                    feature_type = model_cfg.get("feature_type", "regression")
                    eng_cls = ENGINEER_BY_TYPE[feature_type]

                    ablation_images_dir = os.path.join(images_dir, ablation["name"])
                    ablation_models_dir = os.path.join(models_dir, ablation["name"])
                    os.makedirs(ablation_images_dir, exist_ok=True)
                    os.makedirs(ablation_models_dir, exist_ok=True)

                    engineer = eng_cls(file_repo, plotter, use_categorical=flag)
                    runner = FeatureEngineerRunner(engineer)
                    engineered_df = await runner.run(source_directory=source_directory, include_sets=ablation["include"])

                    logging.info(f"Computed ablation features for {ablation['name']} - rows = {len(engineered_df)}")

                    trainer_cls = TRAINER_BY_TYPE[feature_type]
                    trainer = trainer_cls(project_name, model_cls, ablation_images_dir, ablation_models_dir)
                    metrics = trainer.train_and_evaluate(engineered_df)
                    filtered_metrics = {
                        "MAE": metrics["mae"],
                        "RMSE": metrics["rmse"]
                    }
                    trainer.predict_unlabeled_files(engineered_df, latest_only=True)

                    result_row = {
                        "project": project_name,
                        "timestamp": timestamp,
                        "configuration": ablation["name"],
                        "model": model_cls.__name__,
                        **filtered_metrics
                    }

                    ablation_results.append(result_row)

        if is_ablation_study and ablation_results:
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
    tasks = [process_project(project, shared_token_bucket) for project in PROJECTS]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
