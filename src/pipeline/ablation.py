import logging
import os

from src.data_handling.features.base_feature_engineer import ALL_FEATURE_GROUPS
from src.data_handling.features.feature_engineer_runner import FeatureEngineerRunner
from src.pipeline.configs import ENGINEER_BY_TYPE, TRAINER_BY_TYPE


class AblationStudy:
    def __init__(self, project_name, file_repo, plotter, images_dir, models_dir, source_directory, timestamp):
        self.project_name = project_name
        self.file_repo = file_repo
        self.plotter = plotter
        self.images_dir = images_dir
        self.models_dir = models_dir
        self.source_directory = source_directory
        self.timestamp = timestamp

    async def run(self, models):
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
                flag = model_cfg.get("use_categorical", False)
                feature_type = model_cfg.get("feature_type", "regression")
                eng_cls = ENGINEER_BY_TYPE[feature_type]

                ablation_images_dir = os.path.join(self.images_dir, ablation["name"])
                ablation_models_dir = os.path.join(self.models_dir, ablation["name"])
                os.makedirs(ablation_images_dir, exist_ok=True)
                os.makedirs(ablation_models_dir, exist_ok=True)

                engineer = eng_cls(self.file_repo, self.plotter, use_categorical=flag)
                runner = FeatureEngineerRunner(engineer)
                engineered_df = await runner.run(
                    source_directory=self.source_directory, include_sets=ablation["include"]
                )

                logging.info(f"Computed ablation features for {ablation['name']} - rows = {len(engineered_df)}")

                trainer_cls = TRAINER_BY_TYPE[feature_type]
                trainer = trainer_cls(self.project_name, model_cfg["class"], ablation_images_dir, ablation_models_dir)
                training_result = trainer.train_and_evaluate(engineered_df)

                result_row = {
                    "project": self.project_name,
                    "configuration": ablation["name"],
                    **training_result,
                    "timestamp": self.timestamp,
                }

                trainer.predict_unlabeled_files(engineered_df, latest_only=True)

                ablation_results.append(result_row)

        return ablation_results
