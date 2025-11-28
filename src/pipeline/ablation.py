import asyncio
import logging

import pandas as pd

from src.data_handling.features.feature_engineer_runner import FeatureEngineerRunner
from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.factories.trainer_factory import TrainerFactory
from src.pipeline.configs import ENGINEER_BY_TYPE
from src.pipeline.results_writer import append_to_master_results


class AblationStudy:
    def __init__(self, project_name, file_repo, plotter, images_dir, models_dir, source_directory, timestamp,
                 master_results_path, labelling_config=None):
        self.project_name = project_name
        self.file_repo = file_repo
        self.plotter = plotter
        self.images_dir = images_dir
        self.models_dir = models_dir
        self.source_directory = source_directory
        self.timestamp = timestamp
        self.master_results_path = master_results_path
        self._features_cache = {}
        self.factory = TrainerFactory()
        self.labelling_config = labelling_config

    async def _get_or_create_features(self, model_cfg):
        """
        Efficiently gets or creates the full feature set for a given model configuration.
        Caches the result to avoid re-computation.
        """
        feature_type = model_cfg.get("feature_type", "regression")
        eng_cls = ENGINEER_BY_TYPE[feature_type]

        cache_key = eng_cls
        if cache_key not in self._features_cache:
            logging.info(f"Cache missing for {cache_key}. Generating full feature set...")
            engineer = eng_cls(self.file_repo, self.plotter, labelling_config=self.labelling_config)
            runner = FeatureEngineerRunner(engineer)
            engineered_df, categorical_cols = await runner.run(
                source_directory=self.source_directory, include_sets=feature_generator_registry.get_all_names()
            )
            self._features_cache[cache_key] = (engineered_df, categorical_cols)
            logging.info(f"Cached features for {cache_key} - rows = {len(engineered_df)}")

        return self._features_cache[cache_key]

    async def run(self, models):
        ablation_results = []
        all_groups = feature_generator_registry.get_all_names()
        ablation_configs = [{"name": "ALL", "include": all_groups}]

        for feature in all_groups:
            ablation_configs.append({
                "name": f"all_except_{feature}",
                "include": [f for f in all_groups if f != feature]
            })

        for model_cfg in models:
            master_df, all_categorical_cols = await self._get_or_create_features(model_cfg)
            data_split = model_cfg.get("split_strategy", "by_file")

            for ablation in ablation_configs:
                model_name = model_cfg['class'].__name__
                logging.info(f"Running ablation study: {ablation['name']} for model: {model_name}")

                columns_to_use = self._get_columns_for_ablation(master_df, ablation["include"])
                ablation_df = master_df[columns_to_use]

                ablation_categorical_cols = [col for col in all_categorical_cols if col in ablation_df.columns]

                trainer = self.factory.create_trainer(
                    project_name=self.project_name,
                    model_cfg=model_cfg,
                    images_dir=self.images_dir,
                    models_dir=self.models_dir,
                    ablation_name=ablation["name"]
                )

                data_to_pass = (ablation_df, ablation_categorical_cols)
                training_result = await asyncio.to_thread(trainer.train_and_evaluate, data_to_pass)

                result_data = {
                    "project": self.project_name,
                    "model": model_name,
                    "split_strategy": data_split,
                    "configuration": ablation["name"],
                    "timestamp": self.timestamp,
                }

                if isinstance(training_result, dict):
                    result_data.update(training_result)
                else:
                    result_data.update(vars(training_result))

                ablation_results.append(result_data)
                append_to_master_results(result_data.copy(), self.master_results_path)
                logging.debug(f"Appended results for {model_name} / {ablation['name']} to {self.master_results_path}")

        return ablation_results

    def _get_columns_for_ablation(self, df: pd.DataFrame, include_groups: list[str]):
        essential_cols = {
            "path", "date", "completion_date", "completion_reason", "committer", "committer_grouped",
            "days_until_completion"
        }
        selected_cols = {col for col in df.columns if col in essential_cols}

        for group_name in include_groups:
            generator = feature_generator_registry.get(group_name)
            if generator:
                feature_names = generator.get_feature_names(df)
                for item in feature_names:
                    if item.endswith('_'):
                        selected_cols.update(df.columns[df.columns.str.startswith(item)])
                    elif item in df.columns:
                        selected_cols.add(item)

            else:
                logging.warning(f"Feature group '{group_name}' not found in ALL_FEATURE_GENERATORS. Skipping.")

        return list(selected_cols)
