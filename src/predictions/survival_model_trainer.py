import logging
import os

import numpy as np
import pandas as pd

from src.visualisations.model_plotting import ModelPlotter


class SurvivalModelTrainer:
    def __init__(self, project_name, model, images_dir, output_dir="models"):
        self.project_name = project_name
        self.model = model()

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model_plotter = ModelPlotter(project_name, model, images_dir=images_dir)

        self.logger = logging.getLogger(self.__class__.__name__)

    def split_by_file(self, df: pd.DataFrame, test_ratio=0.2, random_state=42):
        """Keep all snapshots of each file together in train or test."""
        paths = df["path"].unique()
        np.random.seed(random_state)
        np.random.shuffle(paths)

        cutoff = int(len(paths) * (1 - test_ratio))
        train_paths, test_paths = paths[:cutoff], paths[cutoff:]

        train_df = df[df["path"].isin(train_paths)].copy()
        test_df = df[df["path"].isin(test_paths)].copy()
        return train_df, test_df

    def get_feature_cols(self, df: pd.DataFrame, include_size: bool = False):
        """All numeric columns minus identifiers and the survival targets."""
        drop = {
            "path", "date", "completion_date", "completion_reason",
            "duration", "event", "committer", "committer_grouped"
        }
        if not include_size:
            drop.add("size")

        return [
            c for c in df.select_dtypes(include=np.number).columns
            if c not in drop
        ]
    
    def train_and_evaluate(self, file_data_df):
        train_df, test_df = self.split_by_file(file_data_df)

        feat_cols = self.get_feature_cols(train_df)
        self.logger.info(f"Using features: {feat_cols}")

        x_train = train_df[feat_cols].values
        y_train = train_df[["duration", "event"]].to_numpy()

        x_test = test_df[feat_cols].values
        y_test = test_df[["duration", "event"]].to_numpy()

        self.model.train(x_train, y_train)

        metric = self.model.evaluate(x_test, y_test)
        self.logger.info(f"Survival model concordance: {metric:.4f}")

        eval_df = test_df[["path", "date"]].copy()
        eval_df["predicted_risk"] = self.model.predict(x_test)
        eval_df["duration"] = test_df["duration"]
        eval_df["event"] = test_df["event"]
        eval_csv = os.path.join(self.output_dir, "survival_evaluation.csv")
        eval_df.to_csv(eval_csv, index=False)
        self.logger.info(f"Saved survival evaluation to {eval_csv}")

        model_path = os.path.join(self.output_dir, f"{self.model.__class__.__name__}.pkl")
        self.model.save_model(model_path)

        return {"concordance": metric, "model_path": model_path}


    def predict_unlabeled_files(self, file_data_df: pd.DataFrame, latest_only: bool = True):
        """
            For any snapshots that were not marked 'event=1' (i.e. right-censored),
            predict a risk score or survival time.
        """
        unlabeled = file_data_df[file_data_df["event"] == 0].copy()

        if latest_only:
            unlabeled = (
                unlabeled.sort_values("date")
                .groupby("path", as_index=False)
                .tail(1)
                .reset_index(drop=True)
            )

        if unlabeled.empty:
            self.logger.info("No censored snapshots to predict.")
            return pd.DataFrame()

        feat_cols = self.get_feature_cols(unlabeled)
        x_pred = unlabeled[feat_cols].values

        risks = self.model.predict(x_pred)
        unlabeled["predicted_risk"] = risks

        out_csv = os.path.join(self.output_dir, "survival_predictions.csv")
        unlabeled.to_csv(out_csv, index=False)
        self.logger.info(f"Saved survival predictions to {out_csv}")

        return unlabeled[["path", "date", "predicted_risk"]]
