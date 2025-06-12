import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sksurv.metrics import brier_score

from src.predictions.survival_analysis.cox_timevarying import CoxTimeVaryingFitterModel
from src.visualisations.model_plotting import ModelPlotter


class SurvivalModelTrainer:
    def __init__(self, project_name, model, images_dir, output_dir="models"):
        self.project_name = project_name
        self.model = model(auto_tune_flag=True)

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
            "date", "completion_date", "completion_reason",
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

        n_train_events = (train_df["event"] == 1).sum()
        n_test_events = (test_df["event"] == 1).sum()

        self.logger.debug(f"Train samples: We have {n_train_events}/{len(train_df)} samples with event = 1")
        self.logger.debug(f"Test samples: We have {n_test_events}/{len(test_df)} samples with event = 1")

        feat_cols = self.get_feature_cols(train_df)
        self.logger.info(f"Using features: {feat_cols}")

        if isinstance(self.model, CoxTimeVaryingFitterModel):
            x_train = train_df[feat_cols + ["path"]]
            y_train = train_df[["event", "stop", "start"]]

            x_test = test_df[feat_cols + ["path"]]
            y_test = test_df[["event", "stop", "start"]]

            #x_train = x_train.replace([np.inf, -np.inf], np.nan)
            #x_train = x_train.fillna(0)
            self.model.train(x_train, y_train)
        else:
            groups = train_df["path"].values
            x_train = train_df[feat_cols].values
            y_train = train_df[["duration", "event"]].to_numpy()

            x_test = test_df[feat_cols].values
            y_test = test_df[["duration", "event"]].to_numpy()

            self.model.train(x_train, y_train, groups=groups)

        # 1. Concordance
        c_index = self.model.evaluate(x_test, y_test)
        self.logger.info(f"Survival model concordance: {c_index:.4f}")

        if isinstance(self.model, CoxTimeVaryingFitterModel):
            self.logger.warning("Skipping Brier score, horizon classification, and calibration: "
                                "not supported by CoxTimeVaryingFitter.")
        else:
            # 2. Brier score
            brier = self.evaluate_brier_score(x_test, y_test, times=[30, 90, 180])
            self.logger.info(f"Brier scores (days to score): {brier}")
    
            # 3. Horizon classification (excluding early-censored)
            for h in [30, 90, 180]:
                # drop rows where duration < h and event==0 (uncertain) to avoid bias
                keep_mask = (test_df["duration"] >= h) | (test_df["event"] == 1)
                x_h = x_test[keep_mask.values]
                y_h = y_test[keep_mask.values]
                self.evaluate_horizon_classification(x_h, y_h, horizon=h)

            # 4. Calibration at 90 days
            keep_mask = (test_df["duration"] >= 90) | (test_df["event"] == 1)
            x_cal = x_test[keep_mask.values]
            y_cal = y_test[keep_mask.values]
            self.plot_calibration_curve(x_cal, y_cal, horizon=90)

            # Save per-file evaluation at horizon=90
            eval_df = test_df.loc[keep_mask, ["path", "date", "duration", "event"]].copy()
            x_eval = x_test[keep_mask.values]
            eval_df["predicted_risk"] = self.model.predict_risk(x_eval, horizon=90)
            eval_csv = os.path.join(self.output_dir, "survival_evaluation.csv")
            eval_df.to_csv(eval_csv, index=False)
            self.logger.info(f"Saved survival evaluation to {eval_csv}")

            self.model_plotter.plot_risk_histogram(eval_df, risk_col="predicted_risk")

        # Save model
        model_path = os.path.join(self.output_dir, f"{self.model.__class__.__name__}.pkl")
        self.model.save_model(model_path)

        return {"concordance": c_index, "model_path": model_path}


    def predict_unlabeled_files(self, file_data_df: pd.DataFrame, latest_only: bool = True):
        """
            For any snapshots that were not marked 'event=1' (i.e. right-censored),
            predict a risk score or survival time.
        """
        if isinstance(self.model, CoxTimeVaryingFitterModel):
            self.logger.warning("Censored prediction skipped: not supported by CoxTimeVaryingFitter.")
            return pd.DataFrame()

        censored = file_data_df[file_data_df["event"] == 0].copy()

        if latest_only:
            censored = (
                censored.sort_values("date")
                .groupby("path", as_index=False)
                .tail(1)
                .reset_index(drop=True)
            )

        if censored.empty:
            self.logger.info("No censored snapshots to predict.")
            return pd.DataFrame()

        feat_cols = self.get_feature_cols(censored)
        x_pred = censored[feat_cols].values
        censored["predicted_risk"] = self.model.predict_risk(x_pred, horizon=90)
        out_csv = os.path.join(self.output_dir, "survival_predictions.csv")
        censored.to_csv(out_csv, index=False)
        self.logger.info(f"Saved survival predictions to {out_csv}")

        return censored[["path", "date", "predicted_risk"]]

    def evaluate_brier_score(self, x_test, y_test, times: list[int]):
        """
        Computes Brier score at given times.
        :param x_test:
        :param y_test:
        :param times: a list of times to evaluate
        :return: a dict of {time: brier_score}
        """
        y_test_structured = self.model._format_labels(y_test)
        surv_funcs = self.model.model.predict_survival_function(x_test)
        surv_probs = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
        eval_times, brier_scores = brier_score(y_test_structured, y_test_structured, surv_probs, times)

        return dict(zip(eval_times, brier_scores))

    def evaluate_horizon_classification(self, x_test, y_test, horizon=90, threshold=0.5):
        """
        Binary classification: event within `horizon` days or not.
                Returns metrics dict and logs AUROC, accuracy, confusion matrix.
        :param x_test: test dataset of x
        :param y_test: test dataset of y
        :param horizon: the days as max to look at
        :param threshold:
        :return:
        """
        y_true = (y_test[:, 1] == 1) & (y_test[:, 0] <= horizon)

        # predict risk = P(event ≤ horizon) = 1 - S(horizon)
        surv_funcs = self.model.model.predict_survival_function(x_test)
        risks = np.array([1 - fn(horizon) for fn in surv_funcs])

        auc = roc_auc_score(y_true, risks)
        preds = (risks > threshold).astype(int)
        acc = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)

        self.logger.info(f"Horizon={horizon}d — AUROC: {auc:.4f}, Acc: {acc:.4f}\nConfusion:\n{cm}")

        return {"horizon": horizon, "auc": auc, "accuracy": acc, "confusion_matrix": cm.tolist()}

    def plot_calibration_curve(self, x_test, y_test, horizon=90, bins=10):
        """
        Plot observed vs. predicted event rate at `horizon`.
        :param x_test:
        :param y_test:
        :param horizon:
        :param bins:
        :return:
        """
        y_true = (y_test[:, 1] == 1) & (y_test[:, 0] <= horizon)
        surv_funcs = self.model.model.predict_survival_function(x_test)
        risks = np.array([1 - fn(horizon) for fn in surv_funcs])

        df = pd.DataFrame({"y_true": y_true, "risk": risks})
        df["bin"] = pd.qcut(df["risk"], q=bins, labels=False, duplicates="drop")
        observed_risk = df.groupby("bin")["y_true"].mean()
        predicted_risk = df.groupby("bin")["risk"].mean()

        self.model_plotter.plot_calibration_curve(observed_risk, predicted_risk, horizon)