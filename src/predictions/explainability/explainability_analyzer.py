import logging

import numpy as np
import shap
from matplotlib import pyplot as plt
from shap import TreeExplainer, LinearExplainer
from shap.utils._exceptions import InvalidModelError
from sklearn.inspection import PartialDependenceDisplay


class ExplainabilityAnalyzer:
    def __init__(self, model, feature_names, model_plotter):
        self.model = model
        self.feature_names = feature_names
        self.model_plotter = model_plotter

        self.logging = logging.getLogger(self.__class__.__name__)

    def _get_shap_explainer(self, X_background=None):
        if self.model.model is None:
            self.logging.warning("Explainability skipped: model.model is None.")
            return None

        try:
            # For tree-based models
            return TreeExplainer(self.model.model)
        except InvalidModelError:
            pass  # Try LinearExplainer next

            try:
                # For linear models
                return LinearExplainer(self.model.model, X_background if X_background is not None else "auto")
            except Exception as e:
                self.logging.warning(f"Could not initialize SHAP explainer: {e}")
                return None
    
    def analyze_worst_predictions(self, errors_df, top_n=3):
        worst_preds = errors_df.sort_values("abs_error", ascending=False).head(top_n)
        X_worst = worst_preds[self.feature_names]

        explainer = self._get_shap_explainer(X_background=X_worst)
        if not explainer:
            return

        shap_values = explainer.shap_values(X_worst)

        for i in range(top_n):
            residual_error = worst_preds.iloc[i]['residual']
            title = f"SHAP for Worst Prediction #{i + 1} (Error: {residual_error:.2f} days)"
            filename = f"worst_prediction_{i + 1}_shap_bar.png"

            self.model_plotter.plot_shap_bar(
                shap_values[i],
                feature_names=self.feature_names,
                title=title,
                filename=filename
            )

    def analyze_best_predictions(self, errors_df, top_n=3):
        best_preds = errors_df.sort_values("abs_error", ascending=True).head(top_n)
        X_best = best_preds[self.feature_names]

        explainer = self._get_shap_explainer(X_background=X_best)
        if explainer is None:
            self.logging.warning("Skipping interaction analysis: Could not get SHAP explainer.")
            return

        shap_values = explainer.shap_values(X_best)

        for i in range(top_n):
            residual_error = best_preds.iloc[i]['residual']
            title = f"SHAP for Best Prediction #{i + 1} (Error: {residual_error:.2f} days)"
            filename = f"best_prediction_{i + 1}_shap_bar.png"

            self.model_plotter.plot_shap_bar(
                shap_values[i],
                feature_names=self.feature_names,
                title=title,
                filename=filename
            )

    def analyze_feature_interactions(self, X, top_n_features=3):
        self.logging.info("Analyzing SHAP feature interactions...")

        explainer = self._get_shap_explainer(X_background=X)
        if explainer is None:
            self.logging.warning("Skipping interaction analysis: Could not get SHAP explainer.")
            return

        shap_values_full = explainer.shap_values(X)

        mean_abs_shap = np.abs(shap_values_full).mean(axis=0)
        top_feature_indices = np.argsort(mean_abs_shap)[-top_n_features:]

        try:
            shap_interaction_values = explainer.shap_interaction_values(X)
        except Exception as e:
            self.logging.warning(f"Could not compute SHAP interaction values: {e}")
            return

        main_feature_idx = top_feature_indices[-1]
        main_feature_name = self.feature_names[main_feature_idx]

        for i in range(top_n_features - 1):
            interaction_feature_idx = top_feature_indices[i]
            interaction_feature_name = self.feature_names[interaction_feature_idx]

            plt.figure()  # Ensure a new figure is created
            shap.dependence_plot(
                (main_feature_name, interaction_feature_name),
                shap_interaction_values, X,
                display_features=X,
                show=False
            )

            plt.tight_layout()
            filename = f"shap_interaction_{main_feature_name}_vs_{interaction_feature_name}.png"
            self.model_plotter.save_plot(filename)


        self.logging.info("Plotting specific hard-coded feature interactions...")
        hard_coded_pairs = [
            ('age_in_days', 'lag_1_size'),
            ('commit_interval_days', 'age_in_days')
        ]

        for feat1, feat2 in hard_coded_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                try:
                    plt.figure()
                    shap.dependence_plot(
                        (feat1, feat2),
                        shap_interaction_values, X,
                        # display_features=X_df,
                        show=False
                    )
                    plt.tight_layout()
                    filename = f"shap_interaction_manual_{feat1}_vs_{feat2}.png"
                    self.model_plotter.save_plot(filename)
                except Exception as e:
                    self.logging.warning(f"Failed to plot manual interaction for {feat1} vs {feat2}: {e}")

            else:
                self.logging.warning(
                    f"Skipping interaction plot for {feat1} vs {feat2}: One or both features not found in data.")


    def analyze_shap_by_committer(self, errors_df, top_n_committers=5):
        if "committer_grouped" not in errors_df.columns:
            self.logging.warning("Skipping SHAP analysis by committer: 'committer_grouped' column not found.")
            return

        explainer = self._get_shap_explainer()
        if explainer is None:
            return  
        
        top_committers = errors_df["committer_grouped"].value_counts().head(top_n_committers).index

        for committer in top_committers:
            subset = errors_df[errors_df["committer_grouped"] == committer]
            if subset.empty:
                continue

            X = subset[self.feature_names].values
            shap_values = explainer.shap_values(X)

            title_suffix = f"Committer: {committer}"
            self.model_plotter.plot_shap_summary(shap_values, X, self.feature_names, title=title_suffix,
                                                 filename=f"top_errors_shap_summary_{committer}.png")
            self.model_plotter.plot_shap_bar(shap_values[0], self.feature_names, title=title_suffix + " (bar)")


    def analyze_error_sources(self, errors_df, top_n=15):
        errors_df["extension"] = errors_df["path"].str.extract(r"\.([a-zA-Z0-9]+)$")[0].fillna("no_ext")
        errors_df["top_dir"] = errors_df["path"].str.split("/").str[0].fillna("root")

        top_errors = errors_df.sort_values("abs_error", ascending=False).head(top_n)
        self.logging.info("Top errors:\n%s", top_errors[["path", "date", "actual", "pred", "residual"]])

        def get_error_stats(group_by_col):
            stats = (errors_df.groupby(group_by_col)["abs_error"].agg(['mean', 'std']).sort_values(
                by="mean", ascending=False).head(top_n))
            return stats['mean'], stats['std']

        mae_by_ext, std_by_ext = get_error_stats("extension")
        mae_by_dir, std_by_dir = get_error_stats("top_dir")
        mae_by_reason, std_by_reason = get_error_stats("completion_reason")
        if "committer_grouped" in errors_df.columns:
            mae_by_committer, std_by_committer = get_error_stats("committer_grouped")
        else:
            mae_by_committer, std_by_committer = (None, None)

        mae_by_bins, std_by_bins = get_error_stats("true_bin")

        self.model_plotter.plot_bar(mae_by_ext, title="MAE per file type", xlabel="File Extension", ylabel="MAE",
                                    yerr=std_by_ext)
        self.model_plotter.plot_bar(mae_by_dir, title="MAE per top level directory", xlabel="Directory",
                                    ylabel="MAE", yerr=std_by_dir, filename="mae_per_top_dir.png")
        self.model_plotter.plot_bar(mae_by_reason, title="MAE per reason", xlabel="Completion Reason",
                                    ylabel="MAE", yerr=std_by_reason, filename="mae_per_completion_reason.png")
        if mae_by_committer is not None:
            self.model_plotter.plot_bar(mae_by_committer, title="MAE per Committer", xlabel="Committer", ylabel="MAE",
                                        yerr=std_by_committer, filename="mae_per_committer.png")
        self.model_plotter.plot_bar(mae_by_bins, title="MAE per actual days", xlabel="Completion days bins",
                                    ylabel="MAE", yerr=std_by_bins, filename="mae_per_completion_bins.png")

        self.model_plotter.plot_violin(errors_df, x="extension", y="abs_error",
                                       title="Error Distribution per File Type",
                                       xlabel="File Extension", ylabel="Absolute Error",
                                       filename="error_dist_by_ext.png")

        self.model_plotter.plot_violin(errors_df, x="top_dir", y="abs_error",
                                       title="Error Distribution per Top Level Directory",
                                       xlabel="Directory", ylabel="Absolute Error",
                                       filename="error_dist_by_dir.png")

    def analyze_pdp_ice(self, X, top_n_features=5):
        if not hasattr(self.model.model, "feature_importances_"):
            self.logging.warning("Model does not support feature importances, skipping PDP/ICE plots.")
            return

        importances = self.model.model.feature_importances_
        top_feature_indices = np.argsort(importances)[-top_n_features:]

        self.logging.info(f"Generating PDP/ICE plots for top {top_n_features} features...")

        for feature_idx in top_feature_indices:
            feature_name = self.feature_names[feature_idx]

            fig, ax = plt.subplots(figsize=(8,6))

            PartialDependenceDisplay.from_estimator(
                self.model.model,
                X,
                features=[feature_idx],
                feature_names=self.feature_names,
                kind="both",
                subsample=50,
                ice_lines_kw={"color": "blue", "alpha": 0.2, "linewidth": 0.5},
                pd_line_kw={"color": "red", "linestyle": "--", "linewidth": 2},
                ax=ax
            )

            filename = f"pdp_ice_{feature_name}.png"
            self.model_plotter.save_plot(filename)
