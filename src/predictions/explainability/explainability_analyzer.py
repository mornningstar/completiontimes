import logging

import numpy as np
import shap
from matplotlib import pyplot as plt
from shap import TreeExplainer, LinearExplainer
from shap.utils._exceptions import InvalidModelError
from sklearn.inspection import PartialDependenceDisplay


class ExplainabilityAnalyzer:
    def __init__(self, model, feature_names, categorical_features, model_plotter):
        self.model = model
        self.feature_names = feature_names
        self.categorical_features = set(categorical_features or [])
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

            fig, ax = plt.subplots()
            
            shap.dependence_plot(
                (main_feature_name, interaction_feature_name),
                shap_interaction_values, X,
                display_features=X,
                show=False,
                ax=ax
            )

            ax.set_ylabel("")
            plt.title(f"SHAP Interaction: '{main_feature_name}' vs '{interaction_feature_name}'")
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
                    fig, ax = plt.subplots()
                    
                    shap.dependence_plot(
                        (feat1, feat2),
                        shap_interaction_values, X,
                        show=False,
                        ax=ax
                    )

                    ax.set_ylabel("")
                    ax.set_title(f"SHAP Interaction: '{feat1}' vs '{feat2}'")
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
            #self.model_plotter.plot_shap_bar(shap_values[0], self.feature_names, title=title_suffix + " (bar)")


    def analyze_error_sources(self, errors_df, top_n=15):
        errors_df["extension"] = errors_df["path"].str.extract(r"\.([a-zA-Z0-9]+)$")[0].fillna("no_ext")
        errors_df["top_dir"] = errors_df["path"].str.split("/").str[0].fillna("root")

        top_errors = errors_df.sort_values("abs_error", ascending=False).head(top_n)
        self.logging.info("Top errors:\n%s", top_errors[["path", "date", "actual", "pred", "residual"]])

        def get_error_stats(group_by_col):
            stats = (errors_df.groupby(group_by_col)["abs_error"].agg(['mean', 'std']).sort_values(
                by="mean", ascending=False).head(top_n))
            return stats

        stats_by_ext = get_error_stats("extension")
        stats_by_dir = get_error_stats("top_dir")
        stats_by_reason = get_error_stats("completion_reason")
        stats_by_committer = None

        if "committer_grouped" in errors_df.columns:
            stats_by_committer = get_error_stats("committer_grouped")

        stats_by_bins = get_error_stats("true_bin")

        self.model_plotter.plot_bar(stats_by_ext['mean'], title="MAE per file type", xlabel="File Extension", ylabel="MAE",
                                    yerr=stats_by_ext['std'])

        self.model_plotter.plot_bar(stats_by_dir['mean'], title="MAE per top level directory", xlabel="Directory",
                                    ylabel="MAE", yerr=stats_by_dir['std'], filename="mae_per_top_dir.png")

        self.model_plotter.plot_bar(stats_by_reason['mean'], title="MAE per reason", xlabel="Completion Reason",
                                    ylabel="MAE", yerr=stats_by_reason['std'], filename="mae_per_completion_reason.png")

        if stats_by_committer is not None:
            self.model_plotter.plot_bar(stats_by_committer['mean'], title="MAE per Committer", xlabel="Committer",
                                        ylabel="MAE", yerr=stats_by_committer['std'], filename="mae_per_committer.png")

        self.model_plotter.plot_bar(stats_by_bins['mean'], title="MAE per actual days", xlabel="Completion days bins",
                                    ylabel="MAE", yerr=stats_by_bins['mean'], filename="mae_per_completion_bins.png")

        self.model_plotter.plot_violin(errors_df, x="extension", y="abs_error",
                                       title="Error Distribution per File Type",
                                       xlabel="File Extension", ylabel="Absolute Error",
                                       filename="error_dist_by_ext.png")

        self.model_plotter.plot_violin(errors_df, x="top_dir", y="abs_error",
                                       title="Error Distribution per Top Level Directory",
                                       xlabel="Directory", ylabel="Absolute Error",
                                       filename="error_dist_by_dir.png")

        return stats_by_bins, stats_by_ext, stats_by_dir, stats_by_reason, stats_by_committer

    def analyze_pdp_ice(self, X, top_n_features=5, top_n_categorical_to_pair=3):
        if not hasattr(self.model.model, "feature_importances_"):
            self.logging.warning("Model does not support feature importances, skipping PDP/ICE plots.")
            return

        importances = self.model.model.feature_importances_

        all_sorted_indices = np.argsort(importances)[::-1]

        top_numerical_indices = []
        top_categorical_indices = []

        for idx in all_sorted_indices:
            feature_name = self.feature_names[idx]
            if feature_name in self.categorical_features:
                if len(top_categorical_indices) < top_n_categorical_to_pair:
                    top_categorical_indices.append(idx)
            else:
                if len(top_numerical_indices) < top_n_features:
                    top_numerical_indices.append(idx)

            # Stop searching once we have enough of both
            if len(top_numerical_indices) == top_n_features and len(
                    top_categorical_indices) == top_n_categorical_to_pair:
                break

        self.logging.info(f"Generating 1D PDP/ICE plots for top {len(top_numerical_indices)} numerical features...")

        # 1D Plot (for top numerical)
        for feature_idx in top_numerical_indices:
            feature_name = self.feature_names[feature_idx]
            fig, ax = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(
                self.model.model,
                X,
                features=[feature_idx],  # single feature for 1D
                feature_names=self.feature_names,
                kind="both",
                subsample=50,
                ice_lines_kw={"color": "blue", "alpha": 0.2, "linewidth": 0.5},
                pd_line_kw={"color": "red", "linestyle": "--", "linewidth": 2},
                ax=ax
            )
            filename = f"pdp_ice_1D_{feature_name}.png"
            self.model_plotter.save_plot(filename)

        # 2D Plot Logic (Pairing top numerical vs top categorical)
        top_categorical_names = [self.feature_names[i] for i in top_categorical_indices]
        if not top_categorical_names:
            self.logging.warning("No categorical features found or passed for 2D PDP plots.")
            return

        self.logging.info(f"Generating 2D PDP plots for interactions with top categorical: {top_categorical_names}")

        for num_idx in top_numerical_indices:
            num_name = self.feature_names[num_idx]

            for cat_name in top_categorical_names:
                self.logging.debug(f"Plotting 2D PDP for: {num_name} vs {cat_name}")
                fig, ax = plt.subplots(figsize=(10, 8))
                try:
                    PartialDependenceDisplay.from_estimator(
                        self.model.model,
                        X,
                        features=[(num_name, cat_name)],
                        feature_names=self.feature_names,
                        kind="average",
                        ax=ax
                    )
                    ax.set_title(f"2D PDP: {num_name} vs {cat_name}")
                    filename = f"pdp_2D_{num_name}_vs_{cat_name}.png"
                    self.model_plotter.save_plot(filename)
                except Exception as e:
                    self.logging.warning(f"Could not plot 2D PDP for {num_name} vs {cat_name}: {e}")
                    plt.close(fig)
