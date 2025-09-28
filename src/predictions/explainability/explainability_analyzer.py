import logging

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
            # For linear models — use small background dataset if possible
            return LinearExplainer(self.model.model, X_background or "auto")
        except Exception as e:
            self.logging.warning(f"Explainability skipped: {e}")
            return None

    def analyze_top_errors(self, errors_df, top_n=10):
        top_errors = errors_df.sort_values("abs_error", ascending=False).head(top_n)
        X_top = top_errors[self.feature_names].values

        explainer = self._get_shap_explainer(X_background=X_top)
        if explainer is None:
            return

        shap_values = explainer.shap_values(X_top)

        self.model_plotter.plot_shap_summary(shap_values, X_top, self.feature_names)
        self.model_plotter.plot_shap_bar(shap_values[0], self.feature_names)

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
        top_feature_names = [self.feature_names[i] for i in top_feature_indices]

        self.logging.info(f"Generating PDP/ICE plots for top features: {top_feature_names}")

        PartialDependenceDisplay.from_estimator(
            self.model.model,
            X,
            features=top_feature_indices,
            feature_names=self.feature_names,
            kind="both",  # Plot both PDP and ICE
            ice_lines_kw={"color": "blue", "alpha": 0.3, "linewidth": 0.5},
            pd_line_kw={"color": "red", "linestyle": "--", "linewidth": 2},
        )

        self.model_plotter.save_plot("pdp_ice_top_features.png")