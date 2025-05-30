import logging

from shap import TreeExplainer


class ExplainabilityAnalyzer:
    def __init__(self, model, feature_names, model_plotter):
        self.model = model
        self.feature_names = feature_names
        self.model_plotter = model_plotter

        self.logging = logging.getLogger(self.__class__.__name__)

    def analyze_top_errors(self, errors_df, top_n=10):
        if not hasattr(self.model.model, "feature_importances_"):
            self.logging.warning("Skipping explainability: model is not tree-based.")
            return

        top_errors = errors_df.sort_values("abs_error", ascending=False).head(top_n)
        X_top = top_errors[self.feature_names].values

        explainer = TreeExplainer(self.model.model)
        shap_values = explainer.shap_values(X_top)

        self.model_plotter.plot_shap_summary(shap_values, X_top, self.feature_names)
        self.model_plotter.plot_shap_bar(shap_values[0], self.feature_names)

    def analyze_shap_by_committer(self, errors_df, top_n_committers=5):
        top_committers = errors_df["committer_grouped"].value_counts().head(top_n_committers).index
        explainer = TreeExplainer(self.model.model)

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
        print(top_errors[["path", "date", "actual", "pred", "residual"]])

        mae_by_ext = errors_df.groupby("extension")["abs_error"].mean().sort_values(ascending=False).head(top_n)
        mae_by_dir = errors_df.groupby("top_dir")["abs_error"].mean().sort_values(ascending=False).head(top_n)
        mae_by_reason = errors_df.groupby("completion_reason")["abs_error"].mean().sort_values(ascending=False)
        mae_by_committer = errors_df.groupby("committer_grouped")["abs_error"].mean().sort_values(ascending=False)
        mae_by_completion_days_bins = (errors_df.groupby("true_bin", observed=False)["abs_error"].mean()
                                       .sort_values(ascending=False))

        self.model_plotter.plot_bar(mae_by_ext, title="MAE per file type", xlabel="File Extension", ylabel="MAE")
        self.model_plotter.plot_bar(mae_by_dir, title="MAE per top level directory", xlabel="Directory",
                                    ylabel="MAE", filename="mae_per_top_dir.png")
        self.model_plotter.plot_bar(mae_by_reason, title="MAE per mixins reason", xlabel="Completion Reason",
                                    ylabel="MAE", filename="mae_per_completion_reason.png")
        self.model_plotter.plot_bar(mae_by_committer, title="MAE per Committer", xlabel="Committer", ylabel="MAE",
                                    filename="mae_per_committer.png")
        self.model_plotter.plot_bar(mae_by_completion_days_bins, title="MAE per actual mixins days",
                                    xlabel="Completion days bins", ylabel="MAE", filename="mae_per_completion_bins.png")
