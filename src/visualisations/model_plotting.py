import logging

import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt

from src.visualisations.plotting import Plotter


class ModelPlotter(Plotter):
    def __init__(self, project_name, model=None, images_dir='images'):
        if model:
            super().__init__(project_name, f"{images_dir}/{model.__name__}")
        else:
            super().__init__(project_name, images_dir)

        self.logging = logging.getLogger(self.__class__.__name__)

    def plot_residuals(self, y_true, y_pred):
        residuals = y_true - y_pred

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(residuals, kde=True)
        plt.title("Distribution of Residuals")
        plt.xlabel("Errors (y_true - y_pred)")

        plt.subplot(1, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Predictions vs. Errors")
        plt.xlabel("Predicted Days Until Completion")
        plt.ylabel("Errors (y_true - y_pred)")
        plt.tight_layout()

        self.save_plot(f"residuals.png")

    def plot_errors_vs_actual(self, y_true, y_pred):
        errors = y_true - y_pred

        self._init_plot(title="Errors vs. Actual Days Until Completion", xlabel="Actual Days Until Completion",
                        ylabel="Errors (y_true - y_pred)")

        plt.scatter(y_true, errors, alpha=0.7)
        plt.axhline(0, color="red", linestyle="--")
        plt.tight_layout()

        self.save_plot("errors_vs_actual.png")

    def plot_completion_donut(self, completed, total):
        remaining = total - completed
        labels = ['Completed', 'Incomplete']
        sizes = [completed, remaining]

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, startangle=90, autopct='%1.1f%%', wedgeprops={'width': 0.3})
        ax.set_title('Completed Files')

        self.save_plot(f"completed_files_donut.png")

    def plot_predictions_vs_actual(self, y_true, y_pred):
        self._init_plot(title="Predicted vs. Actual Days Until Completion", xlabel="Actual Days",
                        ylabel="Predicted Days")

        plt.scatter(y_true, y_pred, alpha=0.7)
        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 color='red', linestyle='--', label='Perfect Prediction')

        plt.legend()
        plt.tight_layout()
        self.save_plot("predicted_vs_actual.png")

    def plot_top_errors(self, y_true, y_pred, n=10):
        errors = abs(y_true - y_pred)
        top_indices = errors.argsort()[-n:][::-1]

        df = pd.DataFrame({
            'Actual': y_true[top_indices],
            'Predicted': y_pred[top_indices],
            'Error': errors[top_indices]
        })

        df = df.sort_values('Error', ascending=True)

        self._init_plot(title=f"Top {n} Prediction Errors", xlabel="Days Until Completion")

        df[['Actual', 'Predicted']].plot(kind='barh')
        plt.tight_layout()

        self.save_plot("top_prediction_errors.png")

    def plot_feature_distribution(self, df, feature_name):
        self._init_plot(title=f"Distribution of {feature_name}", xlabel=feature_name)
        sns.histplot(df[feature_name].dropna(), kde=True)
        self.save_plot(f"{feature_name}_distribution.png")

    def plot_feature_correlations(self, features_df, target_series):
        """
        Computes the correlation between the features and the target feature using Pearson (between -1 and +1).
        -1 and +1 indicate the highest correlations.
        :param features_df:
        :param target_series:
        :return:
        """
        corr = features_df.corrwith(target_series).sort_values(key=abs, ascending=False).head(20)
        self._init_plot(title="Top Feature Correlations with Target", xlabel="Correlation Coefficient")
        sns.barplot(x=corr.values, y=corr.index, orient="h")
        self.save_plot("feature_correlations.png")

    def plot_model_feature_importance(self, feature_names, importances, top_n=20):
        sorted_idx = importances.argsort()[::-1][:top_n]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = importances[sorted_idx]

        self._init_plot(title="Top Feature Importances (Model)", xlabel="Importance")
        sns.barplot(x=sorted_values, y=sorted_names)
        self.save_plot("feature_importance.png")

    def plot_error_types_pie(self, error_types):
        self._init_plot(title="Distribution of Error Types", xlabel="Error Type", ylabel="Amount", figsize=(6,6))
        counts = error_types.value_counts()
        plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140)
        self.save_plot("error_types_pie.png")

    def plot_shap_summary(self, shap_values, X, feature_names, title: str = None,
                          filename: str = "top_errors_shap_summary.png"):
        shap.summary_plot(shap_values, features=X, feature_names=feature_names, show=False)
        plt.tight_layout()
        if title:
            plt.title(title)
        self.save_plot(filename)

    def plot_shap_bar(self, shap_values_row, feature_names, title: str = None):
        self._init_plot()
        shap.bar_plot(shap_values_row, feature_names=feature_names, show=False)
        plt.tight_layout()
        if title:
            plt.title(title)
        self.save_plot("top_shap_bar.png")