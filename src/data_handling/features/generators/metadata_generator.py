import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator

@feature_generator_registry.register
class MetadataFeatureGenerator(AbstractFeatureGenerator):
    def get_feature_names(self) -> list[str]:
        return [
            'path_depth', 'in_test_dir', 'in_docs_dir', 'weekday', 'month', 'is_conifg_file', 'is_markdown',
            'is_desktop_entry', 'is_workflow_file', 'has_readme_name', 'is_source_code', 'is_script',
            'ext_'
        ]

    def generate(self, df: pd.DataFrame, use_categorical: bool = False, **kwargs) -> pd.DataFrame:
        df["file_extension"] = df["path"].str.extract(r'\.([a-zA-Z0-9]+)$')[0].fillna("unknown")
        ext_counts = df["file_extension"].value_counts()
        top_exts = ext_counts[(ext_counts >= 10) | (ext_counts.rank(method="min") <= 10)].index
        df["file_extension"] = df["file_extension"].apply(lambda x: x if x in top_exts else "other")

        if not use_categorical:
            dummies = pd.get_dummies(df["file_extension"], prefix="ext")
            df = pd.concat([df, dummies], axis=1)

        df["path_depth"] = df["path"].str.count("/")
        df["in_test_dir"] = df["path"].str.lower().str.contains(r"/tests?/").astype(int)
        df["in_docs_dir"] = df["path"].str.lower().str.contains(r"/(?:docs|documentation)/").astype(int)
        df["weekday"] = df["date"].dt.weekday  # Monday = 0
        df["month"] = df["date"].dt.month  # January = 1

        path_lower = df["path"].str.lower()
        config_extensions = {"json", "yaml", "yml", "ini", "toml", "env", "cfg", "conf"}
        source_code_exts = (".py", ".js", ".ts", ".rb", ".java", ".cpp", ".c", ".cs", ".go", ".rs", ".php", ".swift")

        df["is_config_file"] = df["file_extension"].isin(config_extensions)
        df["is_markdown"] = path_lower.str.endswith((".md", ".markdown")).astype(int)
        df["is_desktop_entry"] = path_lower.str.endswith(".desktop").astype(int)
        df["is_workflow_file"] = path_lower.str.contains(r"\.github/workflows/").astype(int)
        df["has_readme_name"] = path_lower.str.contains(r"readme").astype(int)
        df["is_source_code"] = path_lower.str.endswith(source_code_exts).astype(int)
        df["is_script"] = path_lower.str.endswith((".sh", ".bat")).astype(int)

        return df
