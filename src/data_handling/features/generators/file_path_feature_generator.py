import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator


@feature_generator_registry.register
class FilePathFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self):
        super().__init__()
        self.feature_names_ = None

    def get_feature_names(self) -> list[str]:
        if self.feature_names_ is None:
            raise RuntimeError(
                "The 'generate' method must be called before 'get_feature_names' "
                "to determine the dynamic feature names."
            )
        return self.feature_names_

    def generate(self, df: pd.DataFrame, top_n_ext: int = 15, **kwargs) -> pd.DataFrame:
        initial_columns = set(df.columns)

        path_lower = df['path'].str.lower()

        # Basic path features
        df['path_depth'] = df['path'].str.count('/').fillna(0)
        df['file_extension'] = df['path'].str.extract(r'\.([a-zA-Z0-9]+)$')[0].fillna('no_ext')

        # Group rare extensions
        ext_counts = df['file_extension'].value_counts()
        top_exts = ext_counts.head(top_n_ext).index
        df['file_extension_grouped'] = df['file_extension'].apply(
            lambda x: x if x in top_exts else 'other'
        )

        dummies = pd.get_dummies(df['file_extension_grouped'], prefix='ext')
        df = pd.concat([df, dummies], axis=1)

        # Flag features based on path contents
        df['in_test_dir'] = path_lower.str.contains(r'[/_]tests?[/_]').astype(int)
        df['in_docs_dir'] = path_lower.str.contains(r'/(?:docs|documentation)/').astype(int)
        df['is_config_file'] = path_lower.str.contains(r'\.(json|yaml|yml|ini|toml|cfg|conf)$').astype(int)
        df['is_markdown'] = path_lower.str.endswith(('.md', '.markdown')).astype(int)
        df['is_github_workflow'] = path_lower.str.contains(r'\.github/workflows/').astype(int)
        df['is_readme'] = path_lower.str.contains(r'readme').astype(int)

        df.drop(columns=['file_extension', 'file_extension_grouped'], inplace=True, errors='ignore')

        # Determine the list of newly added feature columns
        final_columns = set(df.columns)
        new_feature_names = list(final_columns - initial_columns)

        self.feature_names_ = new_feature_names

        return df
