import pandas as pd

from src.data_handling.features.feature_generator_registry import feature_generator_registry
from src.data_handling.features.generators.abstract_feature_generator import AbstractFeatureGenerator


@feature_generator_registry.register
class FilePathFeatureGenerator(AbstractFeatureGenerator):
    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        # Static binary flags that are always created
        feature_names = [
            'path_depth', 'in_test_dir', 'in_docs_dir', 'is_config_file',
            'is_markdown', 'is_github_workflow', 'is_readme'
        ]

        if 'path' in df.columns:
            file_extension = df['path'].str.extract(r'\.([a-zA-Z0-9]+)$')[0].fillna('no_ext')
            ext_counts = file_extension.value_counts()
            top_exts = ext_counts.head(15).index

            extension_features = [f"ext_{ext}" for ext in top_exts]
            extension_features.append("ext_other")
            feature_names.extend(extension_features)

        return feature_names

    def generate(self, df: pd.DataFrame, top_n_ext: int = 15, **kwargs) -> tuple[pd.DataFrame, list[str]]:
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

        binaries = (["in_test_dir", "in_docs_dir", "is_config_file", "is_markdown", "is_github_workflow", "is_readme"] +
                    dummies.columns.tolist())

        df.drop(columns=['file_extension', 'file_extension_grouped'], inplace=True, errors='ignore')

        return df, binaries
