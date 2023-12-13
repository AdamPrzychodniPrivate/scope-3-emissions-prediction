from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model_input_table, preprocess_companies, preprocess_shuttles, preprocess_data, feature_engineering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs=["preprocessed_companies", "companies_columns"],
                name="preprocess_companies_node",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="preprocessed_shuttles",
                name="preprocess_shuttles_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_shuttles", "preprocessed_companies", "reviews"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
            node(
                func=preprocess_data,
                inputs=["data", "params:feature_options"],
                outputs="preprocessed_scope3",
                name="preprocess_scope3_node",
            ),
            node(
                func=feature_engineering,
                inputs=["preprocessed_scope3"],
                outputs="model_input_table_scope3",
                name="feature_engineering_scope3_node",
            ),
        ]
    )
