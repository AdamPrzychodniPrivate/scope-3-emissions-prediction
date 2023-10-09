from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_scope3, feature_engineering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_scope3,
                inputs=["scope3_data_3", "params:feature_options"],
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
