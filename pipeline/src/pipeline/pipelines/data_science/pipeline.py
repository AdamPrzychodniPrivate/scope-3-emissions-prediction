from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model, split_data_scope3, train_model_scope3, evaluate_model_scope3


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                name="evaluate_model_node",
                outputs="metrics",
            ),
            node(
                func=split_data_scope3,
                inputs=["model_input_table_scope3", "params:model_options_scope3"],
                outputs=["X_train_scope3", "X_test_scope3", "y_train_scope3", "y_test_scope3"],
                name="split_data_node_scope3",
            ),
            node(
                func=train_model_scope3,
                inputs=["X_train_scope3", "y_train_scope3"],
                outputs="regressor_scope3",
                name="train_model_node_scope3",
            ),
            node(
                func=evaluate_model_scope3,
                inputs=["regressor_scope3", "X_test_scope3", "y_test_scope3"],
                outputs=None,
                name="evaluate_model_node_scope3",
            ),
        ]
    )
