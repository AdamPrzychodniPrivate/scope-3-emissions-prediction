import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px  # noqa:  F401
import plotly.graph_objs as go
import seaborn as sn


# This function uses plotly.express
def compare_passenger_capacity_exp(preprocessed_shuttles: pd.DataFrame):
    return (
        preprocessed_shuttles.groupby(["shuttle_type"])
        .mean(numeric_only=True)
        .reset_index()
    )


# This function uses plotly.graph_objects
def compare_passenger_capacity_go(preprocessed_shuttles: pd.DataFrame):

    data_frame = (
        preprocessed_shuttles.groupby(["shuttle_type"])
        .mean(numeric_only=True)
        .reset_index()
    )
    fig = go.Figure(
        [
            go.Bar(
                x=data_frame["shuttle_type"],
                y=data_frame["passenger_capacity"],
            )
        ]
    )

    return fig


def create_confusion_matrix(companies: pd.DataFrame):
    actuals = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    predicted = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1]
    data = {"y_Actual": actuals, "y_Predicted": predicted}
    df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
    confusion_matrix = pd.crosstab(
        df["y_Actual"], df["y_Predicted"], rownames=["Actual"], colnames=["Predicted"]
    )
    sn.heatmap(confusion_matrix, annot=True)
    return plt


import pandas as pd
import plotly.express as px

def visualize_industries(companies: pd.DataFrame):
    """
    Creates a horizontal bar chart visualizing the number of companies in each industry.

    Args:
        companies: DataFrame containing a column 'Industry (Exiobase)' with industry names.

    Returns:
        A Plotly figure object representing the industry distribution.
    """
    if 'Industry (Exiobase)' not in companies:
        raise ValueError("DataFrame must contain an 'Industry (Exiobase)' column")

    # Count the number of occurrences for each industry
    industry_counts = companies['Industry (Exiobase)'].value_counts()

    # Create DataFrame for plotting
    df_industry = pd.DataFrame({'Industry': industry_counts.index, 'Counts': industry_counts.values})

    # Sort the DataFrame by 'Counts' in descending order
    df_industry = df_industry.sort_values('Counts', ascending=False)

    # Calculate plot height based on the number of industries
    height = len(industry_counts) * 15  # Modify the multiplier to get the desired bar width

    # Create the bar chart
    fig = px.bar(df_industry, x='Counts', y='Industry', title='Number of Companies from Each Industry (Exiobase)', orientation='h', height=height)

    # Reverse the order of categories on the y-axis
    fig.update_yaxes(autorange="reversed")

    return fig
