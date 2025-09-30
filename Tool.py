import plotly.express as px

def visualize_profiles(df, param="temperature_mean"):
    fig = px.scatter(df, x=param, y="pressure_mean", color="float_id",
                     hover_data=["date", "lat", "lon"])
    fig.update_yaxes(autorange="reversed")
    return fig

def compare_profiles(df):
    fig = px.line(df, x="pressure_mean", y="temperature_mean",
                  color="float_id", hover_data=["date"])
    fig.update_xaxes(title="Depth (dbar)")
    fig.update_yaxes(title="Temperature (°C)")
    return fig