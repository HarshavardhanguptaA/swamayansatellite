import plotly.express as px

def correlation_heatmap(corr_df, title="Cross-Satellite Feature Correlation"):
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        origin="lower",
        title=title
    )

    fig.update_layout(
        height=600,
        coloraxis_colorbar=dict(title="Correlation")
    )

    return fig
import plotly.graph_objects as go

def lag_correlation_plot(lag_dict, title="Lag Correlation"):

    lags = list(lag_dict.keys())
    values = list(lag_dict.values())

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=lags,
            y=values,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Lag (timesteps)",
        yaxis_title="Correlation coefficient",
        height=450
    )

    return fig
