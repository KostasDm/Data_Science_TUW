import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

df = pd.read_excel("national_state_sector_2002_2024_caps_21feb2025_tons.xlsx", sheet_name='State')

emission_cols = [c for c in df.columns if c.startswith("emissions")]


df_long = df.melt(id_vars=["State", "Pollutant", "Sector"], value_vars=emission_cols, var_name="Year", value_name="Emissions")

df_long["Year"] = df_long["Year"].str.replace("emissions", "").astype(int)


### Initialize Dash

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H3("State emission levels by Pollutant"),

        html.Div(
            [
                dcc.Dropdown(
                    id="state-dropdown",
                    options=[
                        {"label": s, "value": s}
                        for s in sorted(df_long["State"].unique())
                    ],
                    value=df_long["State"].iloc[0],
                ),

                dcc.Dropdown(
                    id="pollutant-dropdown",
                    options=[
                        {"label": p, "value": p}
                        for p in sorted(df_long["Pollutant"].unique())
                    ],
                    value=df_long["Pollutant"].iloc[0],
                ),
            ],
            style={"width": "100%", "gap": "50px"},
        ),

        dcc.Graph(id="emissions-lineplot"),
    ]
)

# --- Callback ---
@app.callback(
    Output("emissions-lineplot", "figure"),
    Input("state-dropdown", "value"),
    Input("pollutant-dropdown", "value"),
)
def update_plot(state, pollutant):
    df_to_plot = df_long[ (df_long["State"] == state) & (df_long["Pollutant"] == pollutant)]

    fig = px.line(
        df_to_plot,
        x="Year",
        y="Emissions",
        color="Sector",  # optional, remove if not needed
        markers=True,
        title=f"{pollutant} Emissions in {state}",
    )

    fig.update_yaxes(type="log", title="Emissions (log scale)")
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Emissions",
        template="plotly_white",
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True)