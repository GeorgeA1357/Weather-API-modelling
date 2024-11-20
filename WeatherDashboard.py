import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import dash
from dash import dcc, html, Dash
import plotly.graph_objs as go
import time
from joblib import dump,load 
import os
from datetime import datetime,timedelta

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def historical_data(latitude, longitude, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(2).ValuesAsNumpy()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "temperature_2m_max": daily_temperature_2m_max,
        "temperature_2m_min": daily_temperature_2m_min,
        "temperature_2m_mean": daily_temperature_2m_mean
    }

    return pd.DataFrame(data=daily_data)

# Function to fetch forecast data
def forecast_data(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "forecast_days": 14
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "temperature_2m_max": daily_temperature_2m_max,
        "temperature_2m_min": daily_temperature_2m_min
    }

    return pd.DataFrame(data=daily_data)


locations = {
    "Boston": {"latitude": 42.3601, "longitude": -71.0589},
    "Hartford": {"latitude": 41.7637, "longitude": -72.6851},
    "Portland": {"latitude": 43.6591, "longitude": -70.2568},
    "New York City": {"latitude": 40.7128, "longitude": -74.0060},
    "Philadelphia": {"latitude": 39.9526, "longitude": -75.1652},
    "Newark": {"latitude": 40.7357, "longitude": -74.1724},
    "Pittsburgh": {"latitude": 40.4406, "longitude": -79.9959},
    "Baltimore": {"latitude": 39.2904, "longitude": -76.6122}
}

location_to_padd = {
    "Boston": "PADD 1A",
    "Hartford": "PADD 1A",
    "Portland": "PADD 1A",
    "New York City": "PADD 1B",
    "Philadelphia": "PADD 1B",
    "Newark": "PADD 1B",
    "Pittsburgh": "PADD 1B",
    "Baltimore": "PADD 1B"
}


# Fetch historical and forecast data for all locations
# historical_data_dict = {}
# forecast_data_dict = {}
# for location, coords in locations.items():
#     historical_data_dict[location] = historical_data(
#         latitude=coords["latitude"],
#         longitude=coords["longitude"],
#         start_date="2000-01-01",
#         end_date="2024-11-15"
#     )
#     forecast_data_dict[location] = forecast_data(
#         latitude=coords["latitude"],
#         longitude=coords["longitude"]
#     )


# fetching data in batches so as to not exceed API minutely limit
def fetch_data_in_batches(locations, batch_size=4, wait_time=60):
    historical_data_dict = {}
    forecast_data_dict = {}
    
    location_items = list(locations.items())
    
    for i in range(0, len(location_items), batch_size):
        batch = location_items[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}: {[loc[0] for loc in batch]}")

        for location, coords in batch:
            try:
                historical_data_dict[location] = historical_data(
                    latitude=coords["latitude"],
                    longitude=coords["longitude"],
                    start_date="2000-01-01",
                    #end_date="2024-11-15"
                    end_date=(datetime.now()-timedelta(days=1)).strftime("%Y-%m-%d")

                )
                forecast_data_dict[location] = forecast_data(
                    latitude=coords["latitude"],
                    longitude=coords["longitude"]
                )
                print(f"Successfully fetched data for {location}")
            except Exception as e:
                print(f"Error fetching data for {location}: {e}")

        if i + batch_size < len(location_items):  # Wait if there are more batches
            print(f"Waiting for {wait_time} seconds to respect API limits...")
            time.sleep(wait_time)

    return historical_data_dict, forecast_data_dict

#historical_data_dict, forecast_data_dict = fetch_data_in_batches(locations, batch_size=4, wait_time=60)

def user_prompt_data(historical_file='historical_data_dict.joblib',forecast_file='forecast_data_dict.joblib',user_prompt=False):
    if os.path.exists(historical_file) and os.path.exists(forecast_file):
        response = input("Saved data files found. Do you want to fetch new data? (yes/no): ").strip().lower()
        if response =="yes":
            user_prompt=True
        else:
            pass

    if user_prompt:
        print("Fetching new data from API...")
        historical_data_dict, forecast_data_dict = fetch_data_in_batches(locations, batch_size=4, wait_time=60)
        print("Saving data to files...")
        dump(historical_data_dict, historical_file)
        dump(forecast_data_dict, forecast_file)
    else:
        print("Loading data from saved files...")
        historical_data_dict = load(historical_file)
        forecast_data_dict = load(forecast_file)
        print("Loaded")
    
    return historical_data_dict, forecast_data_dict
    

historical_data_dict, forecast_data_dict=user_prompt_data()

processed_data = {}
for location, h_df in historical_data_dict.items():
    f_df = forecast_data_dict[location]

    h_df = h_df.rename(columns={"temperature_2m_mean": "mean", "temperature_2m_max": "max", "temperature_2m_min": "min"})
    h_df["date"] = pd.to_datetime(h_df["date"])
    h_df["year"] = h_df["date"].dt.year
    h_df["day_count"] = h_df["date"].dt.dayofyear

    f_df = f_df.rename(columns={"temperature_2m_max": "max", "temperature_2m_min": "min"})
    f_df["date"] = pd.to_datetime(f_df["date"])
    f_df["day_count"] = f_df["date"].dt.dayofyear

    # Filter historical data to overlap with forecast
    historical_overlap = h_df[h_df["day_count"].isin(f_df["day_count"])]
    mean_historical = historical_overlap.groupby("day_count").mean().reset_index()

    # Calculate the average of the last 3 years
    mean_last_3_years = h_df[h_df["day_count"].isin(f_df["day_count"])]
    mean_last_3_years=mean_last_3_years[mean_last_3_years["year"]>=2021]
    mean_last_3_years = mean_last_3_years.groupby("day_count").mean().reset_index()

    processed_data[location] = {
        "historical": h_df,
        "forecast": f_df,
        "mean_historical": mean_historical,
        "mean_last_3_years": mean_last_3_years  # Add this to processed data
    }




# Assuming processed_data and location_to_padd are already defined
padd1a_locations = {k: v for k, v in processed_data.items() if location_to_padd[k] == "PADD 1A"}
padd1b_locations = {k: v for k, v in processed_data.items() if location_to_padd[k] == "PADD 1B"}

# Dash app
app = Dash(__name__)

def generate_graphs(location, data):
    h_df = data["historical"]
    f_df = data["forecast"]
    h_df=h_df[h_df["day_count"].isin(f_df["day_count"])]
    mean_h_df = data["mean_historical"]
    mean_last_3_years = data["mean_last_3_years"]

    years = sorted(h_df["year"].unique())
    year_weights = {year: max(0.5, 3 - (f_df["date"].dt.year.max() - year) * 0.2) for year in years}
    year_alpha = {year: max(0.3, 1 - (f_df["date"].dt.year.max() - year) * 0.1) for year in years}

    graphs = [
        # Max Temperature Graph
        dcc.Graph(
            id=f"{location}-max-plot",
            figure={
                "data": [
                    go.Scatter(
                        x=h_df[h_df["year"] == year]["day_count"],
                        y=h_df[h_df["year"] == year]["max"],
                        mode="lines",
                        line={"width": year_weights[year], "color": "blue"},
                        opacity=year_alpha[year],
                        name=f"{year}"
                    ) for year in years
                ] + [
                    go.Scatter(
                        x=f_df["day_count"],
                        y=f_df["max"],
                        mode="lines",
                        line={"width": 5, "color": "orange"},
                        name="Forecast Max"
                    )
                ],
                "layout": go.Layout(
                    title=f"Max Temperatures",
                    xaxis={"title": "Day of Year"},
                    yaxis={"title": "Temperature (°C)"},
                    template="plotly_white"
                )
            },
            style={"height": "100%", "width": "100%"}

            
        ),
        # Min Temperature Graph
        dcc.Graph(
            id=f"{location}-min-plot",
            figure={
                "data": [
                    go.Scatter(
                        x=h_df[h_df["year"] == year]["day_count"],
                        y=h_df[h_df["year"] == year]["min"],
                        mode="lines",
                        line={"width": year_weights[year], "color": "green"},
                        opacity=year_alpha[year],
                        name=f"{year}"
                    ) for year in years
                ] + [
                    go.Scatter(
                        x=f_df["day_count"],
                        y=f_df["min"],
                        mode="lines",
                        line={"width": 5, "color": "red"},
                        name="Forecast Min"
                    )
                ],
                "layout": go.Layout(
                    title=f"Min Temperatures",
                    xaxis={"title": "Day of Year"},
                    yaxis={"title": "Temperature (°C)"},
                    template="plotly_white"
                )
            },
            style={"height": "100%", "width": "100%"}
        ),
        dcc.Graph(
            id=f"{location}-average-max-plot",
            figure={
                "data": [
                    go.Scatter(
                        x=mean_h_df["day_count"],
                        y=mean_h_df["max"],
                        mode="lines",
                        line={"width": 3, "dash": "dash", "color": "blue"},
                        name="Historical Avg Max"
                    ),
                    go.Scatter(
                        x=mean_last_3_years["day_count"],
                        y=mean_last_3_years["max"],
                        mode="lines",
                        line={"width": 3, "dash": "dot", "color": "purple"},
                        name="Last 3 Years Avg Max"
                    ),
                    go.Scatter(
                        x=f_df["day_count"],
                        y=f_df["max"],
                        mode="lines",
                        line={"width": 3, "dash": "dash", "color": "orange"},
                        name="Forecast Max"
                    )
                ],
                "layout": go.Layout(
                    title=f"Average Max Temperatures",
                    xaxis={"title": "Day of Year"},
                    yaxis={"title": "Temperature (°C)"},
                    template="plotly_white"
                )
            },
            style={"height": "100%", "width": "100%"}
        ),
        # Average Min Temperature Graph
        dcc.Graph(
            id=f"{location}-average-min-plot",
            figure={
                "data": [
                    go.Scatter(
                        x=mean_h_df["day_count"],
                        y=mean_h_df["min"],
                        mode="lines",
                        line={"width": 3, "dash": "dash", "color": "green"},
                        name="Historical Avg Min"
                    ),
                    go.Scatter(
                        x=mean_last_3_years["day_count"],
                        y=mean_last_3_years["min"],
                        mode="lines",
                        line={"width": 3, "dash": "dot", "color": "purple"},
                        name="Last 3 Years Avg Min"
                    ),
                    go.Scatter(
                        x=f_df["day_count"],
                        y=f_df["min"],
                        mode="lines",
                        line={"width": 3, "dash": "dash", "color": "red"},
                        name="Forecast Min"
                    )
                ],
                "layout": go.Layout(
                    title=f"Average Min Temperatures",
                    xaxis={"title": "Day of Year"},
                    yaxis={"title": "Temperature (°C)"},
                    template="plotly_white"
                )
            },
            style={"height": "100%", "width": "100%"}
        )
    ]
    return graphs

def create_tab(location_data, title):
    return dcc.Tab(
        label=title,
        children=[
            html.Div([
                html.Div([
                    html.H2(location, style={"text-align": "center"}),
                    *generate_graphs(location, data)
                ], 
                style={
                    "display": "flex",
                    "flex-direction": "column",
                    "align-items": "center",
                    "padding": "5px",
                    "margin": "5px",
                    "width": "32%",  # Adjusted width for three columns
                    "box-sizing": "border-box"
                })
                for location, data in location_data.items()
            ], 
            style={
                "display": "flex",
                "flex-wrap": "wrap",  # Allows wrapping to next row
                "justify-content": "space-between",  # Distribute space evenly
                "padding": "10px",
                "overflow-y": "auto",  # Enable vertical scrolling if needed
                "box-sizing": "border-box"
            })
        ]
    )

app.layout = html.Div([
    html.H1("Weather Dashboard", style={'text-align': 'center'}),
    dcc.Tabs([
        create_tab(padd1a_locations, "PADD 1A"),
        create_tab(padd1b_locations, "PADD 1B")
    ])
])

if __name__ == "__main__":
    app.run_server(host="192.168.2.112", port=8050, debug=False)
