import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
import plotly.graph_objs as go



# Setup Open-Meteo API client
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

    print(f"\nHistorical Data for Coordinates {latitude}°N, {longitude}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone: {response.Timezone()} {response.TimezoneAbbreviation()}")

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

    daily_dataframe = pd.DataFrame(data=daily_data)
    return daily_dataframe


def forecast_data(latitude, longitude):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "forecast_days": 14
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    print(f"Coordinates: {response.Latitude()}°N, {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone: {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()} seconds")

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


#f_df = forecast_data(52.5244, 13.4105)
#f_df.to_csv(r"C:\Users\george.argyrou\Downloads\BerlinWeatherDataForecast.csv")



# Example Usage
#df = historical_data(latitude=52.5244, longitude=13.4105, start_date="2000-01-01", end_date="2024-11-15")
# print(df)
#df.to_csv(r"C:\Users\george.argyrou\Downloads\BerlinWeatherData.csv")

f_df_raw=pd.read_csv(r"C:\Users\george.argyrou\Downloads\BerlinWeatherDataForecast.csv")
df_raw=pd.read_csv(r"C:\Users\george.argyrou\Downloads\BerlinWeatherData.csv")

df=df_raw.copy()
df=df.drop(["Unnamed: 0"],axis=1).rename(columns={"temperature_2m_mean":"mean",
                                                  "temperature_2m_max":"max",
                                                  "temperature_2m_min":"min"})

df["date"]=pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["day_count"] = df["date"].dt.dayofyear

f_df=f_df_raw.copy()
f_df=f_df.drop(["Unnamed: 0"],axis=1).rename(columns={"temperature_2m_max":"max",
                                                  "temperature_2m_min":"min"})

f_df["date"]=pd.to_datetime(f_df["date"])
f_df["day_count"] = f_df["date"].dt.dayofyear

h_df=df[df["day_count"].isin(f_df["day_count"])]
mean_h_df=h_df.groupby("day_count").mean().reset_index()



current_year = f_df["date"].dt.year.max()
years = sorted(h_df["year"].unique())
year_weights = {year: max(0.5, 3 - (current_year - year) * 0.2) for year in years}
year_alpha = {year: max(0.3, 1 - (current_year - year) * 0.1) for year in years}  # Alpha (0.3 to 1)

plt.figure(figsize=(14, 10))

for year in years:
    subset = h_df[h_df["year"] == year]
    plt.plot(
        subset["day_count"],
        subset["max"],
        label=f"Historical Max {year}",
        linewidth=year_weights[year],
        alpha=year_alpha[year],
        color="blue",
    )
plt.plot(f_df["day_count"], f_df["max"], label="Forecast Max", color="orange", linestyle="--", linewidth=3, alpha=1)
plt.title("Max Temperatures: Historical vs Forecast", fontsize=20)
plt.xlabel("Day of Year", fontsize=14)
plt.ylabel("Max Temperature (°C)", fontsize=14)
# plt.legend(fontsize=12, loc="upper left", frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 10))

for year in years:
    subset = h_df[h_df["year"] == year]
    plt.plot(
        subset["day_count"],
        subset["min"],
        label=f"Historical Min {year}",
        linewidth=year_weights[year],
        alpha=year_alpha[year],
        color="green",
    )
plt.plot(f_df["day_count"], f_df["min"], label="Forecast Min", color="red", linestyle="--", linewidth=3, alpha=1)
plt.title("Min Temperatures: Historical vs Forecast", fontsize=20)
plt.xlabel("Day of Year", fontsize=14)
plt.ylabel("Min Temperature (°C)", fontsize=14)
# plt.legend(fontsize=12, loc="upper left", frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()



plt.figure(figsize=(14, 10))
plt.plot(mean_h_df["day_count"], mean_h_df["max"], label="Historical Avg Max", color="orange", linestyle="--", linewidth=3, alpha=1)
plt.title("Max Temperatures: Historical vs Forecast", fontsize=20)
plt.xlabel("Day of Year", fontsize=14)
plt.ylabel("Max Temperature (°C)", fontsize=14)
plt.plot(f_df["day_count"], f_df["max"], label="Forecast Max", color="Blue", linestyle="--", linewidth=3, alpha=1)
plt.legend(fontsize=12, loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
plt.plot(mean_h_df["day_count"], mean_h_df["min"], label="Historical Avg Min", color="orange", linestyle="--", linewidth=3, alpha=1)
plt.title("Max Temperatures: Historical vs Forecast", fontsize=20)
plt.xlabel("Day of Year", fontsize=14)
plt.ylabel("Max Temperature (°C)", fontsize=14)
plt.plot(f_df["day_count"], f_df["min"], label="Forecast Min", color="Blue", linestyle="--", linewidth=3, alpha=1)
plt.legend(fontsize=12, loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()










# unique_years = df["year"].unique()
# num_years = len(unique_years)

# fig, axes = plt.subplots(nrows=num_years, ncols=1, figsize=(10, 5 * num_years), sharex=True)

# for i, year in enumerate(unique_years):
#     subset = df[df["year"] == year]
#     ax = axes[i] if num_years > 1 else axes  # Handle single subplot case
#     ax.plot(subset["day_count"], subset["mean"], label=f"Year {year}", color="C0", linewidth=2)
#     ax.set_title(f"Mean Temperature for Year {year}", fontsize=14)
#     ax.set_ylabel("Mean Temperature", fontsize=12)
#     ax.grid(True, linestyle="--", alpha=0.5)
#     ax.legend(loc="upper right", fontsize=10)

# axes[-1].set_xlabel("Day Count", fontsize=12)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(12, 8))

# for year in unique_years:
#     subset = df[df["year"] == year]
#     plt.plot(subset["day_count"], subset["mean"], label=f"Year {year}", linewidth=2, alpha=0.8)

# plt.title("Mean Temperature by Day Count (All Years)", fontsize=18)
# plt.xlabel("Day Count", fontsize=14)
# plt.ylabel("Mean Temperature", fontsize=14)
# plt.legend(
#     title="Year", 
#     fontsize=12, 
#     title_fontsize=14, 
#     loc="upper left", 
#     bbox_to_anchor=(1, 1)
# )
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.show()


# Initialize Dash app
app = dash.Dash(__name__)

# Dashboard layout
app.layout = html.Div([
    html.H1("Weather Dashboard", style={'text-align': 'center'}),
    
    html.Div([
        dcc.Graph(
            id="max-temperature-plot",
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
                        line={"width": 5, "color": "#FF7F0E"},
                        name="Forecast Max"
                    )
                ],
                "layout": go.Layout(
                    title="Max Temperatures: Historical vs Forecast",
                    xaxis={"title": "Day Count"},
                    yaxis={"title": "Max Temperature (°C)"},
                    width=1200,  
                    height=600,  
                    legend={"x": 1, "y": 1},
                    template="plotly_white"
                )
            }
        )
    ]),

    html.Div([
        dcc.Graph(
            id="min-temperature-plot",
            figure={
                "data": [
                    go.Scatter(
                        x=h_df[h_df["year"] == year]["day_count"],
                        y=h_df[h_df["year"] == year]["min"],
                        mode="lines",
                        line={"width": year_weights[year], "color": "green"},
                        opacity=year_alpha[year],
                        name=f"={year}"
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
                    title="Min Temperatures: Historical vs Forecast",
                    xaxis={"title": "Day Count"},
                    yaxis={"title": "Min Temperature (°C)"},
                    width=1200,  
                    height=600,  
                    legend={"x": 1, "y": 1},
                    template="plotly_white"
                )
            }
        )
    ]),

    html.Div([
        dcc.Graph(
            id="average-max-plot",
            figure={
                "data": [
                    go.Scatter(
                        x=mean_h_df["day_count"],
                        y=mean_h_df["max"],
                        mode="lines",
                        line={"width": 3, "dash": "dash", "color": "orange"},
                        name="Historical Avg Max"
                    ),
                    go.Scatter(
                        x=f_df["day_count"],
                        y=f_df["max"],
                        mode="lines",
                        line={"width": 3, "dash": "dash", "color": "blue"},
                        name="Forecast Max"
                    )
                ],
                "layout": go.Layout(
                    title="Average Max Temperatures: Historical vs Forecast",
                    xaxis={"title": "Day Count"},
                    yaxis={"title": "Max Temperature (°C)"},
                    width=1200,  
                    height=600,  
                    legend={"x": 1.3, "y": 1},
                    template="plotly_white"
                )
            }
        )
    ]),

    html.Div([
        dcc.Graph(
            id="average-min-plot",
            figure={
                "data": [
                    go.Scatter(
                        x=mean_h_df["day_count"],
                        y=mean_h_df["min"],
                        mode="lines",
                        line={"width": 3, "dash": "dash", "color": "orange"},
                        name="Historical Avg Min"
                    ),
                    go.Scatter(
                        x=f_df["day_count"],
                        y=f_df["min"],
                        mode="lines",
                        line={"width": 3, "dash": "dash", "color": "blue"},
                        name="Forecast Min"
                    )
                ],
                "layout": go.Layout(
                    title="Average Min Temperatures: Historical vs Forecast",
                    xaxis={"title": "Day Count"},
                    yaxis={"title": "Min Temperature (°C)"},
                    width=1200,  
                    height=600,  
                    legend={"x": 1, "y": 1},
                    template="plotly_white"
                )
            }
        )
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)