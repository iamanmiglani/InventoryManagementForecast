# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import altair as alt

# -----------------------------------
# Data Loading and Preparation
# -----------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("units_sold_data.csv", parse_dates=['Date'])
    # Clean column names: remove extra spaces
    data.columns = data.columns.str.strip()
    return data

def split_data(data):
    # Training: all dates up to November 30, 2023
    train = data[data['Date'] <= pd.to_datetime("2023-11-30")].copy()
    # Test: December 1 - December 10, 2023 (all available; we’ll use a subset per slider)
    test = data[(data['Date'] >= pd.to_datetime("2023-12-01")) &
                (data['Date'] <= pd.to_datetime("2023-12-10"))].copy()
    return train, test

# -----------------------------------
# Forecasting Functions
# -----------------------------------
def forecast_prophet_univariate(train, forecast_days):
    df_train = train[['Date', 'Units_sold']].rename(columns={'Date': 'ds', 'Units_sold': 'y'})
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    forecast_test = forecast[['ds', 'yhat']].tail(forecast_days)
    return forecast_test

def forecast_prophet_multivariate(train, forecast_days, regressors):
    df_train = train.copy().rename(columns={'Date': 'ds', 'Units_sold': 'y'})
    model = Prophet()
    for reg in regressors:
        model.add_regressor(reg)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=forecast_days)
    for reg in regressors:
        future[reg] = df_train[reg].iloc[-1]
    forecast = model.predict(future)
    forecast_test = forecast[['ds', 'yhat']].tail(forecast_days)
    return forecast_test

def alternative_forecast_model(train, test_subset):
    last_value = train['Units_sold'].iloc[-1]
    forecast_values = [last_value] * len(test_subset)
    forecast_test = pd.DataFrame({"yhat": forecast_values}, index=test_subset['Date'])
    forecast_test = forecast_test.reset_index().rename(columns={'index': 'ds'})
    return forecast_test

def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

# -----------------------------------
# Sidebar: Forecast Horizon Selection
# -----------------------------------
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Select forecast days (December):", min_value=1, max_value=10, value=10)

# -----------------------------------
# Main App: Load Data and Split
# -----------------------------------
st.title("Agentic Forecasting Pipeline with RL-Stopping Rule")
data = load_data()
st.write("### Data Overview", data.head())

train, test = split_data(data)
# Use only the first 'forecast_days' rows from test data for evaluation
test_subset = test.head(forecast_days)

# -----------------------------------
# Forecast Computation and Stopping Logic
# -----------------------------------
st.write("## Forecast Computation and Decision")

# Always compute the univariate forecast first
forecast_uni = forecast_prophet_univariate(train, forecast_days)
mape_uni = calculate_mape(test_subset['Units_sold'].values, forecast_uni['yhat'].values)

# Initialize variables for multivariate and alternative forecasts
forecast_multi = None
mape_multi = None
forecast_alt = None
mape_alt = None

if mape_uni <= 15:
    final_method = "univariate"
    final_forecast = forecast_uni
    decision_note = "Univariate forecast accepted (MAPE ≤ 15%). No further models computed."
else:
    if all(col in train.columns for col in ['Promotion', 'Price']):
        forecast_multi = forecast_prophet_multivariate(train, forecast_days, ['Promotion', 'Price'])
        mape_multi = calculate_mape(test_subset['Units_sold'].values, forecast_multi['yhat'].values)
        if mape_multi <= 15:
            final_method = "multivariate"
            final_forecast = forecast_multi
            decision_note = "Multivariate forecast accepted (MAPE ≤ 15%). Alternative model not computed."
        else:
            forecast_alt = alternative_forecast_model(train, test_subset)
            mape_alt = calculate_mape(test_subset['Units_sold'].values, forecast_alt['yhat'].values)
            error_dict = {"univariate": mape_uni, "multivariate": mape_multi, "alternative": mape_alt}
            final_method = min(error_dict, key=error_dict.get)
            final_forecast = forecast_uni if final_method == "univariate" else (forecast_multi if final_method == "multivariate" else forecast_alt)
            decision_note = ("None of the forecasts met the acceptable threshold. " +
                             f"Method with the lowest MAPE selected: {final_method}.")
    else:
        forecast_alt = alternative_forecast_model(train, test_subset)
        mape_alt = calculate_mape(test_subset['Units_sold'].values, forecast_alt['yhat'].values)
        error_dict = {"univariate": mape_uni, "alternative": mape_alt}
        final_method = min(error_dict, key=error_dict.get)
        final_forecast = forecast_uni if final_method == "univariate" else forecast_alt
        decision_note = ("Multivariate forecast not available. " +
                         f"Final forecast selected: {final_method}.")

# -----------------------------------
# Display Forecast Metrics
# -----------------------------------
st.write("### Forecast Comparison Metrics")
metrics = {"Forecast Method": [], "MAPE (%)": []}
metrics["Forecast Method"].append("Univariate")
metrics["MAPE (%)"].append(f"{mape_uni:.2f}")
if forecast_multi is not None:
    metrics["Forecast Method"].append("Multivariate")
    metrics["MAPE (%)"].append(f"{mape_multi:.2f}")
if forecast_alt is not None:
    metrics["Forecast Method"].append("Alternative")
    metrics["MAPE (%)"].append(f"{mape_alt:.2f}")
final_mape = calculate_mape(test_subset['Units_sold'].values, final_forecast['yhat'].values)
metrics["Forecast Method"].append("Final Decision")
metrics["MAPE (%)"].append(f"{final_method} ({final_mape:.2f})")
metrics_df = pd.DataFrame(metrics)
st.table(metrics_df)

st.write("**Decision Note:**", decision_note)

# -----------------------------------
# Visualization: Historical Data (last 60 days) and Final Forecast
# -----------------------------------
st.write("### Historical Data and Final Forecast")

# Select the last 60 days of historical data (if available)
historical_data = train[['Date', 'Units_sold']].copy().sort_values("Date").tail(60)
historical_data["Type"] = "Historical"

# Prepare final forecast data and rename columns for consistency
forecast_vis = final_forecast.copy().rename(columns={'ds': 'Date', 'yhat': 'Units_sold'})
forecast_vis["Type"] = "Forecast"

# Combine historical data and forecast so that the time axis is continuous.
plot_df = pd.concat([
    historical_data[['Date', 'Units_sold', 'Type']],
    forecast_vis[['Date', 'Units_sold', 'Type']]
]).sort_values("Date")

# Create an interactive Altair chart (users can zoom/pan)
chart = alt.Chart(plot_df).mark_line().encode(
    x=alt.X('Date:T', axis=alt.Axis(title="Date")),
    y=alt.Y('Units_sold:Q', axis=alt.Axis(title="Units Sold")),
    color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Forecast'],
                                                range=['blue', 'green']))
).interactive().properties(
    width=700,
    height=400,
    title="Last 60 Days of Historical Data (Blue) and Final Forecast (Green)"
)
st.altair_chart(chart, use_container_width=True)

# -----------------------------------
# Final Forecast Output (Detailed)
# -----------------------------------
st.write("## Final Forecast Output")
st.dataframe(final_forecast)
