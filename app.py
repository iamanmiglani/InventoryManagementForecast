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
    # Remove any leading/trailing spaces from column names
    data.columns = data.columns.str.strip()
    return data

def split_data(data):
    # Training data: all dates up to November 30, 2023
    train = data[data['Date'] <= pd.to_datetime("2023-11-30")].copy()
    # Test data: December 1 to December 10, 2023
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
    # Return only the forecast for the last forecast_days
    forecast_test = forecast[['ds', 'yhat']].tail(forecast_days)
    return forecast_test

def forecast_prophet_multivariate(train, forecast_days, regressors):
    df_train = train.copy().rename(columns={'Date': 'ds', 'Units_sold': 'y'})
    model = Prophet()
    for reg in regressors:
        model.add_regressor(reg)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=forecast_days)
    # For future dates, use the last observed value for each regressor.
    for reg in regressors:
        future[reg] = df_train[reg].iloc[-1]
    forecast = model.predict(future)
    forecast_test = forecast[['ds', 'yhat']].tail(forecast_days)
    return forecast_test

def alternative_forecast_model(train, test_subset):
    # A simple alternative: use the last observed value from training data.
    last_value = train['Units_sold'].iloc[-1]
    forecast_values = [last_value] * len(test_subset)
    # Create a DataFrame with dates from test_subset
    forecast_test = pd.DataFrame({"yhat": forecast_values}, index=test_subset['Date'])
    forecast_test = forecast_test.reset_index().rename(columns={'index': 'ds'})
    return forecast_test

def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

# -----------------------------------
# RL Q-Table Helper (for display only)
# -----------------------------------
def choose_forecast_method(q_table, state="high"):
    """
    Given a Q-table and a state, return the forecast method with the highest Q-value.
    (This function is now only used to display Q-values.)
    """
    actions = ["univariate", "multivariate", "alternative"]
    q_values = {action: q_table.get((state, action), 0) for action in actions}
    best_action = max(q_values, key=q_values.get)
    return best_action, q_values

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
# Compute All Forecasts and their MAPE
# -----------------------------------
st.write("## Forecast Results")

# Univariate Forecast
forecast_uni = forecast_prophet_univariate(train, forecast_days)
mape_uni = calculate_mape(test_subset['Units_sold'].values, forecast_uni['yhat'].values)

# Multivariate Forecast (if additional regressors exist)
if all(col in train.columns for col in ['Promotion', 'Price']):
    forecast_multi = forecast_prophet_multivariate(train, forecast_days, ['Promotion', 'Price'])
    mape_multi = calculate_mape(test_subset['Units_sold'].values, forecast_multi['yhat'].values)
else:
    forecast_multi = None
    mape_multi = None

# Alternative Forecast
forecast_alt = alternative_forecast_model(train, test_subset)
mape_alt = calculate_mape(test_subset['Units_sold'].values, forecast_alt['yhat'].values)

# -----------------------------------
# Final Forecast Decision with Stopping Rule
# -----------------------------------
# If univariate forecast error is acceptable (MAPE <= 15%), choose it immediately.
if mape_uni <= 15:
    final_method = "univariate"
    final_forecast = forecast_uni
    decision_note = "Univariate forecast accepted (MAPE ≤ 15%). RL layer stops here."
else:
    # Otherwise, if multivariate is available and acceptable, choose it.
    if forecast_multi is not None and mape_multi <= 15:
        final_method = "multivariate"
        final_forecast = forecast_multi
        decision_note = "Multivariate forecast accepted (MAPE ≤ 15%)."
    # Else if alternative forecast is acceptable, choose it.
    elif mape_alt <= 15:
        final_method = "alternative"
        final_forecast = forecast_alt
        decision_note = "Alternative forecast accepted (MAPE ≤ 15%)."
    else:
        # None meet the threshold; choose the method with the lowest MAPE.
        error_dict = {"univariate": mape_uni,
                      "multivariate": mape_multi if mape_multi is not None else float('inf'),
                      "alternative": mape_alt}
        final_method = min(error_dict, key=error_dict.get)
        if final_method == "univariate":
            final_forecast = forecast_uni
        elif final_method == "multivariate":
            final_forecast = forecast_multi
        else:
            final_forecast = forecast_alt
        decision_note = ("None of the forecasts met the acceptable MAPE threshold. "
                         f"Method with the lowest MAPE selected: {final_method}.")

# Optionally, load and display RL Q-table info (for transparency)
try:
    with open("trained_rl_agent.pkl", "rb") as f:
        trained_q_table = pickle.load(f)
    # For display purposes, we show Q-values for state "high"
    rl_choice, q_values = choose_forecast_method(trained_q_table, "high")
except Exception as e:
    q_values = {}
    rl_choice = "N/A"
    st.warning("RL Q-table could not be loaded. " + str(e))

# -----------------------------------
# Display Forecast Comparison Metrics
# -----------------------------------
st.write("### Forecast Comparison Metrics")
metrics = {
    "Forecast Method": ["Univariate", "Multivariate", "Alternative", "Final Decision"],
    "MAPE (%)": [
        f"{mape_uni:.2f}",
        f"{mape_multi:.2f}" if mape_multi is not None else "N/A",
        f"{mape_alt:.2f}",
        "Final: " + f"{final_method} ({calculate_mape(test_subset['Units_sold'].values, final_forecast['yhat'].values):.2f})"
    ]
}
metrics_df = pd.DataFrame(metrics)
st.table(metrics_df)

st.write("**Decision Note:**", decision_note)
st.write("**RL Q-Table (state='high'):**", q_values)

# -----------------------------------
# Visualization: Historical Data and Final Forecast
# -----------------------------------
st.write("### Historical Data and Final Forecast")

# Historical data (all training data) in blue
historical_data = train[['Date', 'Units_sold']].copy()
historical_data["Type"] = "Historical"

# Final forecast in green; adjust column names for consistency
forecast_vis = final_forecast.copy().rename(columns={'ds': 'Date', 'yhat': 'Units_sold'})
forecast_vis["Type"] = "Forecast"

# Combine historical and forecast data
plot_df = pd.concat([historical_data[['Date', 'Units_sold', 'Type']], forecast_vis[['Date', 'Units_sold', 'Type']]])
plot_df = plot_df.sort_values("Date")

chart = alt.Chart(plot_df).mark_line().encode(
    x='Date:T',
    y='Units_sold:Q',
    color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Forecast'],
                                                range=['blue', 'green']))
).properties(
    width=700,
    height=400,
    title="Historical Data (Blue) and Final Forecast (Green)"
)
st.altair_chart(chart, use_container_width=True)

# -----------------------------------
# Detailed Forecast Outputs
# -----------------------------------
st.write("## Detailed Forecast Outputs")

st.write("#### Univariate Forecast")
st.dataframe(forecast_uni)

if forecast_multi is not None:
    st.write("#### Multivariate Forecast")
    st.dataframe(forecast_multi)
else:
    st.write("#### Multivariate Forecast: Not available (regressors missing)")

st.write("#### Alternative Forecast")
st.dataframe(forecast_alt)

st.write("#### Final Forecast (Final Decision: " + final_method + ")")
st.dataframe(final_forecast)
