# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

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
    # Prophet requires 'ds' (date) and 'y' (target) columns.
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
    # Add extra regressors (e.g., Promotion, Price)
    for reg in regressors:
        model.add_regressor(reg)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=forecast_days)
    # For future dates, simply fill with the last observed value for each regressor.
    for reg in regressors:
        future[reg] = df_train[reg].iloc[-1]
    forecast = model.predict(future)
    forecast_test = forecast[['ds', 'yhat']].tail(forecast_days)
    return forecast_test

def alternative_forecast_model(train, test_subset):
    # A simple alternative: use the last observed value from training data.
    last_value = train['Units_sold'].iloc[-1]
    forecast_values = [last_value] * len(test_subset)
    # Create a DataFrame with dates from the test_subset
    forecast_test = pd.DataFrame({"yhat": forecast_values}, index=test_subset['Date'])
    forecast_test = forecast_test.reset_index().rename(columns={'index': 'ds'})
    return forecast_test

def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

# -----------------------------------
# RL Q-Table Based Forecast Selection
# -----------------------------------
def choose_forecast_method(q_table, state="high"):
    """
    Given the Q-table and a state (e.g., "high" meaning high forecast error),
    return the forecast method with the highest Q-value.
    """
    actions = ["univariate", "multivariate", "alternative"]
    # Get Q-values for each action (defaulting to 0)
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
st.title("Agentic Forecasting Pipeline with RL")
data = load_data()
st.write("### Data Overview", data.head())

train, test = split_data(data)
# Use only the first 'forecast_days' rows from test data for comparison
test_subset = test.head(forecast_days)

# -----------------------------------
# Compute All Forecasts
# -----------------------------------
st.write("## Forecast Results")

# Univariate Forecast
forecast_uni = forecast_prophet_univariate(train, forecast_days)
mape_uni = calculate_mape(test_subset['Units_sold'].values, forecast_uni['yhat'].values)

# Multivariate Forecast (if extra regressors are present)
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
# Load Trained RL Q-Table and Decide Final Forecast
# -----------------------------------
try:
    with open("trained_rl_agent.pkl", "rb") as f:
        trained_q_table = pickle.load(f)
except Exception as e:
    st.error("Error loading the RL Q-table. Please check that 'trained_rl_agent.pkl' exists.")
    st.stop()

# Here we assume that our current state is "high" (forecast error is high)
current_state = "high"
final_method, q_values = choose_forecast_method(trained_q_table, current_state)
st.write(f"**RL Agent recommends forecast method:** `{final_method}`")
st.write("**Q-values for each action:**", q_values)

# Select the final forecast based on RL decision
if final_method == "univariate":
    final_forecast = forecast_uni
elif final_method == "multivariate":
    if forecast_multi is not None:
        final_forecast = forecast_multi
    else:
        st.warning("Multivariate regressors not available. Falling back to univariate forecast.")
        final_forecast = forecast_uni
elif final_method == "alternative":
    final_forecast = forecast_alt
else:
    st.error("No valid forecast method chosen by the RL agent.")
    st.stop()

# Get final forecast MAPE
final_mape = calculate_mape(test_subset['Units_sold'].values, final_forecast['yhat'].values)

# -----------------------------------
# Display Forecast Metrics
# -----------------------------------
st.write("### Forecast Comparison Metrics")
metrics = {
    "Forecast Method": ["Univariate", "Multivariate", "Alternative", "Final (RL)"],
    "MAPE (%)": [
        f"{mape_uni:.2f}",
        f"{mape_multi:.2f}" if mape_multi is not None else "N/A",
        f"{mape_alt:.2f}",
        f"{final_mape:.2f}"
    ]
}
metrics_df = pd.DataFrame(metrics)
st.table(metrics_df)

# -----------------------------------
# Visualization: Historical and Final Forecast
# -----------------------------------
st.write("### Historical Data and Final Forecast")

# Create a DataFrame for historical data (all training data)
historical_data = train[['Date', 'Units_sold']].copy()

# Merge the historical data with final forecast:
# - Plot historical data in blue.
# - Plot forecasted data in green.
historical_data['Type'] = 'Historical'
final_forecast['Type'] = 'Forecast'
final_forecast = final_forecast.rename(columns={'ds': 'Date', 'yhat': 'Units_sold'})

# Combine the two dataframes
plot_df = pd.concat([
    historical_data[['Date', 'Units_sold', 'Type']],
    final_forecast[['Date', 'Units_sold', 'Type']]
])

# Sort by date for proper plotting
plot_df = plot_df.sort_values("Date")

# Plot using Streamlit's built-in charting with color differentiation.
import altair as alt
chart = alt.Chart(plot_df).mark_line().encode(
    x='Date:T',
    y='Units_sold:Q',
    color='Type:N'
).properties(
    width=700,
    height=400,
    title="Historical Data (Blue) and Forecast (Green)"
)
st.altair_chart(chart, use_container_width=True)

# -----------------------------------
# Detailed Forecast Outputs (Optional)
# -----------------------------------
st.write("## Detailed Forecast Outputs")

st.write("#### Univariate Forecast")
st.dataframe(forecast_uni)

if forecast_multi is not None:
    st.write("#### Multivariate Forecast")
    st.dataframe(forecast_multi)
else:
    st.write("#### Multivariate Forecast: Not available (required regressors missing)")

st.write("#### Alternative Forecast")
st.dataframe(forecast_alt)

st.write("#### Final Forecast (as chosen by RL Agent)")
st.dataframe(final_forecast)
