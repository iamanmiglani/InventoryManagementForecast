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
    # Clean column names: remove any leading/trailing spaces
    data.columns = data.columns.str.strip()
    return data

def split_data(data):
    # Training data: all dates up to November 30, 2023
    train = data[data['Date'] <= pd.to_datetime("2023-11-30")].copy()
    # Test data: the first 10 days of December 2023
    test = data[(data['Date'] >= pd.to_datetime("2023-12-01")) &
                (data['Date'] <= pd.to_datetime("2023-12-10"))].copy()
    return train, test

# -----------------------------------
# Forecasting Functions
# -----------------------------------
def forecast_prophet_univariate(train, periods):
    # Prophet requires columns named 'ds' (date) and 'y' (target)
    df_train = train[['Date', 'Units_sold']].rename(columns={'Date': 'ds', 'Units_sold': 'y'})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast

def forecast_prophet_multivariate(train, periods, additional_regressors):
    # Build a Prophet model with extra regressors
    df_train = train.copy().rename(columns={'Date': 'ds', 'Units_sold': 'y'})
    m = Prophet()
    for reg in additional_regressors:
        m.add_regressor(reg)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=periods)
    # For additional regressors, we simply use the last observed value as a placeholder.
    for reg in additional_regressors:
        future[reg] = df_train[reg].iloc[-1]
    forecast = m.predict(future)
    return forecast

def alternative_forecast_model(train, test):
    # Example alternative: use the last observed value for all test dates.
    last_value = train['Units_sold'].iloc[-1]
    forecast_values = [last_value] * len(test)
    # Return a DataFrame to match Prophet's output format
    return pd.DataFrame({"yhat": forecast_values}, index=test['Date'])

def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

# -----------------------------------
# RL Agent Helper: Forecast Method Selection
# -----------------------------------
def choose_forecast_method(q_table, state="high"):
    """
    Given a trained Q-table and current state (here, "high" indicates forecast error is high),
    return the forecast method with the highest Q-value.
    """
    actions = ["univariate", "multivariate", "alternative"]
    # Get Q-values for each action; default to 0 if not present.
    possible_actions = {action: q_table.get((state, action), 0) for action in actions}
    best_action = max(possible_actions, key=possible_actions.get)
    return best_action

# -----------------------------------
# Main Streamlit App
# -----------------------------------
st.title("Agentic Forecasting with RL-Controlled Forecast Layer")

# Load and show data preview
data = load_data()
st.write("### Data Overview")
st.dataframe(data.head())

# Split data into training and testing (10 dates in December)
train, test = split_data(data)

# -----------------------------------
# Load Trained RL Q-Table
# -----------------------------------
try:
    with open("trained_rl_agent.pkl", "rb") as f:
        trained_q_table = pickle.load(f)
except Exception as e:
    st.error("Error loading the RL Q-table. Make sure 'trained_rl_agent.pkl' exists. " + str(e))
    st.stop()

# -----------------------------------
# RL Layer Decides the Forecast Method
# -----------------------------------
# In a production scenario the RL layer would dynamically update the state.
# Here we assume a current "high" error state, so the RL agent selects a method
current_state = "high"
forecast_method = choose_forecast_method(trained_q_table, current_state)
st.write(f"**RL Agent recommends forecast method:** `{forecast_method}`")

# -----------------------------------
# Execute Forecast Based on RL Decision
# -----------------------------------
periods = len(test)  # should be 10 for December 1-10

if forecast_method == "univariate":
    forecast = forecast_prophet_univariate(train, periods)
    # Extract only the forecast for the test period (the tail of the forecast)
    forecast_test = forecast[['ds', 'yhat']].tail(periods)
elif forecast_method == "multivariate":
    # Make sure additional regressors exist; here we expect 'Promotion' and 'Price'
    if not all(col in train.columns for col in ['Promotion', 'Price']):
        st.error("Required regressors (Promotion, Price) not found in the data.")
        st.stop()
    forecast = forecast_prophet_multivariate(train, periods, additional_regressors=['Promotion', 'Price'])
    forecast_test = forecast[['ds', 'yhat']].tail(periods)
elif forecast_method == "alternative":
    forecast_test = alternative_forecast_model(train, test)
    # Create a dummy 'ds' column for consistency if needed
    forecast_test = forecast_test.reset_index().rename(columns={'index': 'ds'})
else:
    st.error("No valid forecast method chosen by the RL agent.")
    st.stop()

# -----------------------------------
# Compute and Display Performance
# -----------------------------------
# Make sure test data index aligns with forecast dates.
# If forecast_test contains a 'ds' column with dates, extract those for alignment.
actual = test['Units_sold'].values
if 'ds' in forecast_test.columns:
    forecast_values = forecast_test['yhat'].values
else:
    forecast_values = forecast_test['yhat'].values

mape = calculate_mape(actual, forecast_values)
st.write(f"**Forecast MAPE:** {mape:.2f}%")

# Display forecast method and chart
st.write("### Forecast vs Actual")
results_df = pd.DataFrame({
    "Actual": actual,
    "Forecast": forecast_values
}, index=test['Date'])
st.line_chart(results_df)
