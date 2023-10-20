import streamlit as st
import pandas as pd
import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

# Set page configuration with the "Soko Smart Forecasts" title
st.set_page_config(
    page_title="Soko Smart Forecasts",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Replace the data file path with the local path
data_file_path =  "https://raw.githubusercontent.com/karanja-john/Food-prices-project/main/wfp_food_prices_kenunitprices.csv"
data = pd.read_csv(data_file_path)

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Filter data based on the selected crop
selected_crop = st.selectbox("Select a crop", ["Maize", "Beans"])
crop_data = data[data['commodity'] == selected_crop].copy()  # Use copy to avoid chained assignment

# Data Preprocessing (scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
crop_data['Unitprice_scaled'] = scaler.fit_transform(crop_data['Unitprice'].values.reshape(-1, 1))

# Set the best parameters
best_order = (1, 0, 1)
best_seasonal_order = (1, 0, 1, 12)

# Define a function to forecast prices for a selected date
def forecast_prices(selected_date, forecast_horizon):
    # Convert the selected_date to Pandas Timestamp for comparison
    selected_date = pd.Timestamp(selected_date)

    # Split the data into training and testing sets based on the selected date
    train = crop_data[crop_data['date'] < selected_date]['Unitprice_scaled']

    # Fit a SARIMAX model with the best parameters based on the available data
    sarimax_model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
    sarimax_result = sarimax_model.fit()

    # Forecast prices for the specified forecast horizon (in days)
    forecasted_values = sarimax_result.get_forecast(steps=forecast_horizon)
    forecasted_values = scaler.inverse_transform(forecasted_values.predicted_mean.values.reshape(-1, 1))
    return forecasted_values

# Set a maximum forecast horizon (e.g., 365 days) from the last available date
max_forecast_date = crop_data['date'].max() + pd.DateOffset(days=365)

# Date Selection Widget with a maximum date
selected_date = st.date_input("Select a date for price forecast", value=datetime.datetime.now().date(), max_value=max_forecast_date)

# Forecast horizon input
forecast_horizon = st.number_input("Enter the forecast horizon (in days):", min_value=1, max_value=365, value=30)

# Forecast button
if st.button("Forecast"):
    # Check if the selected date is within the forecast range
    if selected_date <= max_forecast_date:
        forecasted_values = forecast_prices(selected_date, forecast_horizon)
        st.success(f"Dynamic price forecast for {selected_crop} starting from {selected_date} for the next {forecast_horizon} days:")
        st.line_chart(pd.DataFrame({'Forecasted Prices': forecasted_values.flatten()}, index=pd.date_range(start=selected_date, periods=forecast_horizon)))
    else:
        st.warning(f"The selected date is outside the maximum forecast range. Please select a date within {max_forecast_date}.")
