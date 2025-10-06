# HOLT WINTERS METHOD
 
# AIM:
To implement the Holt Winters Method Model using Python.

# ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions

# PROGRAM:
```python
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load dataset
file_path = "asiacup.csv"
data = pd.read_csv(file_path)

# Clean column names
data.columns = data.columns.str.strip()

# Correct column names
year_col = 'Year'
strike_col = 'Avg Bat Strike Rate'

# Convert Year to datetime
data[year_col] = pd.to_datetime(data[year_col], format='%Y', errors='coerce')

# Drop missing values
data = data.dropna(subset=[year_col, strike_col])

# Convert strike rate to numeric
data[strike_col] = pd.to_numeric(data[strike_col], errors='coerce')

# Set Year as index
data.set_index(year_col, inplace=True)

# Prepare yearly average
yearly_data = data[strike_col].resample('YS').mean()
yearly_data = yearly_data.dropna()

# Split into training (90%) and testing (10%)
split_index = int(0.9 * len(yearly_data))
train_data = yearly_data[:split_index]
test_data = yearly_data[split_index:]

# Decide on seasonality
if len(train_data) >= 6:
    seasonal_periods = 3  # 3-year seasonality (tweak if needed)
    seasonal_type = 'add'
    print("Using seasonal model.")
else:
    seasonal_periods = None
    seasonal_type = None
    print("Using non-seasonal model (not enough data).")

# Fit Holt-Winters model
fitted_model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal=seasonal_type,
    seasonal_periods=seasonal_periods
).fit()

# Forecast for test period
test_predictions = fitted_model.forecast(len(test_data))

# Plot train/test/predicted
plt.figure(figsize=(12, 8))
train_data.plot(label='Train')
test_data.plot(label='Test')
test_predictions.plot(label='Predicted', color='red', linestyle='dashed')
plt.title('Train, Test, and Predicted: Avg Bat Strike Rate')
plt.xlabel('Year')
plt.ylabel('Average Batting Strike Rate')
plt.legend()
plt.grid(True)
plt.show()

# Error metrics
mae = mean_absolute_error(test_data, test_predictions)
mse = mean_squared_error(test_data, test_predictions)
print(f"Mean Absolute Error = {mae:.4f}")
print(f"Mean Squared Error = {mse:.4f}")

# Retrain full model and forecast next 3 years
final_model = ExponentialSmoothing(
    yearly_data,
    trend='add',
    seasonal=seasonal_type,
    seasonal_periods=seasonal_periods
).fit()

forecast_predictions = final_model.forecast(steps=3)

# Plot original + forecast
plt.figure(figsize=(12, 8))
yearly_data.plot(label='Original Data')
forecast_predictions.plot(label='Forecasted (Next 3 Years)', color='purple', linestyle='dashed')
plt.title('Original and Forecasted Avg Bat Strike Rate')
plt.xlabel('Year')
plt.ylabel('Average Batting Strike Rate')
plt.legend()
plt.grid(True)
plt.show()

```

### OUTPUT:

## TEST_PREDICTION

<img width="1122" height="696" alt="image" src="https://github.com/user-attachments/assets/c96abb5a-123f-4935-b9a1-654a07979591" />

### FINAL_PREDICTION

<img width="1122" height="682" alt="image" src="https://github.com/user-attachments/assets/c10ac21a-c58a-4154-b2a0-fd8353331652" />



# RESULT:
Thus the program run successfully based on the Holt Winters Method model.
