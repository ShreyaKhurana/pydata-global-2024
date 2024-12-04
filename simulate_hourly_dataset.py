import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# simulate an hourly frequency timeseries dataset with trend, daily and weekly seasonality such that daily and weekly
# seasonality shows some clear curves and add some effect
# on Thanksgiving and christmas
# but also add random noise so it is not too easy to predict,
# Add more noise on some days than others and some trends that change throughout the year to the time series
# such that it looks like it is upward trending curves


# Define the time range
date_range = pd.date_range(start='2023-01-01', end='2024-12-31 23:00:00', freq='h')

# Initialize the DataFrame
df = pd.DataFrame(index=date_range)
df['timestamp'] = df.index

# Define trend changes
def create_trend(date):
    if date < pd.Timestamp('2023-07-01'):
        # Linear increase from January to July 2023
        return np.interp((date - pd.Timestamp('2023-01-01')).days, [0, 181], [0, 20])
    elif date < pd.Timestamp('2024-02-01'):
        # Linear decrease from July 2023 to February 2024
        return np.interp((date - pd.Timestamp('2023-07-01')).days, [0, 214], [20, 10])
    else:
        # Linear increase from February 2024 to December 2024
        return np.interp((date - pd.Timestamp('2024-02-01')).days, [0, 333], [10, 30])

df['trend'] = df['timestamp'].apply(create_trend)

# Seasonality
daily_wave = 10 * np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
weekly_wave = 5 * np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
df['seasonality'] = daily_wave + weekly_wave

# Special effects
holiday_weeks = {
    "Thanksgiving": ["2023-11-20/2023-11-26", "2024-11-25/2024-12-01"],
    "Christmas": ["2023-12-23/2023-12-29", "2024-12-24/2024-12-30"],
    "NewYear": ["2023-12-30/2024-01-05", "2024-12-30/2025-01-05"]
}

def apply_weekly_effects(date, holidays):
    for holiday_name, periods in holidays.items():
        for period in periods:
            start_date, end_date = (pd.to_datetime(p) for p in period.split('/'))
            if start_date <= date <= end_date:
                days_to_holiday = (end_date - date).days
                return -10 * (7 - days_to_holiday)  # Intensify effect as date approaches
    return 0

df['special_effects'] = df['timestamp'].apply(lambda x: apply_weekly_effects(x, holiday_weeks))

# Random base noise
random_noise = np.random.normal(0, 2, len(df))

# Amplified noise on weekends
weekend_noise_multiplier = 3
df['weekend_noise'] = np.where(df['timestamp'].dt.dayofweek.isin([5, 6]),
                               np.random.normal(0, weekend_noise_multiplier * 5, len(df)),
                               0)

# Combine all effects
df['value'] = df['trend'] + df['seasonality'] + df['special_effects'] + random_noise + df['weekend_noise']

# rename columns to ds and y and save only these columns
df = df[['timestamp', 'value']]
df.columns = ['ds', 'y']

# save as csv
df.to_csv('data/hourly_simulated_ts.csv', index=False)

# Display the first few rows of the DataFrame
print(df.head())

# Plotting
plt.figure(figsize=(14, 7))
df['y'].plot(title='Time Series Simulation with Adjusted Trend')
# plt.show()