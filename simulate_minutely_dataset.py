# simulate a minutely frequency timeseries dataset for one week, with daily and weekly seasonality such that daily and weekly
# seasonality shows some clear curves i.e. weekends have more variance than weekdays and working hours for example
# like 9am -6 pm have lesser variance than other hours
# also add random noise so it is not too easy to predict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (0.0, 1.0, 0.0, 0.0)
})

# Create a date range for one week with minute frequency
idx = pd.date_range(start='2024-11-01', periods=14*24*60, freq='T')

# Generate base time-series data
data = np.sin(np.array(range(len(idx))) * (2 * np.pi / (24*60)))  # Daily seasonality
data += np.sin(np.array(range(len(idx))) * (2 * np.pi / (7*24*60)))  # Weekly seasonality

# Add hourly variance effects based on working hours (09:00 to 18:00)
hour = idx.hour
work_hours = ((hour >= 9) & (hour < 18))
data[work_hours] *= 0.1

# Introduce additional variance and noise during weekends
weekend = (idx.weekday >= 5)
data[weekend] += np.random.normal(0, 0.3, size=data[weekend].shape)

# General random noise
data += np.random.normal(0, 0.1, size=data.shape)

# Create DataFrame
df = pd.DataFrame(data, index=idx, columns=['value'])


# Plot the data
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['value'], label='Minutely timeseries data')
plt.title('Minutely dataset')
plt.xlabel('Time')
plt.ylabel('y')
plt.show()

# use index column as ds and value column as y
df.reset_index(inplace=True)
df.columns = ['ds', 'y']

# save as csv
# df.to_csv('data/minutely_simulated_ts.csv', index=False)


'''
github copilot generated code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the time range
date_range = pd.date_range(start='2024-11-01', end='2024-11-07 23:59:00', freq='T')

# Initialize the DataFrame
df = pd.DataFrame(index=date_range)
df['timestamp'] = df.index

# Seasonality
daily_wave = 10 * (df['timestamp'].dt.hour >= 9) & (df['timestamp'].dt.hour <= 18)
weekly_wave = 5 * (df['timestamp'].dt.dayofweek < 5)
df['seasonality'] = daily_wave + weekly_wave

# Random base noise
random_noise = np.random.normal(0, 0.5, len(df))

# Amplified noise on weekends
df['y'] = df['seasonality'] + random_noise
df['y'] = df['y'] + (df['timestamp'].dt.dayofweek >= 5) * 7

# Plot the data
df['y'].plot()
plt.title('Minutely dataset')
plt.xlabel('Time')
plt.ylabel('y')
plt.show()
'''

