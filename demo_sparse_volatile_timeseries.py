import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (0.0, 0.0, 0.0, 0.0)
})

# simulate a timeseries for 5 hours with 1 minute frequency
# that is extremely volatile and fluctuates between 0 and 10
# and generate an anomaly for 10 minutes somewhere in the middle

# Create a date range for one week with minute frequency
idx = pd.date_range(start='2024-11-01', periods=5*60, freq='min')

# Generate base time-series data
data = np.random.uniform(0, 10, len(idx))

# Introduce an anomaly for 10 minutes
anomaly_start = 2*60
anomaly_end = anomaly_start + 20
data[anomaly_start:anomaly_end] += 3

# introduce some noise
data[anomaly_start - 10] += 1
data[anomaly_end + 15] += 1

# Create DataFrame
df = pd.DataFrame(data, index=idx, columns=['value'])
print(df.iloc[anomaly_start:anomaly_end+1])
# Plot the data as is, then aggregated into 2 mins, 5 mins, 10 mins in 4 different plots and plot the anomaly in
# red color
fig, axs = plt.subplots(4, 1, figsize=(5, 8), sharex=True, sharey=True)
axs[0].plot(df.index, df['value'], alpha=0.7)
axs[0].axvspan(df.iloc[anomaly_start].name, df.iloc[anomaly_end].name, color='red', alpha=0.1, label='Anomaly')
axs[0].set_title('Minutely dataset')
axs[0].set_ylabel('y')

# Aggregate the data into 2 mins
df_2min = df.resample('2min').mean()
axs[1].plot(df_2min.index, df_2min['value'], c='cyan', alpha=0.7)
axs[1].axvspan(df_2min.iloc[anomaly_start//2].name, df_2min.iloc[anomaly_end//2].name,
                  color='red', alpha=0.1, label='Anomaly')
axs[1].set_title('2 mins aggregated dataset')
axs[1].set_ylabel('y')

# Aggregate the data into 5 mins
df_5min = df.resample('5min').mean()
axs[2].plot(df_5min.index, df_5min['value'], c='g', label='5 mins aggregated timeseries data', alpha=0.7)
axs[2].axvspan(df_5min.iloc[anomaly_start//5].name, df_5min.iloc[anomaly_end//5].name,
                  color='red', alpha=0.1, label='Anomaly')
axs[2].set_title('5 mins aggregated dataset')
axs[2].set_ylabel('y')

# Aggregate the data into 10 mins
df_10min = df.resample('10min').mean()
axs[3].plot(df_10min.index, df_10min['value'], c='purple', alpha=0.7)
axs[3].axvspan(df_10min.iloc[anomaly_start//10].name, df_10min.iloc[anomaly_end//10].name,
                  color='red', alpha=0.1, label='Anomaly')
axs[3].set_title('10 mins aggregated dataset')
axs[3].set_xlabel('Time')
axs[3].set_ylabel('y')
axs[3].legend(facecolor='white', framealpha=0)
plt.show()


