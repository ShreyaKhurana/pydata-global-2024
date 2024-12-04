import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0)
})


# read in the minutely df
minutely_df = pd.read_csv('data/minutely_simulated_ts.csv')

# convert ds to datetime
minutely_df['ds'] = pd.to_datetime(minutely_df['ds'])

# introduce an anomaly at 2024-11-06 12:00:00 for 30 mins with y that jumps to 3 with some noise
anomaly_start = pd.Timestamp('2024-11-06 12:00:00')
anomaly_end = anomaly_start + pd.Timedelta('30 minutes')
minutely_df.loc[(minutely_df['ds'] >= anomaly_start) & (minutely_df['ds'] < anomaly_end), 'y'] = 3

# add noise to the anomaly period
minutely_df.loc[(minutely_df['ds'] >= anomaly_start) & (minutely_df['ds'] < anomaly_end), 'y'] += np.random.normal(0, 0.1, size=(anomaly_end - anomaly_start).seconds // 60)
# plot
# minutely_df.plot(x='ds', y='y', title='Minutely dataset with anomaly')
# plt.show()

# calculate rolling means and std based on the last hour
minutely_df['rolling_mean'] = minutely_df['y'].rolling(60).mean()
minutely_df['rolling_std'] = minutely_df['y'].rolling(60).std()

# calculate z-score based on the rolling mean and std
minutely_df['z_score'] = (minutely_df['y'] - minutely_df['rolling_mean']) / minutely_df['rolling_std']

# calculate rolling median and rolling median absolute deviation based on the last hour
minutely_df['rolling_median'] = minutely_df['y'].rolling(60).median()
minutely_df['absolute_deviation'] = np.abs(minutely_df['y'] - minutely_df['rolling_median'])
minutely_df['rolling_mad'] = minutely_df['absolute_deviation'].rolling(60).median()

# calculate z-score based on the rolling median and rolling mad
minutely_df['modified_z_score'] = 0.6745 * (minutely_df['y'] - minutely_df['rolling_median']) / minutely_df['rolling_mad']

# plot y and anomaly scores
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111)
ax.plot(minutely_df['ds'], minutely_df['y'], label='y', alpha=0.7)
ax.plot(minutely_df['ds'], minutely_df['z_score'], 'r', alpha=0.7, label='Mean-based z-score')

ax2 = plt.gca().twinx()
ax2.plot(minutely_df['ds'], minutely_df['modified_z_score'], 'g', alpha=0.7, label='Median-based modified z-score')

ax.set_title('Minutely dataset with anomaly score')
ax.set_xlabel('Time')
ax.set_ylabel('Time series and anomaly scores')

ax.legend(loc='upper right')
ax2.legend(loc='upper left')
plt.show()

# save the modified z-score to csv
minutely_df.to_csv('data/minutely_simulated_ts_with_anomaly_scores.csv', index=False)

