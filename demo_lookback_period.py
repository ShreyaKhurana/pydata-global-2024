import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (0.0, 1.0, 0.0, 0.0)
})

# read in the minutely df
minutely_df = pd.read_csv('data/minutely_simulated_ts.csv')

# convert ds to datetime
minutely_df['ds'] = pd.to_datetime(minutely_df['ds'])

# introduce an anomaly at 2024-11-06 12:00:00 for 30 mins with y that jumps to 3
anomaly_start = pd.Timestamp('2024-11-06 12:00:00')
anomaly_end = anomaly_start + pd.Timedelta('30 minutes')
minutely_df.loc[(minutely_df['ds'] >= anomaly_start) & (minutely_df['ds'] < anomaly_end), 'y'] = 3

# add some noise to the anomaly period
minutely_df.loc[(minutely_df['ds'] >= anomaly_start) & (minutely_df['ds'] < anomaly_end), 'y'] += np.random.normal(0, 0.1, size=(anomaly_end - anomaly_start).seconds // 60)


# plot
# minutely_df.plot(x='ds', y='y', title='Minutely dataset with anomaly')
# plt.show()


# calculate z-score of this anomalous period from the entire history
''''
mean = minutely_df['y'].mean()
std = minutely_df['y'].std()
minutely_df['z_score'] = (minutely_df['y'] - mean) / std

# get the anomaly period
anomaly_period = minutely_df[(minutely_df['ds'] >= anomaly_start - pd.Timedelta('2 hours')) &
                             (minutely_df['ds'] < anomaly_end + pd.Timedelta('2 hours'))]


# plot the anomaly period for y and z-scores
plt.figure(figsize=(14, 6))
plt.plot(minutely_df['ds'], minutely_df['y'], label='y', alpha=0.7)
plt.plot(minutely_df['ds'], minutely_df['z_score'], 'r', alpha=0.7, label='anomaly score')

plt.title('Minutely dataset with anomaly score')
plt.xlabel('Time')
plt.ylabel('y')
plt.legend()
plt.show()




# calculate mean and std of weekends
minutely_df['weekend'] = minutely_df['ds'].dt.dayofweek >= 5
weekend_mean = minutely_df[minutely_df['weekend']]['y'].mean()
weekend_std = minutely_df[minutely_df['weekend']]['y'].std()
# print(f'Weekend mean: {weekend_mean}, Weekend std: {weekend_std}')
#
# # calculate mean and std of weekdays
weekday_mean = minutely_df[~minutely_df['weekend']]['y'].mean()
weekday_std = minutely_df[~minutely_df['weekend']]['y'].std()
# print(f'Weekday mean: {weekday_mean}, Weekday std: {weekday_std}')


# calculate anomaly score by computing mean and std of last weekend
mean = weekend_mean
std = weekend_std
minutely_df['z_score'] = (minutely_df['y'] - mean) / std



# get the anomaly period
anomaly_period = minutely_df[(minutely_df['ds'] >= anomaly_start - pd.Timedelta('2 hours')) &
                             (minutely_df['ds'] < anomaly_end + pd.Timedelta('2 hours'))]

# plot the anomaly period for y and z-scores
# plt.figure(figsize=(14, 6))
# plt.plot(anomaly_period['ds'], anomaly_period['y'], label='y', alpha=0.7)
# plt.plot(anomaly_period['ds'], anomaly_period['z_score'], 'r', alpha=0.7, label='anomaly score')
#
# plt.title('Minutely dataset with anomaly score')
# plt.xlabel('Time')
# plt.ylabel('y')
# plt.legend()
# plt.show()

plt.figure(figsize=(14, 6))
plt.plot(minutely_df['ds'], minutely_df['y'], label='y', alpha=0.7)
plt.plot(minutely_df['ds'], minutely_df['z_score'], 'r', alpha=0.7, label='anomaly score')

plt.title('Minutely dataset with anomaly score')
plt.xlabel('Time')
plt.ylabel('y')
plt.legend()
plt.show()

# calculate weekend and weekday mean and std and calculate sepaarte z-scores for each
minutely_df['weekend'] = minutely_df['ds'].dt.dayofweek >= 5

weekend_mean = minutely_df[minutely_df['weekend']]['y'].mean()
weekend_std = minutely_df[minutely_df['weekend']]['y'].std()
weekday_mean = minutely_df[~minutely_df['weekend']]['y'].mean()
weekday_std = minutely_df[~minutely_df['weekend']]['y'].std()

minutely_df['z_score'] = np.where(minutely_df['weekend'],
                                    (minutely_df['y'] - weekend_mean) / weekend_std,
                                    (minutely_df['y'] - weekday_mean) / weekday_std)

plt.figure(figsize=(14, 6))
plt.plot(minutely_df['ds'], minutely_df['y'], label='y', alpha=0.7)
plt.plot(minutely_df['ds'], minutely_df['z_score'], 'r', alpha=0.7, label='anomaly score')

plt.title('Minutely dataset with anomaly score')
plt.xlabel('Time')
plt.ylabel('y')
plt.legend()
plt.show()

'''
# calculate mean and std using last 10 minutes on a rolling basis
minutely_df['rolling_mean'] = minutely_df['y'].rolling(window=10).mean()
minutely_df['rolling_std'] = minutely_df['y'].rolling(window=10).std()
minutely_df['z_score'] = (minutely_df['y'] - minutely_df['rolling_mean']) / minutely_df['rolling_std']

plt.figure(figsize=(14, 6))
plt.plot(minutely_df['ds'], minutely_df['y'], label='y', alpha=0.7)
plt.plot(minutely_df['ds'], minutely_df['z_score'], 'r', alpha=0.7, label='anomaly score')

plt.title('Minutely dataset with anomaly score')
plt.xlabel('Time')
plt.ylabel('y')
plt.legend()
plt.show()

