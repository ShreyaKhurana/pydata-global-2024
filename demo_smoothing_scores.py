import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0)
})

# read minutely data with anomaly_scores
df = pd.read_csv('data/minutely_simulated_ts_with_anomaly_scores.csv')

# plot anomaly scores and median of anomaly scores
df['modified_z_score'].plot()
df['modified_z_score'].rolling(5).median().plot(color='red')
plt.title('Anomaly scores and Smoothing')
plt.xlabel('Time')
plt.ylabel('Anomaly scores')
plt.legend(['Anomaly scores', '5-minute rolling median of anomaly scores'])
plt.show()

