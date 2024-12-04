import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (0.0, 1.0, 0.0, 0.0)
})

# read in anomaly scores df
anomaly_scores = pd.read_csv('data/minutely_simulated_ts_with_anomaly_scores.csv')

# plot histogram of modified z-scores
anomaly_scores['modified_z_score'].hist(bins=100, alpha=0.6)

# plot percentiles at 1% , 0.01% and 0.001% and 99% , 99.99% and 99.999%
percentiles = [0.01, 0.0001, 0.99, 0.9999]
for percentile in percentiles:
    threshold = anomaly_scores['modified_z_score'].quantile(percentile)
    plt.axvline(threshold, color='r', linestyle='dashed')
    percentile_text = round(percentile*100, 2)
#     label line
    plt.text(threshold+1, 100, f'{percentile_text} % percentile: {threshold:.2f}', rotation=90)

plt.title('Z-score distribution and flag rates')
plt.xlabel('z-score')
# plt.legend()
plt.show()
# plt.savefig('img/z_score_distribution_and_flag_rates.png', transparent=True)