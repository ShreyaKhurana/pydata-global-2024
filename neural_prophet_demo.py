import pandas as pd
from neuralprophet import NeuralProphet
import plotly
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

df = pd.read_csv("data/hourly_simulated_ts.csv")

# plot the data
fig = plt.figure(figsize=(14, 6))
plt.plot(df["ds"], df["y"], label="Hourly timeseries data")
plt.show()


m = NeuralProphet(epochs=10, n_changepoints=2, n_lags=0)
# Use static plotly in notebooks
m.set_plotting_backend("plotly")

# Fit the model on the dataset (this might take a bit)
metrics = m.fit(df)

# Create a new dataframe reaching 365 into the future for our forecast, n_historic_predictions also shows historic data
df_future = m.make_future_dataframe(df, n_historic_predictions=True, periods=24*7)
#
# Predict the future
forecast = m.predict(df_future)

# Visualize the forecast
# fig = m.plot(forecast)
# fig.show()

# fig = m.plot_parameters(components=["trend"])
# fig.show()

# df_residuals = pd.DataFrame({"ds": df["ds"], "residuals": df["y"] - forecast["yhat1"]})
# fig = df_residuals.plot(x="ds", y="residuals", figsize=(10, 6))
# plt.show()

# fig = plot_acf(df_residuals["residuals"], lags=24*14)
# plt.show()

# fig = plot_pacf(df_residuals['residuals'], lags=24*14)
# plt.show()


# model with n_lags=12 to show the effect of autocorrelation
# m = NeuralProphet(epochs=10, n_changepoints=2, n_lags=12)
# metrics = m.fit(df)
# forecast = m.predict(df)
# df_residuals = pd.DataFrame({"ds": df["ds"], "residuals": df["y"] - forecast["yhat1"]})
# fig = df_residuals.plot(x="ds", y="residuals", figsize=(10, 6))
# plt.show()

'''
# adding events information for Thanksgiving and Christmas week
holiday_weeks = {
    "Thanksgiving": ["2023-11-20/2023-11-26", "2024-11-25/2024-12-01"],
    "Christmas": ["2023-12-23/2023-12-29", "2024-12-24/2024-12-30"],
    "NewYear": ["2023-12-30/2024-01-05", "2024-12-30/2025-01-05"]
}

# in the above dict, the key is the name of the holiday and the value is the date range for that holiday
# we can make a df_events_df with the columns event and ds, and ds column has all hours of the dates falling in the ranges
# mentioned above
df_events = pd.DataFrame()
for holiday, periods in holiday_weeks.items():
    for period in periods:
        start_date, end_date = (pd.to_datetime(p) for p in period.split('/'))
        df_events = pd.concat([df_events, pd.DataFrame({"event": "holiday", "ds": pd.date_range(start_date, end_date,
                                                                                                freq="h")})])

print(df_events.head())

# m = NeuralProphet(epochs=10, n_changepoints=2, n_lags=0)
m.add_events("holiday")

df_all = m.create_df_with_events(df, df_events)

metrics = m.fit(df_all)
forecast = m.predict(df_all)
fig = m.plot(forecast)
fig.show()
'''
