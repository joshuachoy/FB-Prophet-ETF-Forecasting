# FB Prophet ETF Forecasting
 Forecasting ETF performance with Facebook Prophet

## Forecasting ETF prices
The aim of this project was to utilize Facebook's open source Prophet forecasting model to predict future prices of [State Street's S&P 500 ETF](https://finance.yahoo.com/quote/SPY/). With publicly available data on Yahoo Finance, it was relatively easy to get five year's worth of ETF prices.

The next step was to implement [Facebook's Prophet forecasting model](https://facebook.github.io/prophet/), which is effective and relatively simple to use. It is a robust model that works best well with historical data. Moreover, it's tuning parameters allow for adjustment and increases in accuracy, subjected to various configurations.

### Getting the data
Yahoon Finance provides historical data for free, which you can obtain [here](https://finance.yahoo.com/quote/SPY/history?p=SPY).

```python
# reading the dataset
# 5 years worth of pricing data
df = pd.read_csv('SPY.csv')

# convert Date column to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')

# remove spaces in col headers
df.columns = [str(i).lower().replace(' ','_') for i in df.columns]

# sort values by datetime
df.sort_values(by = 'date', inplace = True, ascending = True)

df.head()
```
![Sample of pricing data](images/df_sample.png)

In this dataset, it has pricing data ranging from 15/6/15 - 12/6/20. We can plot a graph to understand the adjusted closing price over time.
```python
# Plot adjusted closing price of ETF over time
plt = df.plot(x = 'date', y = 'adj_close', linestyle='-', figsize = [10,10], grid = True)
plt.set_title('SPY Index Price')
plt.set_xlabel('Year')
plt.set_ylabel('USD')
```
![SPY Index Adj. closing price](images/SPY_price_2015-2020.png)

### Preparing the data for Prophet
Prophet always takes in a dataframe with 2 columns, ```ds``` (datestamp) & ```y``` (must be numeric), where y is the measurement we want to forecast.
```python
# Create new dataframe for Prophet
# Prophet always takes in 2 columns only, DS & Y
df_prophet = df[['date', 'adj_close']].rename(columns = {'date':'ds', 'adj_close':'y'})
df_prophet.head()
```
![Prophet dataframe](images/prophet_tail.png)

For this project, we want to predict 30 days into the future, from our last available pricing date.

*Let H = 30, where H is our forecast horizon.* 

Using this, we will fit the data into the Prophet model.
```python
# Fitting Prophet model
m = Prophet()
m.fit(df_prophet)
```

Next, create a future dataframe with the forecast horizon.
```python
future = m.make_future_dataframe(periods = H)
```

To make the model more accurate, we should account for weekends and remove them from the model, since they are non-trading days and would not be relevant to future forecasts. More on this can be found [here}(https://facebook.github.io/prophet/docs/non-daily_data.html)
```python
future['day'] = future['ds'].dt.weekday
future = future[future['day'] <5]

forecast = m.predict(future)
```
