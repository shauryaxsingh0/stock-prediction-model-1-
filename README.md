import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def fetch_stock_data(stock_symbol, start_date, end_date):
    ticker = yf.Ticker(stock_symbol)
    df = ticker.history(start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df


def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

   X = df[['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'Year']]
    # Target variable
    y = df['Close']

   return X, y


def predict_stock_prices(stock_symbol, start_date, end_date):
  
   df = fetch_stock_data(stock_symbol, start_date, end_date)
   X, y = prepare_data(df)


   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


   model = RandomForestRegressor(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)


   predictions = model.predict(X_test)


   mse = mean_squared_error(y_test, predictions)
   print(f"Mean Squared Error: {mse}")


   plt.figure(figsize=(12, 6))
   plt.plot(y_test.values, label="Actual Prices", color="blue")
   plt.plot(predictions, label="Predicted Prices", color="red")
   plt.legend()
   plt.title(f"Actual vs Predicted Prices for {stock_symbol}")
   plt.xlabel("Index")
   plt.ylabel("Price")
   plt.show()


if __name__ == "__main__":
   stock_symbol = "AAPL"
   start_date = "2022-01-01"
   end_date = "2023-01-01"

   predict_stock_prices(stock_symbol, start_date, end_date)
