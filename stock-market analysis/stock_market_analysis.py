import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Load dataset
data = pd.read_csv('data/stocks.csv')
data['Date'] = pd.to_datetime(data['Date'])

print("✔ Dataset loaded:", data.shape)
print(data.head())

#EDA 
print("\n🔎 Checking for missing values:\n", data.isnull().sum())
print("\n📈 Data Summary:\n", data.describe())

# Distribution of Closing Price
plt.figure(figsize=(8, 5))
plt.hist(data['Close'], bins=20, color='skyblue')
plt.title("Closing Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Volume by Ticker 
volume_by_ticker = data.groupby('Ticker')['Volume'].sum()
volume_by_ticker.plot(kind='bar', color='orange')
plt.title("Total Volume Traded per Ticker")
plt.xlabel("Company")
plt.ylabel("Volume")
plt.tight_layout()
plt.show()

# Corelation Matrice
corr = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#  Volume vs Close Scater 
plt.scatter(data['Volume'], data['Close'], alpha=0.5)
plt.xlabel('Volume')
plt.ylabel('Close Price')
plt.title('Volume vs Closing Price')
plt.tight_layout()
plt.show()

# plot for Closing Price
plt.boxplot(data['Close'])
plt.title('Boxplot: Closing Price Distribution')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

#  Averages per Stock
plt.figure(figsize=(14, 6))
for ticker in data['Ticker'].unique():
    stock_data = data[data['Ticker'] == ticker].copy()
    stock_data.sort_values('Date', inplace=True)
    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
    plt.plot(stock_data['Date'], stock_data['MA20'], label=f"{ticker} MA20")
plt.legend()
plt.title('20-Day Moving Averages per Stock')
plt.xlabel('Date')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

print("\nStock Market Analysis Completed!")
