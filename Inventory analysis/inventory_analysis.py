import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler



# importing the datset
purchase_price = pd.read_csv('data/2017PurchasePricesDec.csv')
beg_inv = pd.read_csv('data/BegInvFINAL12312016.csv')
end_inv = pd.read_csv('data/EndInvFINAL12312016.csv')
invoice = pd.read_csv('data/InvoicePurchases12312016.csv')
final_purchase = pd.read_csv('data/PurchasesFINAL12312016.csv')
final_sales = pd.read_csv('data/SalesFINAL12312016.csv')



#  forcasting demand 
print("\n[1] Demand Forecasting (ARIMA)")
final_sales['SalesDate'] = pd.to_datetime(final_sales['SalesDate'], errors='coerce')
sales_ts = final_sales.groupby('SalesDate')['SalesDollars'].sum()

model = ARIMA(sales_ts, order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)
print("Next 30 days forecast:\n", forecast.head())

# analysis
print("\n[2] Analysis")
total_sales = final_sales.groupby('Description')['SalesDollars'].sum().sort_values(ascending=False)
cum_pct = total_sales.cumsum() / total_sales.sum()

abc_class = pd.cut(cum_pct, bins=[0, 0.8, 0.95, 1.0], labels=['A', 'B', 'C'])
abc_df = pd.DataFrame({'Sales': total_sales, 'Class': abc_class})
print(abc_df.groupby('Class')['Sales'].sum())

# EOQ Analysis
print("\n[3] EOQ Analysis (assumes demand = quantity sold)")
annual_demand = final_sales.groupby('Description')['SalesQuantity'].sum()
ordering_cost = 50  
carrying_cost_rate = 0.2  

eoq_df = pd.merge(annual_demand, purchase_price[['Description', 'PurchasePrice']], on='Description', how='left')
eoq_df = eoq_df.dropna()
eoq_df['CarryingCost'] = eoq_df['PurchasePrice'] * carrying_cost_rate
eoq_df['EOQ'] = np.sqrt((2 * ordering_cost * eoq_df['SalesQuantity']) / eoq_df['CarryingCost'])
print(eoq_df[['Description', 'EOQ']].head())

#Reorder Point
print("\n[4] Reorder Point Calculation")
lead_time_days = 10  # assumption
avg_daily_demand = eoq_df['SalesQuantity'] / 365
eoq_df['ReorderPoint'] = avg_daily_demand * lead_time_days
print(eoq_df[['Description', 'ReorderPoint']].head())

#Inventory Turnover Ratio 
print("\n[5] Inventory Turnover Ratio")
COGS = final_sales.groupby('Description')['SalesDollars'].sum()
avg_inventory = (beg_inv['onHand'] * beg_inv['Price']).sum() + (end_inv['onHand'] * end_inv['Price']).sum()
avg_inventory /= 2
turnover_ratio = COGS.sum() / avg_inventory
print(f"Inventory Turnover Ratio: {turnover_ratio:.2f}")

# Carrying Cost Analysis 
print("\n[6] Carrying Cost Analysis")
avg_inventory_qty = (beg_inv['onHand'].sum() + end_inv['onHand'].sum()) / 2
avg_cost = purchase_price['PurchasePrice'].mean()
carrying_cost = avg_inventory_qty * avg_cost * carrying_cost_rate
print(f"Annual Carrying Cost Estimate: ₹{carrying_cost:,.2f}")

#  Data Visualization
print("\n[7] Top 5 Selling Products by Revenue")
top5 = total_sales.head(5)
top5.plot(kind='barh', color='skyblue')
plt.title('Top 5 Products by Sales Revenue')
plt.xlabel('Revenue')
plt.tight_layout()
plt.show()

