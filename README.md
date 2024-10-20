# Python RFM Model Analysis
The project aims to analyze customer behavior to enhance marketing strategies for data analysts and marketing teams.

## Table of Contents:
1. [Overall](#overall)
2. [Data Cleaning](#clean)
3. [RFM Segmentation](#rfm)
4. [Visualization](#vis)
5. [Insights & Recommendations](#insight)

<div id='overall'/>
  
## 1. Overall

**Platform**: Google Colab

**Main Techniques**: Exploratory data analysis (EDA), data transformations, aggregation and grouping.

**Context**: SuperStore, a global retail company with a large customer base, seeks to segment customers for its marketing campaigns. Marketing department has requested the Data Analysis Department's assistance in implementing the RFM model.

**Goal**: Based on the dataset, I use Python to classify customers into RFM segment, then visualize results to propose key insights and actionable solutions.
  
**Links to dataset info:** https://docs.google.com/spreadsheets/d/1yNt8-kkoDyYzq8tYbqWRqqrAfyhPNtBlfSo-9aRvbCY/view

<div id='clean'/>
  
## 2. Data Cleaning
### Import libraries used

```python
!pip install squarify

from google.colab import auth
auth.authenticate_user()

import gspread
from google.auth import default
creds, _ = default()
gc = gspread.authorize(creds)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
```
### Import data

```python
data_link='1yNt8-kkoDyYzq8tYbqWRqqrAfyhPNtBlfSo-9aRvbCY'
data = gc.open_by_key(data_link)

order_info = pd.DataFrame(data.worksheet('ecommerce retail').get_all_records())
segmentation = pd.DataFrame(data.worksheet('Segmentation').get_all_records())

print(order_info.head())
print('\n', segmentation.head())
```
### Prepare segmentation list

```python
segmentation.columns = ['segment','score']

segmentation['score'] = segmentation['score'].str.split(',')
segmentation = segmentation.explode('score')

segmentation['score'] = segmentation['score'].str.strip()

segmentation.info()
```
### Order info data cleaning

```python
# drop duplicate
order_info = order_info.drop_duplicates()

# drop cancel order rows
order_info = order_info[~order_info['InvoiceNo'].astype(str).str.startswith('C')]

# remove adjust bad debt order by remove row where InvoiceNo not int
order_info = order_info[order_info['InvoiceNo'].apply(lambda x: isinstance(x, int))]

# drop rows where Quantity or UnitPrice is negative or 0, or CustomerID is empty
order_info = order_info[
    (order_info['Quantity'] > 0) &
    (order_info['CustomerID'] != '') &
    (order_info['UnitPrice'] > 0)
]

# Convert InvoiceDate to datetime
order_info['InvoiceDate'] = pd.to_datetime(order_info['InvoiceDate'], format='%m/%d/%y %H:%M', errors='coerce')
# Extract date
order_info['Date'] = order_info['InvoiceDate'].dt.date
order_info['Date'] = order_info['Date'].astype('datetime64[ns]')

# Drop Invoice Date
order_info = order_info.drop(columns=['InvoiceDate'])

# Rename column
order_info = order_info.rename(columns={
    'InvoiceNo': 'invoiceid',
    'StockCode': 'stockcode',
    'Description': 'description',
    'Quantity': 'quantity',
    'UnitPrice': 'price',
    'CustomerID': 'customerid',
    'Country': 'country',
    'Date': 'date'
})

order_info.reset_index(drop=True)

order_info.info()
```

<div id='rfm'/>

## 3. RFM Segmentation
### Create table Customer with their segment table

```python
# Create Revenue col
order_info['revenue'] = order_info['quantity'] * order_info['price']
order_info.head()

customer_rfm = order_info.groupby('customerid').agg({'date':'max',
                                         'invoiceid':'nunique',
                                         'revenue':'sum'}).reset_index()
customer_rfm['date'] = (pd.to_datetime('31/12/2011', format='%d/%m/%Y') - customer_rfm['date']).dt.days
customer_rfm.columns = ['customerid','recency','frequency','monetory']

# rank customer_rfm from 1 to 5
customer_rfm['r_rank'] = 6 - (pd.qcut(customer_rfm['recency'].rank(method='first'), 5, labels=False) + 1)
customer_rfm['f_rank'] = pd.qcut(customer_rfm['frequency'].rank(method='first'), 5, labels=False) + 1
customer_rfm['m_rank'] = pd.qcut(customer_rfm['monetory'].rank(method='first'), 5, labels=False) + 1

# customer_rfm_score
customer_rfm['score'] = customer_rfm['r_rank'].astype(str) + customer_rfm['f_rank'].astype(str) + customer_rfm['m_rank'].astype(str)

# merge with segmentation
customer_rfm = segmentation.merge(customer_rfm, on ='score', how = 'right')
customer_rfm
```
### Create RFM segment table

```python
rfm = customer_rfm.groupby('segment').agg(
    num_ctm=('customerid', 'count'),
    avg_r=('recency', 'mean'),
    avg_f=('frequency', 'mean'),
    sum_rvn=('monetory', 'sum')
).round(1).reset_index()

rfm
```

<div id='vis'/>

## 4. Data Visualization
### Prepare for visualization

```python
rfm['%_ctm'] = round((rfm['num_ctm'] / rfm['num_ctm'].sum()) * 100, 1)
rfm['%_rvn'] = round((rfm['sum_rvn'] / rfm['sum_rvn'].sum()) * 100, 1)
color = sns.color_palette("RdYlGn_r", len(rfm))
```
### Visualization for Avg Recency, Avg Frequency, and % Revenue by Segment

```python
columns = ['avg_r', 'avg_f', '%_rvn']
titles = ['Average Recency by Segment', 'Average Frequency by Segment', '% Revenue by Segment']
y_labels = ['Day(s)', 'Order Time(s)', '% Revenue']

for i, col in enumerate(columns):
    rfm = rfm.sort_values(by=col, ascending=True if i == 0 else False)
    plt.figure(figsize=(8, 5))
    plt.title(titles[i])
    bars = plt.bar(rfm['segment'], rfm[col], alpha=0.7)
    sns.barplot(x=rfm['segment'], y=rfm[col], palette=color, legend = False)
    plt.bar_label(bars, label_type="center")
    plt.xlabel('Segment')
    plt.ylabel(y_labels[i])
    plt.xticks(rotation=45)
    plt.show()
```
![image](https://github.com/user-attachments/assets/46ddd925-ff0b-4b87-89dc-ae182d2e84e2)

![image](https://github.com/user-attachments/assets/a7455b66-1957-4175-9093-1813f7b01468)

![image](https://github.com/user-attachments/assets/cd163ae7-5120-45de-bfc9-56dfff653b88)

### Visualization for % Customer by Segment

```python
rfm = rfm.sort_values(by='%_ctm', ascending=False)
plt.figure(figsize=(16, 6))
labels_with_values = [f"{segment}\n{ctm:.1f}%" for segment, ctm in zip(rfm['segment'], rfm['%_ctm'])]
squarify.plot(sizes=rfm['%_ctm'], label=labels_with_values, color=color, alpha=0.8)
plt.title('% of Customers by Segment')
plt.axis('off')
plt.show()
```

![image](https://github.com/user-attachments/assets/4c04dda8-469a-4efe-a1b8-4f2945a303e7)

<div id='insight'/>

## 5. Insights and Recommendations
