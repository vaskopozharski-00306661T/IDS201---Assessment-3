# coding: utf-8
# Vaskopozharski-00306661T / IDS201---Assessment-3  


#  load the order data into a Pandas DataFrame and convert our input data into appropriate data types:
import pandas as pd
import re
from datetime import datetime
 
# data from http://www.tasteireland.com.au
data_path = 'tasteireland.txt'
 
orders_list = []
with open(data_path) as f:
    for line in f:
        inner_list = [
            line.strip()
            for line in re.split('\s+', line.strip())
        ]
        orders_list.append(inner_list)
 
orders = pd.DataFrame(
    orders_list,
    columns = ['id', 'date', 'orders', 'spend'])
 
orders['date'] = pd.to_datetime(orders['date'])
orders['spend'] = orders['spend'].astype(float)
orders = orders[orders['spend'] > 0]
orders.head()

#compare their first and last order dates in order to calculate customer age. We use months to measure time because it’s our expectation for the typical interval between purchases.

from datetime import timedelta
from numpy import ceil, maximum
 
group_by_customer = orders.groupby(
    by = orders['id'],
    as_index = False)
customers = group_by_customer['date'] \
    .agg(lambda x: (x.max() - x.min()))
 
customers['age'] = maximum(customers['date'] \
    .apply(lambda x: ceil(x.days / 30)), 1.0)
customers = customers.drop(columns = 'date')
customers.head()

twelve_weeks = timedelta(weeks = 12)
cutoff_date = orders['date'].max()
 
dead = group_by_customer['date'].max()['date'] \
    .apply(lambda x: (cutoff_date - x) > twelve_weeks)
 
churn = dead.sum() / customers['age'].sum()
spend = orders['spend'].sum() / customers['age'].sum()
 
clv_aa = spend/churn 
print(clv_aa)

customers_ac = customers.merge(
    group_by_customer['spend'].sum(),
    on = 'id')
 
customers_ac['clv'] = customers_ac['spend'] / customers_ac['age'] / churn
customers_ac.head()

from lifetimes.utils import summary_data_from_transaction_data
 
data = summary_data_from_transaction_data(
    orders, 'id', 'date', 
    monetary_value_col='spend',
    observation_period_end = cutoff_date)
data.head()

from lifetimes import BetaGeoFitter
 
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(data['frequency'], data['recency'], data['T'])

future_horizon = 10000
data['predicted_purchases'] = bgf.predict(
    future_horizon,
    data['frequency'],
    data['recency'],
    data['T'])
data.head()

from lifetimes import GammaGammaFitter
 
returning_customers_summary = data[data['frequency'] > 0]
 
ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(
    returning_customers_summary['frequency'],
    returning_customers_summary['monetary_value'])
transaction_spend = ggf.conditional_expected_average_profit(
    data['frequency'],
    data['monetary_value']
).mean()
print(transaction_spend)

customers_pm = customers_ac.join(
    data['predicted_purchases'],
    on = 'id',
    how = 'left'
).drop(columns = 'clv')
 
customers_pm['clv'] = customers_pm \
    .apply(
        lambda x: x['predicted_purchases'] * transaction_spend,
        axis = 1)
customers_pm.tail()

