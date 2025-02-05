

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
file1 = 'D:\\Studies\\Project\\Resources\\FDI_in_India.csv'
file2 = 'D:\\Studies\\Project\\Resources\\dipp.csv'


dataframe = pd.read_csv(file1, index_col=0)
dataframe.dropna(thresh = 2, inplace=True)
# dataframe.isnull().sum()

dataframe.head()

dataframe2 = pd.read_csv(file2,index_col=0)
dataframe2.head()

"""## Preprocessing the dataframes and joining"""

data_sum = dataframe.copy()
data_sum['Sum_upto_16-17'] = dataframe.sum(axis=1)
dataframe2 = dataframe2[['AMOUNT_IN_USD']]
data_join = data_sum.join(dataframe2,how='outer')
data_join

"""### The first dataframe has data upto 2017 and the second has data upto 2019. So we use these two dataframes to get data from 2017-19 by subtracting 'Sum_upto_16-17' from 'AMOUNT_IN_USD'. Assuming a third od that investment came in the financial year 2017-18 and the rest two third in the financial year 2018-19"""

dataframe['2017-18'] = (2/3)*(data_join['AMOUNT_IN_USD'] - data_join['Sum_upto_16-17'])
dataframe['2018-19'] = (data_join['AMOUNT_IN_USD'] - data_join['Sum_upto_16-17'])/(3)
dataframe.fillna(0,inplace=True)
dataframe

"""## Transforming the dataset"""

df = dataframe.transpose()
df.index = [x[1] for x in df.index.str.split('-')]
new = str('20') + df.index
new

df = dataframe.transpose()
df.index = [x[1] for x in df.index.str.split('-')]
df.index = str('20') + df.index
df.index = pd.to_datetime(df.index)
print(type(df.index))
df.tail()

"""# FDI in various sectors of Indian economy since 2000"""

def plot_data(data=df,x='Year',y='Amount is million USD',t='FDI IN ALL SECTORS OF INDIAN ECONOMY'):
    plt.figure(figsize=(8,8))
    plt.plot(data)
#     plt.legend(data.columns)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(t)
    plt.show()

plot_data()

"""#### This is a very untidy interpretation of the data. Let us take the top 10 sectors which are most invested in since 2000.

# Analysing the current top 5 sectors which are the most invested in.
"""

def avg_sector(df=dataframe,number=5):
    data = df.copy(deep=True)
    data['avg'] = pd.Series()
    for sector in data.index:
        data['avg'] = data.mean(axis=1)
        top_df = data.sort_values(by='avg',ascending=False,inplace=False)
    top = top_df.head(number).index
    top_df.drop(columns=['avg'],axis=1,inplace=True)
    return top


avg_sector(dataframe)

top_sectors = avg_sector()
df_top = df[top_sectors]
plot_data(data=df_top, x= 'Year', y= 'Amount is million USD',t= 'FDI IN TOP 5 SECTORS OF THE COUNTRY')

"""### These are the top 5 sectors which have received the most foreign investments in the past 19 years. It can be seen that most of these sectors have an 100%+ increment in investments. Few have remained stagnant while some industries like Construction Development has gone down.

# Does government have anything to do with FDIs?

## Analysing how FDIs have changed pre and post Modi government
"""

pre = df.loc[:'2013']
post = df.loc['2014':]

top_pre = avg_sector(pre.transpose())
top_post = avg_sector(post.transpose())

pre = pre[top_pre]
pre.loc['Total',:] = pre.sum(axis=0)
# print(pre.loc['Total'])

post = post[top_post]
post.loc['Total',:] = post.sum(axis=0)
# print('\n')
# print(post.loc['Total'])


plt.figure(figsize=(20,10))
sns.barplot(x=pre.columns,y= pre.loc['Total'])
plt.xticks(np.arange(0,5,1),labels=['SERVICES SECTOR','CONSTRUCTION DEVELOPMENT','TELECOMMUNICATIONS','COMPUTER SOFTWARE & HARDWARE','DRUGS PHARMACEUTICALS'], rotation=75)
plt.title('FDIs in different sectors before 2014')
plt.ylabel('Total amount in million USD')
plt.plot()

plt.figure(figsize=(20,10))
sns.barplot(x=post.columns,y= post.loc['Total'])
plt.xticks(np.arange(0,5,1),labels=['SERVICES SECTOR','COMPUTER SOFTWARE & HARDWARE','TELECOMMUNICATIONS','TRADING ','CONSTRUCTION (INFRASTRUCTURE) ACTIVITIES '], rotation=75)
plt.title('FDIs in different sectors after 2014')
plt.ylabel('Total amount in million USD')
plt.plot()

"""### Computer Software & Hardware and Construction infrastructure activity sectors has gone up in FDIs replacing Construction development and Drugs and Pharamceuticals sector since BJP has taken over.

# Simple Linear Regression on the sectors
"""

new = df[['SERVICES SECTOR (Fin.,Banking,Insurance,Non Fin/Business,Outsourcing,R&D,Courier,Tech. Testing and Analysis, Other)']]
new.rename(columns={'SERVICES SECTOR (Fin.,Banking,Insurance,Non Fin/Business,Outsourcing,R&D,Courier,Tech. Testing and Analysis, Other)':'Services'}, inplace=True)
new.head()

new.plot()
plt.show()

result = adfuller(new['Services'])

print('Test statistic:'+str(result[0]))
print('p value:'+str(result[1]))

new_diff = new.diff(1).dropna(inplace=False)
plt.figure(figsize=(20,10))
new_diff.plot()
plt.show()

result = adfuller(new_diff['Services'])

print('Test statistic:'+ str(result[0]))

print('p value:'+str(result[1]))

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))

plot_acf(new_diff, lags=10, zero=False, ax=ax1)

plot_pacf(new_diff, lags=10, zero=False, ax=ax2)

plt.show()

for p in range(4):
    for q in range(3):
      try:
        model = SARIMAX(new, order=(p,1,q), trend='c')
        results = model.fit()

        print(p, q, results.aic, results.bic)

      except:
        print(p, q, None, None)

model = SARIMAX(new, order=(0,1,2), trend='c')
results = model.fit()

results.plot_diagnostics()
plt.show()

print(results.summary())

one_step_forecast = results.get_prediction(start=-10)

mean_forecast = one_step_forecast.predicted_mean

confidence_intervals = one_step_forecast.conf_int()
# confidence_intervals

lower_limits = confidence_intervals.loc[:,'lower Services']
upper_limits = confidence_intervals.loc[:,'upper Services']

print(mean_forecast)
# one_step_forecast
# new.tail()

plt.plot(new.index, new, label='observed')

plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

plt.fill_between(lower_limits.index,lower_limits, upper_limits, color='pink')

plt.xlabel('Date')
plt.ylabel('FDI in Services Sector')
plt.legend()
plt.show()

dynamic_forecast = results.get_forecast(steps=5)

mean_forecast = dynamic_forecast.predicted_mean

confidence_intervals= dynamic_forecast.conf_int()

lower_limits = confidence_intervals.loc[:,'lower Services']
upper_limits = confidence_intervals.loc[:,'upper Services']


print(mean_forecast)

plt.plot(new.index, new, label='observed')

plt.plot(mean_forecast.index, mean_forecast.values, color='r', label='forecast')

plt.fill_between(mean_forecast.index, lower_limits, upper_limits, color='pink')

plt.xlabel('Year')
plt.ylabel('FDI in Services Sector')
plt.legend()
plt.show()

forcast= results.forecast(1)
forcast
