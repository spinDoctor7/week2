import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import os

import math
import numpy as np
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso


import matplotlib.pyplot as plt
from matplotlib import style

import matplotlib as mpl
mpl.rc('figure', figsize=(8,7))
mpl.__version__



start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2019, 9, 9)
df = web.DataReader("AAPL", 'yahoo', start, end)

close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()

rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')

dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'], 'yahoo',start=start,end=end)['Adj Close']

dfcomp.head()
retscomp = dfcomp.pct_change()
corr = retscomp.corr()

plt.scatter(retscomp.AAPL, retscomp.GE)
plt.xlabel('Returns AAPL')
plt.ylabel('Returns GE')

pd.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));

plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);

plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y, in zip(retscomp.columns, retscomp.mean(),
retscomp.std()):
    plt.annotate(
        label,
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

dfreg.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(dfreg)))
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(dfreg['label'])
y = y[:-forecast_out]

clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X, y)
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X, y)
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X, y)
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X, y)

clflasso = Lasso(alpha=0.1)
clflasso.fit(X, y)

confidencereg = clfreg.score(X, y)
confidencepoly2 = clfpoly2.score(X, y)
confidencepoly3 = clfpoly3.score(X, y)
confidenceknn = clfknn.score(X, y)
confidencelasso = clflasso.score(X, y)

print(confidencereg)
print(confidencepoly2)
print(confidencepoly3)
print(confidenceknn)
print(confidencelasso)

forecast_set = clfreg.predict(X_lately)

forecast_set_lasso = clflasso.predict(X_lately)

dfreg['Forecast'] = np.nan

print(forecast_set)
print(forecast_set_lasso)

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix +=datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
