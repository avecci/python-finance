# Function that takes start date, end date, stocks and allocations as parameters and returns expected return, volatility and Sharpe ratio for the portfolio.
# Requires Python 2.7 and libraries below.
# Code is based on examples from Computational Investing, Part 1 by Tucker Balck and on Python for Finance by Yves Hilpisch.

import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data,wb
import matplotlib.pyplot as plt
%matplotlib inline

def market_func(start_date, end_date, symbols, allocations):
    data = pd.DataFrame()
# We are only interested in close prices, OHLC curve is unnecessary though doable
    for sym in symbols:
        data[sym] = pdr.DataReader(sym, data_source='yahoo', start=start_date, end=end_date)['Adj Close']
    data.colums = symbols

    (data / data.ix[0] * 100).plot(figsize=(10,5))

# Calculate mean returns and mean covariances:
    rets = np.log(data/data.shift(1))

    #print rets.mean()*252
    #print rets.cov()*252

#Expected portfolio return is weight times return for a stock:
    print "Expected return:"
    print np.sum(rets.mean()*np.array(allocations))*252

#Expected portfolio variance is dot product of weights and covariance:
    print "Expected variance:"
    print np.dot(np.array(allocations).T, np.dot(rets.cov()*252, np.array(allocations)))

    print "Expected standard deviation aka volatility:"
    print np.sqrt(np.dot(np.array(allocations).T, np.dot(rets.cov()*252, np.array(allocations))))

    prets = []
    pvols = []

    for p in range(3000):
        allocations = np.random.random(len(symbols))
        allocations /= np.sum(allocations)
        prets.append(np.sum(rets.mean()*allocations)*252)
        pvols.append(np.sqrt(np.dot(allocations.T, np.dot(rets.cov()*252, allocations))))

    prets = np.array(prets)
    pvols = np.array(pvols)
    print "Sharpe ratio:"
    print sum(prets)/sum(pvols)
    
    %pylab
    plt.figure(figsize(20,10))
    plt.scatter(pvols, prets, c=prets/pvols, marker='o', label='Apple, Microsoft, Yahoo, Deutsche Bank and Gold stock returns')
    plt.grid(True)
    plt.xlabel('Expected volatility $\sigma_p$', fontsize=14)
    plt.ylabel('Expected return $\mu_p$', fontsize=14)
    plt.colorbar(label='Sharpe ratio')
    plt.legend()
    plt.show()

#Parameters:
    #Get stock returns of Apple, Microsoft, Yahoo!, Deutsche Bank and Gold (as commodity):
    #symbols = ['AAPL', 'MSFT', 'YHOO', 'DB', 'GLD']
    #end_date = '2017-02-28'

#Investor is not allowed to set up short positions in a security. 100 % of investors wealth is divided among assets.
#Here we buy equal amount of all five assets, i.e. weight for each is 20 %.
    #allocations = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

print market_func('2010-01-01', '2016-12-31', ['AAPL', 'MSFT', 'YHOO', 'DB', 'GLD'], [0.1, 0.2, 0.3, 0.4, 0.5])