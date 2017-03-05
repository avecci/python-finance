# Modelling CAPM and calculating alpha and beta from historical values.
import pandas_datareader as pdr
from pandas_datareader import data, wb
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Fetch data for Apple stock prices and for S&P-500 Index from Yahoo Finance:

def capm(start_date, end_date, ticker1, ticker2):
    df = pdr.get_data_yahoo(ticker1, start_date, end_date)
    dfb = pdr.get_data_yahoo(ticker2, start_date, end_date)

# create a time-series
    rts = df.resample('M').last()
    rbts = dfb.resample('M').last()
    dfsm = pd.DataFrame({'s_adjclose' : rts['Adj Close'], 'b_adjclose' : rbts['Adj Close']}, index=rts.index)

# compute returns
    dfsm[['s_returns', 'b_returns']] = dfsm[['s_adjclose','b_adjclose']]/dfsm[['s_adjclose','b_adjclose']].shift(1) -1
    dfsm = dfsm.dropna()

    covmat = np.cov(dfsm["s_returns"], dfsm["b_returns"])

# calculate measures now
    beta = covmat[0,1]/covmat[1,1]
    alpha = np.mean(dfsm["s_returns"])-beta*np.mean(dfsm["b_returns"])

# r_squared     = 1.0 - SS_res/SS_tot
    y = beta * dfsm["b_returns"] + alpha
    SS_res = np.sum(np.power(y - dfsm["s_returns"],2))
    SS_tot = covmat[0,0]*(len(dfsm) - 1) # SS_tot is sample_variance*(n-1)

    r_squared = 1.0 - SS_res/SS_tot
# Volatility for the full time and 1-year momentum
    volatility = np.sqrt(covmat[0,0])
    momentum = np.prod(1+dfsm["s_returns"].tail(12).values) - 1.0

# annualize the numbers
    prd = 12.0 # used monthly returns; 12 periods to annualize
    alpha = alpha*prd
    volatility = volatility*np.sqrt(prd)

    print "Beta, alpha, r_squared, volatility, momentum:"
    print beta, alpha, r_squared, volatility, momentum

    %pylab
    fig,ax = plt.subplots(1,figsize=(20,10))
    ax.scatter(dfsm["b_returns"], dfsm['s_returns'], label="Data points")
    beta,alpha = np.polyfit(dfsm["b_returns"], dfsm['s_returns'], deg=1)
    ax.plot(dfsm["b_returns"], beta*dfsm["b_returns"] + alpha, color='red', label="CAPM line")


    plt.title('Capital Asset Pricing Model, finding alphas and betas')
    plt.xlabel('Market return $R_m$', fontsize=14)
    plt.ylabel('Stock return $R_i$')
    plt.text(0.05, 0.05, r'$R_i = \beta * R_m + \alpha$', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


capm('2010-01-01', '2016-12-31','AAPL', '^GSPC')


