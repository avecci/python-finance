# Portfolio optimization.

start_date = '2010-01-01'
end_date = '2016-12-31'
symbols = ['AAPL', 'MSFT', 'YHOO', 'DB', 'GLD']

# Start by giving initial values for allocation:
allocations = [0.1, 0.2, 0.3, 0.4, 0.5]

def statistics(allocations):
    rets = np.log(data/data.shift(1))
    allocations = np.array(allocations)
    pret = np.sum(rets.mean()*allocations)*252
    pvol = np.sqrt(np.dot(allocations.T, np.dot(rets.cov()*252, allocations)))
    return np.array([pret, pvol, pret/pvol])

data = pd.DataFrame()
for sym in symbols:
    data[sym] = pdr.DataReader(sym, data_source='yahoo', start=start_date, end=end_date)['Adj Close']
data.colums = symbols

# Calculate mean returns and mean covariances:

import scipy.optimize as sco

def min_func_sharpe(allocations):
    return -statistics(allocations)[2]

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
numofalloc = len(symbols)
bnds = tuple((0,1) for x in range(numofalloc))

# Maximise Sharpe Ratio:
opts = sco.minimize(min_func_sharpe, numofalloc*[1./numofalloc,], method='SLSQP', bounds = bnds, constraints=cons)

print "Allocations for maximised Sharpe ratio:"
print opts['x'].round(3)
allocations = opts['x']
print "Expected returns, expected volatility, Sharpe ratio:"
print statistics(opts['x']).round(3)
# Best result with three stocks: return 23,5 %, Sharpe Ratio 1.06

#Minimize variance:
def min_func_variance(allocations):
    return statistics(allocations)[1]**2

optv = sco.minimize(min_func_variance, numofalloc*[1.0/numofalloc,], method='SLSQP', bounds=bnds, constraints=cons)
print "Allocations for minimised volatility:"
print optv['x'].round(3)
print "Expected returns, expected volatility, Sharpe ratio:"
print statistics(optv['x']).round(3)