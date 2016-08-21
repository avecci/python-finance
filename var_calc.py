import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as scs
import collections
import pygal

S0 = 100
r = 0.05
sigma = 0.25
T = 30 / 365
I = 1000
ST = S0* np.exp((r - 0.5 * sigma**2)*T + sigma * np.sqrt(T) * npr.standard_normal(I))

R_gbm = np.sort(ST-S0)

testihist = np.histogram(R_gbm,bins=100)

print(testihist) #Palauttaa kaksi arrayta jarj y,x
hist_pygal = pygal.Histogram()
#hist.add('absolute return', [(hist,0,10)])
hist_pygal.add('Absolute return',  ([(1, R_gbm[0],R_gbm[1]),(2, R_gbm[1],R_gbm[2])]))
hist_pygal.render_to_file('var1.svg')

plt.hist(R_gbm, bins=100)
plt.xlabel('absolute return')
plt.ylabel('frequency')
plt.grid(True)
 
#plt.show()
plt.savefig('var1.png');

percs = [0.01,0.1,1.0,2.5,5.0,10.0]
var = scs.scoreatpercentile(R_gbm, percs)
print ("%16s  %16s" % ('Confidence level', 'Value-at-Risk'))

print (33 * "-")

for pair in zip (percs, var):
	print ("%16.2f %16.3f" % (100- pair[0], -pair[1]))


M = 50
lamb = 0.75
mu = -0.6
delta = 0.25
dt = 30/365/M

rj = lamb*(np.exp(mu+0.5*delta**2)-1)
S = np.zeros((M+1,I))
S[0] = S0

sn1 = npr.standard_normal((M+1,I))
sn2 = npr.standard_normal((M+1,I))
poi = npr.poisson(lamb*dt, (M+1, I))

for t in range(1, M + 1, 1):
	S[t] = S[t-1] * (np.exp((r - rj - 0.5*sigma**2)*dt + sigma * np.sqrt(dt) * sn1[t]) + (np.exp(mu+delta*sn2[t]) - 1) * poi[t])
	S[t] = np.maximum(S[t],0)

R_jd = np.sort(S[-1]-S0)

plt.hist(R_jd,bins=100)
plt.xlabel('absolute return')
plt.ylabel('frequency')
plt.grid(True)
plt.savefig('var2.png')