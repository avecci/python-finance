#Value-at-Risk simulation using geometric Brownian motion and jump diffusion
# Adapted from Python for Finance by Yves Hilpisch (O'Reilly). Copyright 2015 Yves Hilpisch, 978-1-491-94528-5.

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
from scipy.stats import norm

import pygal
from pygal.style import DefaultStyle
 
from pygal import Config

##### Geometric Brownian motion
S0 		= 100
r 		= 0.05
sigma	= 0.25
T 		= 30 / 365
I 		= 1000
ST 		= S0* np.exp((r - 0.5 * sigma**2)*T + sigma * np.sqrt(T) * npr.standard_normal(I))
R_gbm 	= np.sort(ST-S0)


# Plot distribution plot for GBM:
hist_seaborn = sns.distplot(R_gbm, bins=50, fit=norm, color="Red")
hist_seaborn.set(ylabel="Density", xlabel="Absolute Return")
hist_seaborn.figure.savefig('var1.png')
plt.close()

percs = [0.01,0.1,1.0,2.5,5.0,10.0]
var = scs.scoreatpercentile(R_gbm, percs)
print ("%16s  %16s" % ('Confidence level', 'Value-at-Risk'))
print (33 * "-")

# Print VaR values and probabilities
for pair in zip (percs, var):
	print ("%16.2f %16.3f" % (100- pair[0], -pair[1]))


##### Jump diffusion
M 		= 50
lamb	= 0.75
mu 		= -0.6
delta	= 0.25
dt 		= 30/365/M
rj 		= lamb * (np.exp(mu + 0.5*delta**2)-1)
S 		= np.zeros((M + 1, I))
S[0]	= S0

sn1 	= npr.standard_normal((M + 1, I))
sn2 	= npr.standard_normal((M + 1, I))
poi 	= npr.poisson(lamb*dt, (M+1, I))

for t in range(1, M + 1, 1):
	S[t] = S[t-1] * (np.exp((r - rj - 0.5*sigma**2)*dt + sigma * np.sqrt(dt) * sn1[t]) + (np.exp(mu+delta*sn2[t]) - 1) * poi[t])
	S[t] = np.maximum(S[t],0)

R_jd = np.sort(S[-1] - S0)

# Plot distribution plot JD:
hist_seaborn = sns.distplot(R_jd, bins=50, fit=norm, color="Blue")
hist_seaborn.set(ylabel="Density", xlabel="Absolute Return")
hist_seaborn.figure.savefig('var2.png')
plt.close()

# Plotting JD and GBM in same figure
percs 	= [0.01, 0.1, 1.0, 2.5, 5.0, 10.0]
var 	= scs.scoreatpercentile(R_jd, percs)

# Print VaR values and probabilities
print ("%16s %16s" % ('Confidence level', 'Value-at-Risk'))
print (33 * "-")

for pair in zip(percs,var):
	print ("%16.2f %16.3f" % (100 - pair[0], -pair[1]))

percs 	= list(np.arange(0, 10.1, 0.1))
gbm_var = scs.scoreatpercentile(R_gbm, percs)
jd_var 	= scs.scoreatpercentile(R_jd, percs)


# Plot GBM and JD values in same figure
#plt.plot(percs, gbm_var, 'b', lw=1.5, label='Geometric Brownian Motion', color="Red")
#plt.plot(percs, jd_var, 'r', lw=1.5, label='Jump Diffusion', color="Blue")
#plt.legend(loc=4)
#plt.xlabel('100 - confidence level [%]')
#plt.ylabel('Value-At-Risk')
#plt.grid(True)
#plt.ylim(ymax=0.0)
#plt.savefig('var_total.png')


line_chart			= pygal.Line(human_readable=True, style=DefaultStyle)
line_chart.y_title	= 'Value-At-Risk for geometric Brownian motion and jump diffusion'
line_chart.x_title	= "100 - confidence level [%]"
line_chart.y_title	= "Value-A-Risk"
line_chart.x_labels	= percs
line_chart.add('Geometric Brownian Motion', gbm_var)
line_chart.add('Jump Diffusion',  jd_var)
line_chart.show_legend=True
line_chart.render_to_file('var_total.svg')
