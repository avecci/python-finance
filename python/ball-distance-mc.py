import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

# Simulating a ball throw distance assuming constant acceleration and a flight time of 1.0 s.
# s = v_0*t + 0.5*g*t**2

g = 9.81
t = 1.0

v0 = npr.rand(10) #Pick 10 random values from open interval [0,1).

s = v0*t + 0.5*g*t**2 #Flight distance formula

fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3)

ax1.hist(v0, bins=10)
ax1.set_title('Initial velocity')
ax1.set_ylabel('Frequency')

ax2.plot(s)
ax2.set_title('Flight distance')
ax2.set_ylabel('Distance s')

ax3.plot(v0,s)
ax3.set_title('(v0, s plot))')
ax3.set_ylabel('Distance s')
ax3.set_xlabel('Initial velocity v0')

print (v0)
print (s)
plt.show()
