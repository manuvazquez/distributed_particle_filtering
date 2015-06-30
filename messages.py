#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

exchange_period = 10

# we deploy as many sensors as needed for every PE to have...
average_number_of_sensors_per_PE = 4

# percentage of its particles a certain PE exchanges
#p = 0.15
p = 0.5

# number of particles
N = 200

# the number of PEs
#M = np.arange(2,18,2)
#M = np.array([2,4,8])
M = np.array([2,3,4,5,6],dtype=int)

# number of sensors
#J = 16
J = M*average_number_of_sensors_per_PE

n_messages_DRNA = 1/exchange_period*(J*M + p*N*M) + (exchange_period-1)/exchange_period*(J*M)

n_messages_Mposterior = 1/exchange_period*(J + p*N*M) + (exchange_period-1)/exchange_period*(J)


fig = plt.figure()
ax = fig.gca()

ax.plot(M,n_messages_DRNA,marker='o',label='DRNA')
ax.plot(M,n_messages_Mposterior,marker='+',label='M-posterior')

ax.legend()

fig.show()

import code
code.interact(local=dict(globals(), **locals()))