#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import util

plt.ion()

# mu_log_normal, var_log_normal = 1, 0.25
# mu_log_normal, var_log_normal = 2e-5, 2e-7
mu_log_normal, var_log_normal = 1, 6

mu, var = util.normal_parameters_from_lognormal(mu_log_normal, var_log_normal)

samples = np.random.lognormal(mu, np.sqrt(var), 1000)

count, bins, ignored = plt.hist(samples, 100, normed=True, align='mid')

plt.show()
plt.savefig('log_normal_mu_{}_var_{}.pdf'.format(mu_log_normal, var_log_normal))

import code
code.interact(local=dict(globals(), **locals()))