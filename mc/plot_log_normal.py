#! /usr/bin/env python3

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import util

parser = argparse.ArgumentParser(description='plot log-normal"')
parser.add_argument('mean', type=float, help='mean (of the non-logarithmized random variable)')
parser.add_argument('variance', type=float, help='variance (of the non-logarithmized random variable)')

arguments = parser.parse_args(sys.argv[1:])

plt.ion()

mu_log_normal, var_log_normal = arguments.mean, arguments.variance

mu, var = util.normal_parameters_from_lognormal(mu_log_normal, var_log_normal)

samples = np.random.lognormal(mu, np.sqrt(var), 1000)

count, bins, ignored = plt.hist(samples, 100, normed=True, align='mid')

plt.show()
plt.savefig('log_normal_mu_{}_var_{}.pdf'.format(mu_log_normal, var_log_normal))

import code
code.interact(local=dict(globals(), **locals()))