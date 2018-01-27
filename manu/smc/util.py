import numpy as np


# the logarithm of a bunch of real numbers given its individual logarithms
def log_sum_from_individual_logs(logs):

	descending_sort = np.sort(logs)[::-1]

	return descending_sort[0] + np.log1p(np.exp(descending_sort[1:] - descending_sort[0]).sum())


def normalize_from_logs(logs):

	log_sum = log_sum_from_individual_logs(logs)

	return np.exp(logs - log_sum)
