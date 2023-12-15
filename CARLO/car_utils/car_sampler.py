import numpy as np

def sample_lane_utilities(prior=None, return_prob=False):
	idx = np.arange(4)
	if prior is not None:
		sampled_lane = np.random.choice(idx, p=prior)
		prob = prior[sampled_lane]
	else:
		sampled_lane = np.random.choice(idx)
		prob = 1 / 16
	if return_prob:
		return sampled_lane, prob
	else:
		return sampled_lane