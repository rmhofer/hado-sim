# ________________________________
# ________________________________
# only required for numpy v < 1.7
import numpy as np

def choice(choices, n=1, replace=True, p=None):
	if isinstance(choices, int):
		choices = np.arange(choices)
	if p is None:
		p=np.ones(len(choices))/len(choices)

	if replace==False:
		if n>len(choices):
			return "Not enough choices to sample from with replacement"

	def sample():
		total = np.sum(p)
		treshold = np.random.uniform(0, total)
		for k, weight in enumerate(p):
			total -= weight
			if total < treshold:
				return choices[k]

	return_values = []
	if replace==False:
		for i in np.arange(n):
			while(True):
				c = sample()
				if not c in return_values:
					return_values.append(c)
					break
	else:
		return_values = [sample() for i in np.arange(n)]
	return return_values