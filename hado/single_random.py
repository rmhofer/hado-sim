#single_random.py
import math
import design
__pname__ = 'Single random (no DE)'

def optimize(ado, **kwargs):
	alpha = kwargs.get('alpha', 1.0)
	ado.results.d_evaluated.append(1)
	while(True):
		d = design.design(size=(ado.N_categories, ado.N_features), alpha=alpha)
		u = ado.global_utility(d)
		if not math.isnan(u):
			break
	return d, u