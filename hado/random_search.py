#random_search.py
import numpy as np
from multiprocessing import Process, Queue
import design
__pname__ = 'Random Search'

def optimize(ado, **kwargs):
	N_designs = kwargs.get('n_designs', 100)
	ado.results.d_evaluated.append(N_designs)
	alpha = kwargs.get('alpha', 1.0)
	parallelize = kwargs.get('parallelize', False)
	if parallelize: N_threads = kwargs.get('N_threads', 4)

	def evaluate_N_random(n):
		np.random.seed()
		#randomly generate and evaluate #'n_iterations' designs:
		randomDesigns = [design.design(size=(ado.N_categories, 
			ado.N_features), alpha=alpha) for i in np.arange(n)]

		U = [ado.global_utility(d) for d in randomDesigns]
		
		try:
			idx_best_d = np.nanargmax(U)
		except Exception:
			d = design.design(size=(ado.N_categories, 
				ado.N_features), alpha=alpha)
			return d, ado.global_utility(d)
		else:
			idx_best_d = np.nanargmax(U)
			return randomDesigns[idx_best_d], U[idx_best_d]

	class MultiProcess(Process):
		def __init__(self, n):
			Process.__init__(self)
			self.n = n
			self.queueD = Queue()
			self.queueU = Queue()	

		def run(self):
			d, u = evaluate_N_random(self.n)
			self.queueD.put(d)
			self.queueU.put(u)

	if parallelize:
		split_N = np.array([int(N_designs / N_threads)] * N_threads)
		split_N[0] += N_designs - np.sum(split_N)
		jobs = [MultiProcess(n) for n in split_N]
		for job in jobs: job.start()
		for job in jobs: job.join()
		#find best design among jobs!!
		D = [job.queueD.get() for job in jobs]
		U = [job.queueU.get() for job in jobs]
		
		try:
			idx_best_d = np.nanargmax(U)
		except Exception:
			d = design.design(size=(ado.N_categories, 
				ado.N_features), alpha=alpha)
			return d, ado.global_utility(d)
		else:
			idx_best_d = np.nanargmax(U)
			return D[idx_best_d], U[idx_best_d]
	else:
		return evaluate_N_random(N_designs)
