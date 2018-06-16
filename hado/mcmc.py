#mcmc.py
from __future__ import division
import numpy as np
from scipy.stats import norm
import design
import tqdm
from multiprocessing import Process, Queue
__pname__ = 'Markov Chain Monte Carlo (MCMC)'

def optimize(ado, **kwargs):
	D = (ado.N_features + 1) * ado.N_categories - 1
	niters = kwargs.get('n_designs', 100)
	parallelize = kwargs.get('parallelize', True)
	sigma = kwargs.get('sigma', 0.4)
	alpha = kwargs.get('sigma', 1.0)
	ado.results.d_evaluated.append(niters)
	if parallelize: N_threads = kwargs.get('N_threads', 4)

	class MultiProcess(Process):
		def __init__(self, iterations):
			Process.__init__(self)
			self.iters = iterations
			self.queueD = Queue()
			self.queueU = Queue()	

		def run(self):
			d, u = execute(niters=self.iters)
			self.queueD.put(d)
			self.queueU.put(u)

	def J(t):
		return int(t * 0.04) + 1

	def norm_probabilities(array):
		for index, _ in np.ndenumerate(array):
			array[index] = min(1.0, array[index])
			array[index] = max(0.0, array[index])
		return array

	def proposal_distribution(d, sigma=sigma):
		new_prior = np.absolute(
			np.add(d.prior, np.random.normal(0, sigma, size=d.prior.shape)))
		new_prior /= np.sum(new_prior)

		while True:
			new_lk = np.add(
				d.likelihoods, np.random.normal(0, sigma, size=d.likelihoods.shape))
			new_lk = norm_probabilities(new_lk)
			if np.argwhere(np.sum(new_lk, axis=1) % 3 == 0).size == 0: break
		return design.design(pr=new_prior[:-1], lk=new_lk)

	def execute(niters):
		
		#assign new random seed
		if parallelize: np.random.seed()

		naccept = 0
		samples = np.zeros((D, niters+1))
		while True:
			d = design.design(size=(ado.N_categories, ado.N_features)) #first design is random
			u = ado.global_utility(d)
			if not u == 0: break
		accepted_u = []
		accepted_d = []

		iterator = range(niters)
		if ado.logging: iterator = tqdm.tqdm(iterator)
		
		for i in iterator:
			# j = J(i+1)
			d_i = proposal_distribution(d) #random walk
			u_i = ado.global_utility(d_i)
			rho = min(1, u_i / u) #**j consider sim. annealing
			if np.random.uniform() < rho:
				naccept += 1 #number accepted
				d = d_i
				u = u_i
				accepted_d.append(d)
				accepted_u.append(u)
		ado.log('\t### MCMC acceptance rate: %.2f ###' % float(1.0 * naccept/niters))
		idx_best_d = np.nanargmax(accepted_u)
		return accepted_d[idx_best_d], accepted_u[idx_best_d]
	
	if parallelize:
		iters = int(niters/N_threads) 
		jobs = [MultiProcess(iterations=iters) for i in np.arange(N_threads)]
		for job in jobs: job.start()
		for job in jobs: job.join()
		D = [job.queueD.get() for job in jobs]
		U = [job.queueU.get() for job in jobs]
		
		try:
			idx_best_d = np.nanargmax(U)
		except Exception:
			d = design.design(size=(ado.N_categories, ado.N_features), alpha=alpha)
			return d, ado.global_utility(d)
		else:
			return D[idx_best_d], U[idx_best_d]
	else:
		return execute(niters)
