#smc.py
from __future__ import division
import numpy as np
from scipy.stats import norm
import design
import tqdm
from multiprocessing import Process, Queue
__pname__ = 'Sequential Monte Carlo (SMC)'

#sequential monte carlo
def optimize(ado, **kwargs):

	D = (ado.N_features + 1) * ado.N_categories - 1
	n_designs = kwargs.get('n_designs', 100)
	n_iterations = kwargs.get('n_iterations', 100)
	n_particles = int(kwargs.get('n_iterations', n_designs/n_iterations))
	parallelize = kwargs.get('parallelize', False)
	sigma = kwargs.get('sigma', 0.3)
	alpha = kwargs.get('sigma', 1.0)
	ado.results.d_evaluated.append(n_designs)
	if parallelize: N_threads = kwargs.get('N_threads', 4)

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

		xt = np.empty((niters, n_particles), dtype=np.object)
		ut = np.zeros((niters, n_particles))
		wt = np.zeros((niters, n_particles))
		wt[0,:] = np.ones(n_particles) / n_particles

		for i in np.arange(n_particles):
			while True:
				xt[0,i] = design.design(size=(ado.N_categories, ado.N_features)) #first design is random
				ut[0,i] = ado.global_utility(xt[0,i])
				if not ut[0,i] == 0: break

		iterator = np.arange(start=1, stop=niters)
		# if ado.logging: iterator = tqdm.tqdm(iterator)
		
		#smc
		for t in iterator:
			xt_ = np.empty(n_particles, dtype=np.object)
			ut_ = np.zeros(n_particles)
			wt_ = np.zeros(n_particles)

			#time update
			for i in np.arange(n_particles):
				xt_[i] = proposal_distribution(xt[t-1,i]) #random walk
				#measurement update
				ut_[i] = ado.global_utility(xt_[i])
				wt_[i] = wt[t-1,i] * ut_[i]

			#resample
			idx = np.random.choice(a=n_particles, size=n_particles, p=wt_/np.sum(wt_))
			xt[t,:] = xt_[idx]
			ut[t,:] = ut_[idx]
			wt[t,:] = np.ones(n_particles) / n_particles

		print(np.max(ut))

		# #mcmc
		# for i in iterator:
		# 	# j = J(i+1)
		# 	d_i = proposal_distribution(d) #random walk
		# 	u_i = ado.global_utility(d_i)
		# 	rho = min(1, u_i / u) #**j consider sim. annealing
		# 	if np.random.uniform() < rho:
		# 		naccept += 1 #number accepted
		# 		d = d_i
		# 		u = u_i
		# 		accepted_d.append(d)
		# 		accepted_u.append(u)

		ado.log('\t### SMC sampling finished! ###')
		idx_best_d = np.nanargmax(ut)
		return xt.flatten()[idx_best_d], ut.flatten()[idx_best_d]


	execute(n_iterations)

	d = design.design(size=(ado.N_categories, ado.N_features), alpha=alpha)
	return d, ado.global_utility(d)



'''
### particle filter ###
true_mean = 50
true_sigma = 10
max_iter = 100
N = 10000 #N_particles

accepted_states = np.empty(0)
sigma = 2
xt = np.zeros((max_iter, N))
wt = np.zeros((max_iter, N))
xt[0,:] = np.random.rand(N) * 100 #sample from p0 to obtain initialization
wt[0,:] = np.ones(N) / N

for t in np.arange(start=1, stop=max_iter):
	xt_ = np.zeros(N)
	wt_ = np.zeros(N)

	#time update
	for i in np.arange(N):
		xt_[i] = np.random.normal(xt[t-1,i], sigma)
		#measurement update
		wt_[i] = wt[t-1,i] * norm.pdf(xt_[i], true_mean, true_sigma)

	#resample
	xt[t,:] = np.random.choice(a=xt_, size=N, p=wt_/np.sum(wt_))
	accepted_states = np.concatenate((accepted_states, xt[t,:]))
	wt[t,:] = np.ones(N) / N
'''