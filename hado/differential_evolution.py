#differential_evolution.py
from __future__ import division
import numpy as np
import support as sp
from multiprocessing import Process, Queue
import design
__pname__ = 'Differential Evolution'

def optimize(ado, **kwargs):
	D = (ado.N_features + 1) * ado.N_categories - 1
	max_G = kwargs.get('n_generations', 4)
	N_particles = int(kwargs.get('n_designs', 1000)/max_G)
	ado.results.d_evaluated.append(N_particles*max_G)
	F = kwargs.get('F', 0.6)
	Cr = kwargs.get('Cr', 0.50)
	alpha = kwargs.get('alpha', 1.0)
	parallelize = kwargs.get('parallelize', True)
	N_threads = kwargs.get('N_threads', 4)


	def utility(x_init):
		x = np.copy(x_init)
		penalty = 0
		for i, p in enumerate(x): #penalize when 
			x[i] = min(1.0, x[i]) #out of bounds
			x[i] = max(0.0, x[i])
			penalty += abs(x[i] - x_init[i])

		prior = x[:ado.N_categories-1]
		prior_sum = np.sum(prior)
		if (prior_sum - 1.0) > 0.0:
			for i, _ in enumerate(prior):
				x[i] /= prior_sum
			penalty += abs(prior_sum - 1.0)

		likelihood = x[ado.N_categories-1:]
		d = design.design(pr=prior, lk=np.reshape(
			likelihood, (ado.N_features, ado.N_categories)))
		utility = ado.global_utility(d) - 1000 * penalty
		return d, utility

	def min_max(x):
		for i, p in enumerate(x):
			x[i] = min(1.0, x[i])
			x[i] = max(0.0, x[i])
		return x

	def evolve(particles, utilities, startIDX=0, endIDX=N_particles):
		#(2) Mutate
		amount_replaced = 0
		for i in np.arange(startIDX, endIDX):
			trial_vector = np.copy(particles[i])
			idx = []
			while True:
				# idx = np.random.choice(N_particles, 3, replace=False)
				idx = sp.choice(N_particles, n=3, replace=False)
				if not i in idx: break
			donor_vector = np.array([particles[idx[0]][j] + F * (
				particles[idx[1]][j] - particles[idx[2]][j]) 
					for j in np.arange(D)])
			donor_vector = min_max(donor_vector)

			j_rand = sp.choice(D)[0]
			# j_rand = np.random.choice(D, 1)[0]
			#(3) Cross-Over
			for j in np.arange(D):
				if np.random.rand() <= Cr or j==j_rand:
					trial_vector[j] = donor_vector[j]
			trial_utility = utility(trial_vector)[1]

			# print trial_utility
			if trial_utility >= utilities[i]:
				particles[i] = trial_vector
				utilities[i] = trial_utility
				amount_replaced += 1
		# print "## Relative amount replaced: %.2f" % (amount_replaced/(endIDX-startIDX))
		# print "## Best particle: %.2f" % np.max(utilities)
		return particles[startIDX:endIDX], utilities[startIDX:endIDX]

	def initialize(particles, utilities):
		np.random.seed()
		for i, particle in enumerate(particles):
			while True:
				particles[i][:ado.N_categories-1] = np.random.dirichlet(
					np.ones(ado.N_categories)*alpha, size=1)[0][:-1]
				utilities[i] = utility(particles[i])[1]
				if np.isnan(utilities[i]) == False:
					break
				else:
					particles[i] = np.random.uniform(low=0.0, 
						high=1.0, size=D)
		return particles, utilities

	class MultiProcess(Process):
		def __init__(self, start, end, particles, utilities, initial=False):
			Process.__init__(self)
			self.startIDX = start
			self.endIDX = end
			self.initialFlag = initial
			self.particles = particles
			self.utilities = utilities
			self.N_local_part = end - start
			self.queueP = Queue()
			self.queueU = Queue()

		def run(self):
			if self.initialFlag:
				pp, uu = initialize(particles[self.startIDX:self.endIDX], 
					utilities[self.startIDX:self.endIDX])
			else:
				pp, uu = evolve(self.particles, self.utilities, 
					self.startIDX, self.endIDX)
			for i in np.arange(self.N_local_part):
				for j in np.arange(D): 
					self.queueP.put(pp[i][j]) 
				self.queueU.put(uu[i])

	# (1) Initialize random parameter vectors (genomes)
	particles = np.random.uniform(low=0.0, high=1.0, 
		size=(N_particles, D))
	utilities = np.zeros(N_particles)

	if parallelize:
		#data chunking
		startIDXC = [i*int(N_particles / N_threads
			) for i in np.arange(N_threads)]
		endIDXC = [(i+1)*int(N_particles / N_threads
			) for i in np.arange(N_threads)]
		endIDXC[-1] = N_particles
	
		for generation in np.arange(max_G+1):
			# print "\t\t...starting generation: %u/%u" % (generation+1, max_G)
			jobs = [MultiProcess(s, t, particles, utilities,
				initial=(generation==0)) for s, t in zip(startIDXC, endIDXC)]
			for job in jobs: job.start()
			for job in jobs: job.join()

			newU = []; newP = []
			for job in jobs:
				while not job.queueU.empty(): newU.append(job.queueU.get())
				while not job.queueP.empty(): newP.append(job.queueP.get())
			particles = np.array(newP).reshape((N_particles, D))
			utilities = np.array(newU)
			# print np.nanmax(utilities)
	else:
		particles, utilities = initialize(particles, utilities)
		for i in np.arange(max_G):
			particles, utilities = evolve(particles, utilities)

	#return best design
	idx_best_d = np.nanargmax(utilities)
	#TODO: what if all are NAN??
	x = particles[idx_best_d]
	d, u = utility(x)
	return d, u	
