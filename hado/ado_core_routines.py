IMPORT_PLOTTING_LIBRARIES = True

#standard libraries (available)
import numpy as np

if IMPORT_PLOTTING_LIBRARIES:
	import matplotlib.pyplot as plt
	import seaborn as sns
	from matplotlib.patches import Rectangle
	import ternary_plot

#python libraries
import time
import datetime
import itertools
from fractions import Fraction

#custom imports
import sharma_mittal
import design
import differential_evolution
import random_search
import single_random
import mcmc
import smc

# ____________________________________________________________________________________________________
# ____________________________________________________________________________________________________

class Ado(object):
	"""
		Class that initialized basic (H)ADO object to run simulations / DO

		Arguments are passed on from function call below.
	"""
	def __init__(self, max_steps, n_trials, 
				 orderSpace, degreeSpace, betaSpace,
				 true_order, true_degree, true_beta,
				 N_categories, N_features, prior, g,
				 method, m_args, logging, plotting,
				 save_results, l_flag, filename):
		self.logging = logging
		self.plotting = plotting
		self.save_results = save_results
		
		self.method = method
		self.m_args = m_args
		self.priorName = prior
		self.l_flag = l_flag
		self.log_string = ''
		self.fileID = np.random.randint(100000,999999)

		self.max_steps = max_steps
		self.orderSpace = orderSpace
		self.degreeSpace = degreeSpace
		self.betaSpace = betaSpace
		# self.labels = [('%s' % Fraction(f)) for f in degreeSpace] #as fractions
		self.labels = ['%.1f' % f for f in degreeSpace]
		self.step = 0
		self.stepwise = False
		self.set_response = -1
		self.simulation = True 
		self.filename = filename

		self.N = len(orderSpace)
		self.N_beta = len(betaSpace)
		self.N_categories = N_categories
		self.N_features = N_features
		self.n_trials = n_trials
		self.g = g

		self.bounds = np.array([np.fliplr(np.rot90(
			np.loadtxt("hado/extrema/katVals%s_%s_numOpt_1e4-%s.csv" % 
				(N_categories, i, str(self.g)), delimiter=',', dtype=float), k=3))
					for i in ['best','worst']])

		if prior=='exp':
			self.prior = np.ones((self.N,self.N,self.N_beta))
			p_tmp = np.flipud(np.array(
				np.loadtxt("hado/priors/hadoPriorsExperienceData-%s.csv" % str(self.g), 
				delimiter=',', dtype=float)))
			for i in np.arange(self.N_beta):
				self.prior[:,:,i] = p_tmp
		elif prior=='rand':
			self.prior = np.random.rand(self.N,self.N,self.N_beta)
		else:
			self.prior = np.ones((self.N,self.N,self.N_beta)) #uniform
		self.prior = self.prior / self.prior.sum()

		prior_margin = np.add.reduce(self.prior, axis=2) #marginal prior (sum across betas)
		prior_margin /= np.sum(prior_margin)
		if true_order == None: #sample from prior if not defined
			order_sample = np.sum(prior_margin, axis=0
				) / np.sum(np.sum(prior_margin, axis=0))
			self.true_order = np.random.choice(self.N, 1, 
				replace=True, p=order_sample)[0]
		else: self.true_order = true_order
		if true_degree == None: #sample from prior if not defined
			degree_sample = np.sum(prior_margin, axis=1
				) / np.sum(np.sum(prior_margin, axis=1))
			self.true_degree = np.random.choice(self.N, 1, 
				replace=True, p=degree_sample)[0]
		else: self.true_degree = true_degree
		if true_beta == None: #sample from prior if not defined
			beta_sample = [np.sum(
				self.prior[:,:,i]) for i in np.arange(self.N_beta)]
			self.true_beta = np.random.choice(self.N_beta, 1, 
				replace=True, p=beta_sample)[0]
		else: self.true_beta = true_beta

		self.marginals = np.zeros(self.n_trials+1)
		self.expinfogain = 0
		self.prior_ent = sharma_mittal.sm_entropy(self.prior, r=1.0, t=1.0)

		#create separate 'results' object
		self.results = self.Results(self)
		self.results.post_entropy_array.append( #initial entropy of prior beliefs
			sharma_mittal.sm_entropy(self.prior, r=1.0, t=1.0))


	class Results(object):
		"""
			This is a separate object in which results from the simulation are stored
		"""
		def __init__(self, parent):
			self.time_array = []

			#Mean Squared Error
			self.mse_order_array = []
			self.mse_degree_array = []
			self.mse_beta_array = []

			#Posterior entropy
			self.ent_order_array = []
			self.ent_degree_array = []
			self.ent_beta_array = []
			self.post_entropy_array = []

			#Variance
			self.order_variance_array = []
			self.degree_variance_array = []
			self.beta_variance_array = []
			self.total_variance_array = []

			self.designs = []
			self.dobject = []
			self.d_evaluated = []
			self.utilities = []
			self.pure_utilities = []
			self.responses = []
			self.lscores = []
			
			#What was inferred?
			self.degree_guesses = []
			self.order_guesses = []
			self.beta_guesses = []

	def log(self, string):
		""" For logging """
		self.log_string += string + '\n'
		if self.logging: print(string)

	def tic(self):
		""" For timing """
		self.trial_start_time = time.time()

	def toc(self):
		""" For timing """
		self.trial_stop_time = time.time()
		self.trial_time = self.trial_stop_time - self.trial_start_time

	def viz_all(self, arr, annotate=False, ax=None, title=False, norm=False, cmap=False):
		""" To create the full visualization """
		if ax == None: f, ax = plt.subplots()
		if cmap==False:
			cmap = sns.cubehelix_palette(8, start=.5, rot=-.75, 
				light=.98, as_cmap=True, reverse=True)
		if norm: 
			vmin=0.0
			vmax=1.0
		else: 
			vmin=None
			vmax=None

		if self.g >= 4: #skip labels
			skiplab = np.copy(self.labels)
			skiplab[::2] = np.array([''] * skiplab[::2].size)
			xticklabels, yticklabels = skiplab, skiplab
		else:
			xticklabels, yticklabels = self.labels, self.labels
		
		ax = sns.heatmap(arr, ax=ax, cbar=False, cmap=cmap,
			xticklabels=xticklabels, yticklabels=yticklabels,
			vmin=vmin, vmax=vmax)

		if not annotate==False and not self.simulation: 
			from matplotlib.font_manager import FontProperties
			font = FontProperties()
			font.set_weight('bold')
			true_color = 'crimson'
			guess_color = 'deeppink'
			
			ax.add_patch(Rectangle((self.true_order, self.true_degree), 0.9, 0.9, 
				fill=False, edgecolor=true_color, lw=3))
			ax.text(self.true_order+0.6, self.true_degree+1.4, 
				r'true $\Theta$', ha='center', va='center',
				color=true_color, fontproperties=font)

			if not annotate == 1 and not self.stepwise:
				idx = annotate-2
				ax.add_patch(Rectangle(
					(self.results.order_guesses[idx], (arr.shape[1] - 1) 
						- self.results.degree_guesses[idx]), 0.9, 0.9, 
					fill=False, edgecolor=guess_color, lw=3))
				ax.text(self.results.order_guesses[idx]-0.65, 
					ax.get_ylim()[1]-self.results.degree_guesses[idx]-1.3, 
					r'inferred $\Theta$', ha='left', va='center',
					color=guess_color, fontproperties=font)

		if not title==False: 
			ax.text(0.5, 1.05, title, 
			ha='center', va='center', 
			transform=ax.transAxes)
		ax.set(xlabel='Order (r)', ylabel='Degree (t)')
		ax.set_aspect('equal')
		ax.invert_yaxis()
		return ax

	def viz_prior(self, prior=[], annotate=False, ax=None, title=False):
		if prior==[]: prior = self.prior
		plot_prior = np.add.reduce(prior, axis=2)
		plot_prior /= np.sum(plot_prior)
		return self.viz_all(plot_prior, 
			annotate=annotate, ax=ax, title=title)	

	def viz_relevance(self, d, q, annotate=False, ax=None, title=False, norm=True, cmap=False):
		sm_rel_score = np.zeros((self.N,self.N))
		for (degree, order), _ in np.ndenumerate(sm_rel_score):
			sm_rel_score[degree][order] = self.sm_usefulness(d, self.degreeSpace[degree], self.orderSpace[order])[q]
		return self.viz_all(sm_rel_score, 
			annotate=annotate, ax=ax, title=title, norm=norm, cmap=cmap)	 

	def viz_choice_prob(self, d, annotate=False, ax=None, title=False, norm=True, inv=False, cmap=False):
		sm_luce_choice = np.zeros((self.N,self.N))
		for (degree, order), _ in np.ndenumerate(sm_luce_choice):
			rel = self.sm_usefulness(d, self.degreeSpace[degree], self.orderSpace[order])
			sm_luce_choice[degree][order] = self.choice_probability(rel, beta=self.betaSpace[self.true_beta])
			if inv: sm_luce_choice[degree][order] = 1.0-sm_luce_choice[degree][order]
			
		return self.viz_all(sm_luce_choice, 
			annotate=annotate, ax=ax, title=title, norm=norm, cmap=cmap)	

	# def viz_like(self, d, annotate=False, ax=None, title=False):

	# 	red_lik = np.add.reduce(self.likelihood_function(d), axis=2)
	# 	red_lik /= np.sum(red_lik)
	# 	return self.viz_all(red_lik[:,:,0], annotate=annotate, ax=ax, title=title)	

	def viz_triplet(self, old_prior, d):
		f, axarr = plt.subplots(1,3,figsize=(16,5))
		prior_ent = sharma_mittal.sm_entropy(old_prior, r=1.0, t=1.0)
		post_ent = sharma_mittal.sm_entropy(self.prior, r=1.0, t=1.0)
		
		axarr[0] = self.viz_prior(old_prior, ax=axarr[0], 
			title=r'Prior $\quad P(\Theta) \qquad ent^{SM_{(1,1)}}(P)=%.2f$' % prior_ent,
			annotate=self.step)
		axarr[1] = self.viz_prior(ax=axarr[1],
			title=r'Posterior $\quad P(\Theta|d=d^{*},y=%d) \qquad ent^{SM_{(1,1)}}(P)=%.2f$' % (
				self.results.responses[-1], post_ent),
			annotate=self.step+1)
		# axarr[2] = ternary_plot.plot(d, 
		# 	t=self.degreeSpace[int(self.guess_degree)], 
		# 	r=self.orderSpace[int(self.guess_order)], 
		# 	ax=axarr[2], title=r'Design $d^{*}$')
		axarr[2] = ternary_plot.plot(d, 
			t=self.degreeSpace[int(self.true_degree)], 
			r=self.orderSpace[int(self.true_order)], 
			ax=axarr[2], title=r'"Optimal" Design $d^{*}$')
		y1, y2 = axarr[2].get_ylim()
		axarr[2].set_ylim([y1, y2*1.1])

		f.subplots_adjust(left=0.05, bottom=0.10, right=0.95, top=0.90, 
			wspace=0.0, hspace=0.0)
		return f


	def viz_full(self, d, old_prior):
		f, axarr = plt.subplots(2,4,figsize=(13,7))
		prior_ent = sharma_mittal.sm_entropy(old_prior, r=1.0, t=1.0)

		cmap = sns.cubehelix_palette(12, start=0, rot=0.5, 
				light=1, as_cmap=True, reverse=True, hue=0.0, dark=0.05)

		axarr[0,0] = self.viz_prior(old_prior, ax=axarr[0,0], 
			title=r'P(model), ent(P)=%.2f' % prior_ent, annotate=self.step)

		axarr[1,0] = ternary_plot.plot_small(d, 
			t=self.degreeSpace[int(self.true_degree)], 
			r=self.orderSpace[int(self.true_order)], 
			ax=axarr[1,0], title=r'"Optimal" Design $d^{*}$')
		y1, y2 = axarr[1,0].get_ylim()
		axarr[1,0].set_ylim([y1, y2*1.1])

		#labeling
		vspac = 0.05; ypos = 0.95; xpos = -0.1; fsize=9
		axarr[1,0].text(xpos, ypos, 
			r'info gain (d)=%.2f' % self.results.pure_utilities[-1], fontsize=fsize,
			ha='left', va='center', color='k', transform=axarr[1,0].transAxes)
		axarr[1,0].text(xpos, ypos - vspac, 
			r'learnability=%.2f' % self.results.lscores[-1], fontsize=fsize,
			ha='left', va='center', color='k', transform=axarr[1,0].transAxes)
		axarr[1,0].text(xpos, ypos - 2*vspac, 
			r'goodness=%.2f' % self.results.utilities[-1], fontsize=fsize,
			ha='left', va='center', color='k', transform=axarr[1,0].transAxes)
		axarr[1,0].text(xpos, ypos - 3*vspac, 
			r'choice=$Q_%d$' % (2-self.results.responses[-1]), fontsize=fsize,
			ha='left', va='center', color='k', transform=axarr[1,0].transAxes)

		axarr[0,1] = self.viz_relevance(d, q=0, annotate=False, ax=axarr[0,1], 
			title=r'info gain of $Q_1$ | model (norm.)', cmap=cmap)
		axarr[1,1] = self.viz_relevance(d, q=1, annotate=False, ax=axarr[1,1], 
			title=r'info gain of $Q_2$ | model (norm.)', cmap=cmap)
		axarr[0,2] = self.viz_choice_prob(d, ax=axarr[0,2], 
			title=r'P(choice = $Q_1$ | model), P($Q_1$)=%.2f' % self.marginals[1], annotate=self.step)
		axarr[1,2] = self.viz_choice_prob(d, inv=True, ax=axarr[1,2], 
			title=r'P(choice = $Q_2$ | model), P($Q_2$)=%.2f' % self.marginals[0], annotate=self.step)

		#update posterior
		post_q1 = np.copy(old_prior)
		post_q2 = np.copy(old_prior)
		for (t,r,b), p in np.ndenumerate(old_prior): 
			post_q1[t][r][b] = self.model_likelihood(
				d, self.degreeSpace[t], self.orderSpace[r], self.betaSpace[b])[1] * p
			post_q2[t][r][b] = self.model_likelihood(
				d, self.degreeSpace[t], self.orderSpace[r], self.betaSpace[b])[0] * p
		post_q1 /= np.sum(post_q1)
		post_q2 /= np.sum(post_q2)
		post_ent_q1 = sharma_mittal.sm_entropy(post_q1, r=1.0, t=1.0)
		post_ent_q2 = sharma_mittal.sm_entropy(post_q2, r=1.0, t=1.0)

		axarr[0,3] = self.viz_prior(post_q1, ax=axarr[0,3],
			title=r'P(model | choice=$Q_1$), ent(P)=%.2f' % post_ent_q1, annotate=self.step)
		axarr[1,3] = self.viz_prior(post_q2, ax=axarr[1,3],
			title=r'P(model | choice=$Q_2$), ent(P)=%.2f' % post_ent_q2, annotate=self.step)

		# TODO: rotate tick labels !!!
		for ax in axarr.flatten():
			_ = [tick.label.set_fontsize(7) for tick in ax.xaxis.get_major_ticks()]
			_ = [tick.label.set_fontsize(7) for tick in ax.yaxis.get_major_ticks()]

		f.suptitle('Results after ADO step %d' % self.step, x=0.5, y=0.95, fontsize=14, ha='center', va='center')
		f.subplots_adjust(left=0.05, bottom=0.10, right=0.98, top=0.90, 
			wspace=0.1, hspace=0.3)
		return f


	def quadratic_divergence(self, p, q):
		""" Compute quadratic divergence between two distributions P and Q """
		p = np.array(p).flatten()
		p /= np.sum(p)
		q = np.array(q).flatten()
		q /= np.sum(q)
		if len(p) != len(q): return False
		div = 0
		for pp, qq in zip(p, q):
			div += (pp - qq) ** 2
		return div
	
	def learnability(self, d, split=False, noH=False):
		"""
		Function to compute learnability of a given environment d.

		Description:
		------------
		Learnability has two components: the posterior and the joint 
		distribution. The posterior is scored according to how dissimilar 
		its two largest values are. The joint component penalizes joint 
		feature combinations that are very small (and non-zero).

		Arguments:
		----------
		d (design object): the environment to be evaluated

		Returns:
		--------
		learnability: posterior and joint score (between 0 and 1)

		"""
		def transformPosterior(x, c=15, n=2):
			return 1 if x == 0 else (2/(1+ np.exp(-c*abs(x-n))) - 1 )

		def transformJoint(x, e=0.01, c=30):
			if x == 0: return 1
			return 0 if x <= e else (2/(1+ np.exp(-c*(x-e))) - 1 )

		postScore = []
		altProbabilities = np.zeros(self.N_categories)
		for idx, featureComb in enumerate(d.postComb):
			altProbabilities[np.argmax(featureComb)] += d.margComb[idx]
			n = np.max(featureComb)
			postScore.append(
				[transformPosterior(f, n=n) for f in np.sort(featureComb)[:-1]])
		postScore = np.prod(postScore)
		jointScore = np.prod([transformJoint(x) for x in 
			d.margComb.flatten()])
	
		maxEnt = np.log(np.count_nonzero(altProbabilities))
		ent = sharma_mittal.sm_entropy(altProbabilities, 1.0, 1.0)
		hetScore = 0 if maxEnt == 0 else ent/maxEnt

		if noH: return postScore * jointScore
		if split: return [postScore, jointScore, hetScore]
		else: 	  return postScore * jointScore * hetScore

	def choice_probability(self, sm_relevance, beta):
		#luce choice rule
		nom = np.power(sm_relevance[0], beta)
		denom = (np.power(sm_relevance[0], beta) 
			+ np.power(sm_relevance[1], beta))
		if sm_relevance[0] == sm_relevance[1]: return 0.5
		return 0 if (nom == 0) else nom/denom

	def sm_usefulness(self, d, degree, order, normalize=True):
		prior_entropy = sharma_mittal.sm_entropy(d.prior, degree, order)
		posterior_entropy = [np.sum([
			d.marginal[i][j] * sharma_mittal.sm_entropy(dd, degree, order) 
			for j, dd in enumerate(posterior)]) 
				for i, posterior in enumerate(d.posterior)]
		sm_relevance = prior_entropy - np.array(posterior_entropy)
		if not normalize: return sm_relevance
		oidx = np.where(self.orderSpace==order)[0][0]
		didx = np.where(self.degreeSpace==degree)[0][0]
		upperBound = self.bounds[0][oidx][didx]
		lowerBound = self.bounds[1][oidx][didx]
		if upperBound == lowerBound: return [0,0]
		for i, uu in enumerate(sm_relevance): #PREVENT u from being out of bounds!
			sm_relevance[i] = min(upperBound, sm_relevance[i])
			sm_relevance[i] = max(lowerBound, sm_relevance[i])
		return (sm_relevance - lowerBound)/(upperBound - lowerBound)

	def binomial(self, P, n):
		return np.array([(np.math.factorial(n)/(
			np.math.factorial(n-y)*np.math.factorial(y))) * 
			P**y * (1-P) ** (n-y) for y in np.arange(n+1)])

	def model_likelihood(self, d, degree, order, beta):
		sm_relevance = self.sm_usefulness(d, degree, order)		
		P = self.choice_probability(sm_relevance, beta=beta)
		# Hard bounds
		# if P <= 10e-10: P = 0
		# if P >= 1-10e-10: P = 1
		return self.binomial(P, n=self.n_trials)

	def likelihood_function(self, d):
		likelihood = np.empty((self.N,self.N,
			self.N_beta,self.n_trials+1))
		for (t,r,b), _ in np.ndenumerate(np.empty((self.N,self.N,self.N_beta))):
			likelihood[t][r][b] = self.model_likelihood(d, self.degreeSpace[t], 
				self.orderSpace[r], self.betaSpace[b])		
		return likelihood/np.sum(likelihood)

	def local_utility(self, t, r, b, y, post):
		if self.prior[t][r][b] == 0.: return 0 
		def log0(x): 
			# return np.where(x==0., 0., np.log(x))
			return 0 if x==0 else np.log(x)
		tmp = log0(post/self.prior[r][t][b])
		return 0 if np.isinf(tmp) == True else tmp

	def marginal_response_probability(self, d):
		marginal_sum = np.zeros(self.n_trials+1)
		for (t,r,b), p in np.ndenumerate(self.prior):
			marginal_sum += p * self.model_likelihood(d, self.degreeSpace[t], self.orderSpace[r], self.betaSpace[b])
		marginal_sum /= np.sum(marginal_sum)
		return marginal_sum

	#compute (global) utility of a design, given our beliefs about the subject
	def global_utility(self, d, pure_ig=False):
		
		likelihood = np.empty((self.N,self.N,self.N_beta,self.n_trials+1))
		posterior = np.empty((self.n_trials+1,self.N,self.N,self.N_beta))

		for (t,r,b), p in np.ndenumerate(self.prior):
			likelihood[t][r][b] = self.model_likelihood(
				d, self.degreeSpace[t], self.orderSpace[r], self.betaSpace[b])
			for y in np.arange(self.n_trials+1):
				posterior[y][t][r][b] = p * likelihood[t][r][b][y]
		unnorm_post = np.copy(posterior)
		for y in np.arange(self.n_trials+1): #normalize posteriors
			posterior[y] /= np.sum(posterior[y])

		self.prior_ent = sharma_mittal.sm_entropy(self.prior, r=1.0, t=1.0)

		margpost = [np.sum(unnorm_post[y]) for y in np.arange(self.n_trials+1)]
		post_ent = [sharma_mittal.sm_entropy(unnorm_post[y], r=1.0, t=1.0) for y in np.arange(self.n_trials+1)]
		exp_post_ent = np.dot(margpost, post_ent)
		global_utility = self.prior_ent - exp_post_ent

		# global_utility = 0
		# for (t,r,b), p in np.ndenumerate(self.prior):
		# 	for y in np.arange(self.n_trials+1):
		# 		global_utility += p * likelihood[t][r][b][y] * self.local_utility(
		# 			t,r,b,y,posterior[y][t][r][b])
		# print(global_utility)

		if pure_ig: 	return global_utility

		if self.l_flag: return global_utility * self.learnability(d)
		else:  			return global_utility

	def compute_results(self):
		""" 
		USE THIS FUNCTION TO SAVE RESULTS TO ADO.results AFTER EACH ITERATION

		"""
		#Mean Squared Error
		dist_prior = np.add.reduce(self.prior, axis=2)
		dist_prior /= np.sum(dist_prior) #normalized prior

		#VARIANCE
		order_variance = 0
		degree_variance = 0
		beta_variance = 0
		for (t, r, b), p in np.ndenumerate(self.prior):
			degree_distance = abs(self.true_degree - t)**2
			degree_variance += degree_distance * p
			order_distance = abs(self.true_order - r)**2
			order_variance += order_distance * p
			beta_distance = abs(self.true_beta - b)**2
			beta_variance += beta_distance * p
			# all_distance = degree_distance + order_distance + beta_distance
			# all_variance += all_distance * p
		self.results.order_variance_array.append(np.sqrt(order_variance))
		self.results.degree_variance_array.append(np.sqrt(degree_variance))
		self.results.beta_variance_array.append(np.sqrt(beta_variance))
		self.results.total_variance_array.append(
			np.sqrt(order_variance) + 
			np.sqrt(degree_variance) +
			np.sqrt(beta_variance))

		self.guess_order = np.sum([p * r for p, r in 
			zip((np.sum(dist_prior, axis=0)/
				np.sum(dist_prior)).flatten(), np.arange(self.N))])
		self.results.order_guesses.append(self.guess_order)
		self.results.ent_order_array.append(sharma_mittal.sm_entropy(
			(np.sum(dist_prior, axis=0)/np.sum(dist_prior)).flatten(),
			r=1.0, t=1.0))

		self.guess_degree = np.sum([p * t for p, t in 
			zip((np.sum(dist_prior, axis=1)/
				np.sum(dist_prior)).flatten(), np.arange(self.N))])
		self.results.degree_guesses.append(self.guess_degree)
		self.results.ent_degree_array.append(sharma_mittal.sm_entropy(
			(np.sum(dist_prior, axis=1)/np.sum(dist_prior)).flatten(),
			r=1.0, t=1.0))

		dist_beta = [np.sum(
				self.prior[:,:,i]) for i in np.arange(self.N_beta)]
		dist_beta /= np.sum(dist_beta)
		self.guess_beta = np.sum([p * b for p, b in 
			zip(dist_beta, np.arange(self.N_beta))])
		self.results.beta_guesses.append(self.guess_beta)
		self.results.ent_beta_array.append(sharma_mittal.sm_entropy(
			dist_beta, r=1.0, t=1.0))

		self.mse_order = np.sqrt((self.true_order - self.guess_order) ** 2)
		self.mse_degree = np.sqrt((self.true_degree - self.guess_degree) ** 2)
		self.mse_beta = np.sqrt((self.true_beta - self.guess_beta) ** 2)

		#Shannon Entropy
		self.results.post_entropy_array.append(
			sharma_mittal.sm_entropy(self.prior, r=1.0, t=1.0))
		self.results.mse_order_array.append(self.mse_order)
		self.results.mse_degree_array.append(self.mse_degree)
		self.results.mse_beta_array.append(self.mse_beta)





	#run ado iteration
	def run(self, func):
		""" 
			Recursive function that is at the heart of ADO.
		"""

		#stopping criterion
		if self.step >= self.max_steps: return True
		else: self.step += 1
		
		#specify filename for save file
		if self.save_results:
			if not self.filename:
				fileName = 'hado_' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '_id_' + str(self.fileID) + '_step_' + str(self.step)
			else:
				fileName = self.filename
			fileName = 'data/' + fileName

		#start iteration
		self.tic()
		self.log("\n### STEP %s ###" % self.step)

		#using pre-specified design
		if self.method == 'pre':
			self.log("\t### using pre-specified design ... ###")
			d, u = self.step_design, self.global_utility(self.step_design)
			self.toc()
		#design optimization step
		else:
			self.log("\t### executing optimization method ... ###")
			d, u = func.optimize(self, **self.m_args)
			self.log("\t### ... done! ###")
			self.toc()
			self.log("\t### time per design eval = %.12f" % 
				(self.trial_time / (self.m_args['n_designs'] * 1.0)))
		
		#save results
		self.results.lscores.append(self.learnability(d))
		self.results.dobject.append(d)
		self.results.designs.append(d.oneLine())
		self.results.utilities.append(u)
		self.results.pure_utilities.append(self.global_utility(d, pure_ig=True))
		self.log("\t### U(d) = %2.4f ###" % self.global_utility(d, pure_ig=True))
		lj, jp, lh = self.learnability(d, split=True)
		self.log("\t### L(d) = %2.8f (post), %2.8f (joint) ###" % (lj, jp))
		self.log("\t### Heterogeneity (0-1) = %1.5f ###" % lh)
		self.log("\t### U(.)*L(.)  = %2.4f ###" % u)

		#factor learnability into beta
		if self.l_flag: beta_star = self.betaSpace[self.true_beta] * self.learnability(d, noH=True)
		else: beta_star = self.betaSpace[self.true_beta]
		self.log("\t### effective beta = %.5f" % beta_star)

		if self.set_response != -1:
			response = self.set_response
			self.log("\t### pre-specified response: %s" % response)
		else:
			#simulate experiment
			response_distribution = self.model_likelihood(d, 
				self.degreeSpace[self.true_degree], 
				self.orderSpace[self.true_order], 
				beta_star)
			# response = sp.choice(self.n_trials+1, p=response_distribution)[0] #hack for numpy v. < 1.6
			response = np.random.choice(self.n_trials+1, p=response_distribution)
			self.log("\t### simulated response: %s" % response)
		
		self.results.responses.append(response)
		self.marginals = self.marginal_response_probability(d) #probability of response under model(s)

		#update posterior
		old_prior = np.copy(self.prior)
		posterior_response_0 = np.copy(self.prior)
		posterior_response_1 = np.copy(self.prior)

		def update_prior(prior, response):
			for (t,r,b), p in np.ndenumerate(prior): 
				prior[t][r][b] = self.model_likelihood(
					d, self.degreeSpace[t], self.orderSpace[r], self.betaSpace[b])[response] * p
			prior /= np.sum(prior)
			return prior
		
		posterior_response_0 = update_prior(posterior_response_0, 0)
		posterior_response_1 = update_prior(posterior_response_1, 1)
		
		#actual experiment: 
		self.prior = update_prior(self.prior, response)
		self.prior_ent = sharma_mittal.sm_entropy(self.prior, r=1.0, t=1.0)

		#compute and store results
		self.results.time_array.append(self.trial_time)
		self.compute_results() 
		self.log("\t### execution time (s): %s ###" % self.trial_time)

		# def pround(a, decimals=4):
		# 	a = np.round_(a, decimals=decimals)
		# 	midx = np.argmax(a)
		# 	e = 1 - np.sum(a)
		# 	a[midx] += e
		# 	return a

		if self.plotting or self.save_results:
			f = self.viz_full(d, old_prior)
			if self.save_results:
				fileName +=  '_stage_' + str(self.step).zfill(3)
				f.savefig(fileName+'_figures.pdf', dpi=150, bbox_inches='tight')
				with open(fileName+'_design.txt', 'w') as text_file:
					strtofile = 'Prior\n' + \
						np.array_str(self.results.dobject[-1].prior) + \
						'\n\nLikelihoods\n' + \
						np.array_str(self.results.dobject[-1].marginal) + \
						'\n\nFeature marginals\n' + \
						np.array_str(self.results.dobject[-1].likelihoods) + \
						'\n\nFeature Posteriors\n' + \
						np.array_str(self.results.dobject[-1].posterior) + \
						'\n\nMarginals (combined)\n' + \
						np.array_str(self.results.dobject[-1].margComb) + \
						'\n\nPosteriors (combined)\n' + \
						np.array_str(self.results.dobject[-1].postComb) + \
						'\n\nJoint probabilities\n' + \
						np.array_str(self.results.dobject[-1].joint) + \
						'\n\n!!! Use flattened joint probabilities array (from top left to bottom right) for experience-based learning experiment !!!'
					self.log("\n\nResults successfully SAVED as %s!!" % fileName)
					strtofile = strtofile + '\n\n' + self.log_string
					text_file.write("Environment:\n%s" % strtofile)

				# save design
				np.save(fileName+'_design.npy', d.asNumpy(), allow_pickle=False)
				# save prior
				np.save(fileName+'_prior.npy', old_prior, allow_pickle=False)
				# save posterior(s)
				np.save(fileName+'_posterior_0.npy', posterior_response_1, allow_pickle=False)
				np.save(fileName+'_posterior_1.npy', posterior_response_0, allow_pickle=False)

		'''	
		if self.plotting or self.save_results:
			f = self.viz_triplet(old_prior, d)
			self.viz_like(d, annotate=True, ax=None)
			# if self.plotting: plt.show()
			if self.save_results: f.savefig(
				'plots/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.pdf', 
				dpi=150, bbox_inches='tight')
		'''
		return self.run(func)

	
	def run_full_simulation(self):
		""" Use to run full simualtion experiment, including design optimization and response sampling """
		return self.execute()

	def run_step_simulation(self, prior=None, custom_design=None, response=None, last_step=0):
		""" Run simulations/DO only for one time step, allows for custom design, prior, and response. """ 
		subPath = '' #'data/'

		self.max_steps = last_step+1
		self.step = last_step
		self.stepwise = True

		# CUSTOM PRIOR
		if prior != None: #TODO: check if prior matches !!!
			print "Loading prior from file!!!"
			self.prior = np.load(subPath+prior)
			self.priorName = 'custom'
		
		# CUSTOM DESIGN
		if custom_design != None:
			print "Loading design from file (using pre-specified environment)!!!"
			self.method = 'pre'
			d = np.load(subPath+custom_design)
			size = list(d.shape)[::-1]; size[1] = int(np.log2(size[1]))
			self.N_categories, self.N_features = size
			joint = d.flatten()
			self.step_design = design.design(joint=joint, size=size)

		# CUSTOM RESPONSE
		if response != None:
			print "Response pre-specified"
			self.set_response = response
			self.simulation = False

		return self.execute()


	def execute(self):
		self.log("\nTrue order: %s (idx: %s) degree: " \
				"%s (idx: %s) and beta %s (idx: %s)" % (
				self.orderSpace[self.true_order], 
				self.true_order,
				self.degreeSpace[self.true_degree], 
				self.true_degree,
				self.betaSpace[self.true_beta],
				self.true_beta))
		self.log("Designs evaluated: %s\n" % self.m_args['n_designs'])

		#determine design optimization method
		if 	 self.method == 'de': 			func = differential_evolution
		elif self.method == 'mcmc': 		func = mcmc
		elif self.method == 'smc': 			func = smc
		elif self.method == 'best_rand': 	func = random_search
		elif self.method == 'single_rand':  func = single_random
		else: 								func = random_search
		
		self.log("Optimization method set to: ** " \
			"%s **" % func.__pname__)

		return self.run(func)
		# try:
			# return self.run(func)
		# except Exception as e:
			# raise e
		






# ____________________________________________________________________________________________________
# ____________________________________________________________________________________________________

def initialize(**kwargs):
	#number of categories and features
	N_categories 	= kwargs.get('N_categories', 3)
	N_features 		= kwargs.get('N_features', 2)

	#number of response trials
	n_trials = kwargs.get('n_trials', 1)
	q = kwargs.get('q', 0.00) #??? not used any more ???
	l_flag = kwargs.get('learnability', True) #learnability on/off
	
	#SM grid size and define parameter space
	N = kwargs.get('N', 13) #should not be modified
	N_beta 			= kwargs.get('N_beta', 2)

	g 	= kwargs.get('granularity', 1)
	orderLowerBound = -3
	orderUpperBound = orderLowerBound + N - 2
	orderSpace  = np.logspace(orderLowerBound, orderUpperBound, 
		num=g*(N-2)+1, endpoint=True, base=2.0)
	orderSpace = np.concatenate(([0], orderSpace), axis=0)
	orderSpace = np.round_(orderSpace, 3)

	degreeLowerBound = -3
	degreeUpperBound = degreeLowerBound + N - 2
	degreeSpace = np.logspace(degreeLowerBound, degreeUpperBound, 
		num=g*(N-2)+1, endpoint=True, base=2.0)
	degreeSpace = np.concatenate(([0], degreeSpace), axis=0)
	degreeSpace = np.round_(degreeSpace, 3)

	betaSpace = kwargs.get('betaSpace', 0)

	if betaSpace==0:
		betaLowerBound = -1
		betaUpperBound = betaLowerBound + N_beta - 1
		betaSpace = np.logspace(betaLowerBound, betaUpperBound, 
			num=N_beta, endpoint=True, base=2.0) ** 2

	#subject may have single true underlying parameter triple
	true_beta 	= kwargs.get('true_beta', None)
	true_order 	= kwargs.get('true_order', None)
	true_degree = kwargs.get('true_degree', None)

	#hado stopping criterion
	max_steps = kwargs.get('max_steps', 5)

	prior 		 = kwargs.get('prior', 'uni')
	method 		 = kwargs.get('method', 'rand')
	logging 	 = kwargs.get('logging', True)
	plotting 	 = kwargs.get('plotting', False)
	save_results = kwargs.get('save_results', False)
	filename 	 = kwargs.get('filename', '')

	#method arguments
	m_args = kwargs.get('m_args', {
		'parallelize' : False,
		'n_designs' : 10,
		'n_iterations' : 1
		})

	#create ADO object here =>
	ADO = Ado(max_steps, n_trials,
			 orderSpace, degreeSpace, betaSpace,
			 true_order, true_degree, true_beta,
			 N_categories, N_features, prior, g, 
			 method, m_args, logging, plotting,
			 save_results, l_flag, filename)

	return ADO