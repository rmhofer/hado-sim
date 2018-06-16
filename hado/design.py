#design.py
import numpy as np

class design():
	"""
		class to create and store environment
		computes all relevant probabilities

		inputs (alternatives):
			size (tuple) and alpha -> generates a random design
			pr and lk -> priors and likelihoods
			p -> joint probability vector
	"""
	def __init__(self, **kwargs):
		pr = kwargs.get('pr', [])
		lk = kwargs.get('lk', [])
		joint = kwargs.get('joint', [])
		size = kwargs.get('size', None)

		if not joint == [] and not size == None:
			if len(joint) != 2**size[1] * size[0]:
				print("Not enough probabilities specified!\n\n")
				return False
			tmp_cols = np.reshape(joint, (2**size[1], size[0]))

			self.prior = np.sum(tmp_cols, axis=0)
			self.likelihoods = np.array(
				[[(tmp_cols[0,i] + tmp_cols[j+1,i])/self.prior[i] 
				for i in np.arange(size[0])] for j in np.arange(size[1])])

		elif not size == None: #create random environment
			alpha = kwargs.get('alpha', 0.75)
			while True:
				self.prior = np.random.dirichlet(np.ones(
					size[0]) * alpha, size=1)[0]
				if np.sum(self.prior) - 1.0 == 0: break
			self.likelihoods = np.array([np.random.dirichlet(np.ones(size[1]) * 
				alpha, size=size[0])[:,0] for i in np.arange(size[1])])
		elif not pr == [] and not lk == []:
			self.prior = np.append(np.array(pr), 1.0-np.sum(pr))
			self.likelihoods = np.array(lk)
		else:
			print("No probabilities specified!\n\n")
			return False
		
		self.probArray = np.append(self.prior.flatten(), self.likelihoods.flatten())

		marginal = []
		for lk in self.likelihoods:
			ff = np.sum([pp * ll for pp, ll in zip(self.prior, lk)])
			marginal.append([ff, 1-ff])
		self.marginal = np.array(marginal)

		postArr = []
		for i, lk in enumerate(self.likelihoods):
			temp = []
			temp.append([pp*ll/self.marginal[i][0]
				for pp, ll in zip(self.prior, lk)])
			temp.append([pp*(1-ll)/self.marginal[i][1]
				for pp, ll in zip(self.prior, lk)])
			idx = np.where(np.isnan(temp) == 1) #take a close look at this!
			temp = np.array(temp)
			temp[idx] = 0
			postArr.append(temp)
		if self.likelihoods.shape[0] == 1:
			postArr = postArr[0]
		self.posterior = np.array(postArr)	
		
		# postComb = []
		# for i, p in np.ndenumerate(self.prior):


		# self.joint = []
		self.postComb = np.zeros((self.marginal.flatten().size, self.prior.size))

		for i, p in np.ndenumerate(self.prior):
			Nfeature = len(self.likelihoods[:,i])
			empty = np.zeros((Nfeature,Nfeature))
			f_val = 0
			for f, _ in np.ndenumerate(empty):
				tmp_arr = np.array([([self.likelihoods[:,i][j], 
						1-self.likelihoods[:,i][j]])[f[j]] 
						for j in np.arange(len(f))])
				self.postComb[f_val,i] += np.prod(tmp_arr) * p
				f_val += 1
				# self.joint.append(np.prod(tmp_arr) * p)
		# self.joint = np.array(self.joint)
		self.margComb = np.sum(self.postComb, axis=1)
		self.joint = self.postComb

		with np.errstate(divide='ignore', invalid='ignore'):
			self.postComb = self.postComb/self.postComb.sum(axis=1)[:,None]
		self.postComb = np.nan_to_num(self.postComb)
		questions = []
		for po, ma in zip(self.posterior, self.marginal):
			questions.append(np.array([pp * mm for pp, mm in zip(po, ma)]))
		self.questions = np.array(questions)

	def __str__(self):
		return ("Questions:\n%s" % self.questions)

	def asNumpy(self):
		return self.joint

	def save(self, fileName):
		np.save(fileName, self.asNumpy(), allow_pickle=False)

	def oneLine(self):
		strRep = ''
		# for q in self.questions:
		# 	for f in q:
		# 		strRep += '[ '
		# 		for item in f:
		# 			strRep += '%.4f ' % item
		# 		strRep += ']'

		strRep += '[ '
		for item in self.joint.flatten():
			strRep += '%.12f ' % item
		strRep += ']'
		return strRep

n_c, n_f = 3,2
hado_pilot = [0,
        0.062,
        0,
        0.103,
        0.091,
        0.351,
        0,
        0,
        0,
        0,
        0,
        0.393]

sm1 = [0.040815390241782,
         0.000000000000000,
         0.076451891185478,
         0.119932759739877,
         0.160748149981689,
         0.084264162164448,
         0.131470905663947,
         0.000000000000000,
         0.000000000000000,
         0.386316741022779,
         0.000000000000000,
         0.000000000000000]
sm2 = [0.243152176370674,
         0.412671339801537,
         0.000000000000000,
         0.179665365997266,
         0.009393301073843,
         0.000000000000000,
         0.000000000000000,
         0.000000000000000,
         0.079973453372991,
         0.000000000000000,
         0.000000000000000,
         0.075144363383689]

# hado_pilot_design = design(joint=hado_pilot, size=(n_c, n_f))
# sm1_design = design(joint=sm1, size=(n_c, n_f))
# sm2_design = design(joint=sm2, size=(n_c, n_f))
# sm2_design.save('results/design_sm2')

