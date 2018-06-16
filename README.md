

HADO code for three categories, two binary features (questions), and a binary response (subject either prefered the first or the second feature.)

# Use case 1: we want to do DO for a particular prior, just specify the prior filename
# Use case 2: we want to additionally specify the design used for the experiment (skip DO), specify design via custom_design


No installation necessary, just copy the folder and CD into it!
Simulations are run by executing the run_ado.py file in the main folder:

python run_ado.py [Options]


Example commands:

python run_ado.py --config=config.txt --full --seed=123

python run_ado.py --config=config.txt --seed=123 --stage=2 --prior=data/test/hado_sid_0_stage_001_prior.npy

python run_ado.py --config=config.txt --seed=123 --stage=4 --prior=data/test/hado_sid_0_stage_003_posterior_0.npy --design=data/test/hado_sid_0_stage_004_design.npy



All results (plots/designs/beliefs) will be saved in data // logs will be saved in logs


Options:

--config=path to config file 
	
	use the config flat to specify a patch to the configuration file (a text file, see below), that contains the arguments passed on to the ado initialize method.

--seed=number

	Use a particular random seed for reproducibility

--full
	
	Use the full flag to run a full simulation with the arguments specified in the config file. Otherwise (if the full flag is omitted), run a stepwise simulation.


All the remaining arguments only have to be specified if stepwise simulations are run:

--stage=number

	The current experimental stage

--prior=path to prior file (format: npy)

	Specify a custom belief prior to be used in the current experimental stage

--design=path to design file (format: npy)

	Specify a custom design used to simulate the response.

	Specification of both a prior and a design for the current experimental stage are optional. If no prior is specified, use the prior in config.txt. If no design is specified, perform DO with the optimization method specified in config.txt to search for a design.


//hline


The config file:

true_order=7
true_degree=5
true_beta=0
betaSpace=[1]
granularity=4
method='mcmc'
prior='rand'
N_categories=3
N_features=2
max_steps=3
n_trials=1
learnability=True
logging=True
plotting=False
save_results=True
filename='jonathan_test/hado_sid_0_stage_'
parallelize=True
n_designs=10
n_iterations=1


None of the following is necessary, all can be omitted, if omitted, the method will default to the arguments specified below:

Model parameters, parameter space (order, degree, beta)

	granularity = 1|4 (1) //of logxlog grid, by default 13x13
	betaSpace
	true_beta=index to array, default uniform sample (specifies an index)
	true_order=7
	true_degree=5
	true_beta
	prior = uni|exp|rand

Experimental design parameters:

	N_categories=3
	N_features=2
	n_trials=1

DO parameters (incl. method-specific optimization arguments)
	
	method='mcmc'
	learnability=True
	max_steps=3
	parallelize=True
	n_designs=10
	n_iterations=1


Misc parameters

	logging=True
	plotting=False
	save_results=True
	filename='test/hado_sid_0' #will automatically add _stage_ at the end!!
	it is recommended to separate parameters and values by underscore, e.g. history_0101011 (for history of responses)

Filepath and filename: Specify a custom filepath and name to save the simulation results, relative to the main folder
