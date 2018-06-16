# HADO README

### Simulation code to run Hierarchical Adaptive Design Optimization (Cavagnaro, Myung, Pitt, NIPS 2009) with the experience-based learning paradigm.

People first engage in a classification learning task and subsequently in an information acquisition task. This code only works for the 3 category / 2 binary feature case. Our goal is to infer which model (=> set of three parameters, two parameters specifying degree/order of Sharma-Mittal entropy measure and one parameter associated with the response function) best describes participants behavior. This particular instance of the code only works for a binary response, i.e., subjects either prefer the first or the second feature.

In running simulations we are particularly interested in two use cases:
- Use case 1: We want to run full simulation studies over many stages, simulating sequential responses for particular combinations of degree/order/response beta.
- Use case 2: We want to perform Design Optimization (DO) for a particular prior (our current belief) to determine which experiment (design) to run next 

We can either run a full, multi-stage simulation, or specify prior and/or design and only run a single HADO step. To exectute these commands, no installation is necessary, just download and copy the repository to a local folder.

The code enables us to do both of these things by executing the python script in the main folder, run_ado.py, e.g., in the following ways:

```
python run_ado.py --config=config.txt --full --seed=123
```

```
python run_ado.py --config=config.txt --seed=123 --stage=2 --prior=data/test/hado_sid_0_stage_001_prior.npy
```

```
python run_ado.py --config=config.txt --seed=123 --stage=4 --prior=data/test/hado_sid_0_stage_003_posterior_0.npy --design=data/test/hado_sid_0_stage_004_design.npy
```

## Running simulations

In general, execution follows this format:

```
python run_ado.py [options]
```

**With the following options available:**

```
--config=path to config file
```
	
> Use the config flag to specify the path to the configuration file (a simple text file, see more details below) that contains the arguments for ado initialization.

```
--seed=number
```

>	Optionally use a particular random seed for reproducibility.

```
--full
```
	
>	Use the 'full' flag to run a full simulation with the arguments specified in the config file. None of the remaining three arguments will have an effect if full simulations are run. Otherwise (if the 'full' flag is omitted), the script will run a stepwise simulation.


<hr \>

**Only for stepwise simulations:**

```
--stage=number
```

>	The current experimental stage.

```
--prior=path to prior file (format: npy)
```

>	Specify a custom belief (prior) to be used in the current experimental stage.

```
--design=path to design file (format: npy)
```

>	Specify a custom design used to simulate the response. Specification of both prior and design for the current experimental stage are optional. If no prior is specified, use the prior in the config file. If no design is specified, perform DO with the optimization method specified in the config file to search for a design.


## The config file

All other important parameters are specified in a text file that is passed before execution using the config flag. A variety of parameters can be set. Here is an example from the current config file:

```
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
```

The config file can contain any number of those parameters. If omitted, they will fall back to their default values (specified accordingly). Parameters can be grouped according to the following categpories:

**Model parameters, parameter space (order, degree, beta)**

- granularity: 1 or 4, default 1
	- density of 2D degree/order parameter grid
- betaSpace: array of reals, default 1 
	- use only a single beta value of one 
- true_beta: integer
	- index to above array, at default sample uniformly
- true_order: integer
	- index to order log array, at default sample according to prior
- true_degree: integer
	- index to degree log array, at default sample according to prior
- prior: uni|exp|rand (string)
	- specifies the prior to be used (can be overriden by step-wise prior)

**Experimental design parameters:**

- N_categories: int, default 3
	- number of categories
- N_features: int, default 2
	- number of binary features
- n_trials: int, default 1
	- number of trials per mini experiment (this code only works with 1 trial!!)

**DO parameters (incl. method-specific optimization arguments)**
	
- method: mcmc|smc|rand|de
	- which design optimization method to use. The choice is between MCMC, sequential Monte Carlo, random, and Differential Evolution
- learnability: True
	- whether or not to include design learnability in utility calculations 
- max_steps: int, default 3 
	- maximum number of stages for full simulation
- parallelize: boolean, default True
	- Whether or not to compute multiple chains in parallel (will automatically use the maximum number of cores available)
- n_designs: int, default 10
	- How many experimental design to evaluate in total (across chains) in the DO step
- n_iterations: 1
	- Only relevant for DE: how many iterations to perform


**Misc parameters**

- logging: True
- plotting: False
- save_results: True
- filename: 'test/hado_sid_0'
	- All files will be saved in data/
	- Use the filename argument to specify a subfolder (needs to be created before execution) and filename scheme used for saving
	- The code will automatically append \_stage\_ at the end of the filename
	- When running branching simulations with multiple subjects, it is recommended to use a coding scheme to keep track of subject IDs an their responses, e.g. history_0101011 (for sequential history of responses)
