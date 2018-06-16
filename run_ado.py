#!/usr/bin/python
import sys
import getopt
import hado.ado_core_routines as ado
import numpy as np
from ast import literal_eval

def main():
	#parse simulation parameters
	try:
		opts, args = getopt.getopt(sys.argv[1:], "", ["config=", "stage=", "help=", "full", "prior=", "design=", "seed="])
	except getopt.GetoptError as err:
		print(err) 
		sys.exit(2)

	#default parameters
	current_stage=1
	runFullSimulation=False
	prior_fileName=None
	design_fileName=None
	response=None

	for o, a in opts:
		if o in ("--config"):
			# Read in the config file
			s = ''
			with open(a, 'r') as configFile:
				for line in configFile:
					s += line.split('#', 1)[0].rstrip() + ' '
			kwargs = dict((k, literal_eval(v)) for k, v in (pair.split('=') for pair in s.split()))
			
			#make separate m_args dict and pass on!
			# parallelize=True
			# n_designs=10
			# n_iterations=1

			# initialize the ADO object
			ado_instance = ado.initialize(**kwargs)

		elif o in ("--stage"):
			current_stage = int(a)
		elif o in ("--full"):
			runFullSimulation = True
		elif o in  ("--prior"):
			prior_fileName = a
		elif o in  ("--design"):
			design_fileName = a
		elif o in  ("--seed"):
			np.random.seed(int(a))
		else:
			assert False, "unhandled option"

	#execute simulations
	if runFullSimulation:
		ado_instance.run_full_simulation()
	else:
		ado_instance.run_step_simulation(prior=prior_fileName, custom_design=design_fileName, response=response, last_step=current_stage-1)
			
if __name__ == "__main__":
	main()