import Tkinter
import pandas as pd
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.integrate as integrate
from scipy import optimize as op
from scipy.special import gammainc
import re 

cell_count = 2e6 # Number of retina cells in a single eye
rb_rate = 1./15000.0 # Incidence rate of retinoblastoma
div = '~'*80 # A divider for printing outputs

# Reads observational data into python objects
def read_data(file):
	vals, data = [], []
	csv = pd.read_csv(file)
	for val in csv.columns.values:
		vals.append(val)
		data.append(list(csv.loc[0:,val]))
	return [vals, data]

# Plots observational data and a best-fit curve
def fit_plot(obs, fit, limits, img_name = 'image'):
	fig, ax = plt.subplots(figsize=(8,5))
	x = np.linspace(*limits)
	y = [obs(t) for t in x]
	ax.plot(x, y, color = 'blue', label='ecdf')
	y = [fit(t) for t in x]
	ax.plot(x, y, color = 'red', label='model')
	ax.set_xlabel('age in months at the time of diagnosis')
	ax.set_ylabel('probability of developing cancer')
	plt.legend()
	plt.savefig('plots/'+ img_name + '.png')

# Calculates relative error in L^2 norm
def rel_err(exact, approx, interval):
	a, b = interval
	numerator = integrate.quad(lambda x: (exact(x)-approx(x))**2, a, b)[0]
	denominator = integrate.quad(lambda x: exact(x)**2, a, b)[0]
	return 100*(numerator/denominator)**0.5

# Identifies age-groups in cancer data
def parse_groups(groups):
	age_group = []
	for group in groups:
		ages = list(map(int, re.findall('\d{1,2}', group)))
		age_group.append(ages)
	return age_group		

# Model where each hit occurs with same probability
def model1(t, k, p):
	return gammainc(k, p*t)

# Model where hit probabilities are in arithmatic progression
def model2(t, k, params):
	p, d = params
	i, total, _prod = 1, 0, 1
	while i <= k:
		prod, j, _p = 1, 1, p+(i-1)*d
		while j <= k:
			if j != i:
				prod = prod*(j-i)*d
			j = j+1
		total = total + (1 - np.exp(-t*_p))/(prod*_p)
		_prod = _prod*_p
		i = i+1
	return total*_prod

# Class for managing various cancer data 
class Cancer(object):

	GROUPS = parse_groups(read_data('./../data/03_cancer_agerelated.csv')[1][0][5:-1])
	GR_COUNT = len(GROUPS)
 
 	def __init__(self, name, rates):
 		self.name = name
 		self.rates = list(map(lambda x: float(x) if x != '~' and str(x).lower() != 'nan' else 0, rates[5:-1]))
 		self.total = sum(self.rates)
 
 	# Identifies proper age group given an age
	def id_group(self, age):
		for i, group in enumerate(self.GROUPS):
				if (age >= group[0] and age <= group[1]) or (age > group[1] and age < self.GROUPS[i+1][0]):
					return i
 	# Empirical cdf
 	def ecdf(self, age):
 		index = self.id_group(age)
 		return sum(self.rates[:index+1])*1e-5

 	# Tries to fit model to the data
 	def fit(self, model_flag):
 		params, errors = [], []
 		model = model1 if model_flag == '2(a)' else model2
 		for k in range(2, 15):
 			loss = lambda params: sum([(self.ecdf(t) - model(t, k, params))**2 for t in range(20, 81)])
 			if model_flag == '2(a)':
 				params.append(op.minimize_scalar(loss, bounds= (0,1), method= 'bounded').x)
 			else:
 				cons = ({'type' : 'ineq', 'fun' : lambda x: x[0]+(k-1)*x[1]},
 						{'type' : 'ineq', 'fun' : lambda x: 1-x[0]-(k-1)*x[1]})
 				params.append(op.minimize(loss, (0.0001, 0.00785), constraints=cons).x)
 			errors.append(rel_err(self.ecdf, lambda t: model(t, k, params[k-2]), [20, 80]))
 		k = np.argmin(errors)
 		if model_flag == '2(a)':
 			print("{}\n{}\np = {:2.3}, k = {}, relative error = {:2.3}%\n{}"\
 				.format(self.name, div, params[k], k+2, errors[k], div))
 		else:
 			print("{}\n{}\np = {:2.3}, d = {:2.3}, k = {}, relative error = {:2.3}%\n{}"\
 				.format(self.name, div, params[k][0], params[k][1], k+2, errors[k], div))
 		fit_plot(self.ecdf, lambda t: model(t, k+2, params[k]), [20,80],\
 		 	img_name= self.name + ' model ' + model_flag)
 	