import Tkinter
import pandas as pd
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.integrate as integrate
from scipy import optimize as op
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
def fit_plot(obs, fit, limits, xlabel = 'x', ylabel = 'y', img_name = 'image'):
	fig, ax = plt.subplots(figsize=(8,5))
	x = np.linspace(*limits)
	y = [obs(t) for t in x]
	ax.plot(x, y, color = 'blue')
	y = [fit(t) for t in x]
	ax.plot(x, y, color = 'red')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.savefig('plots/'+ img_name + '.png')

# Calculates relative error in l2 norm
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

# Class for managing various cancer data 
class Cancer(object):

	GROUPS = parse_groups(read_data('./../data/03_cancer_agerelated.csv')[1][0])
	GR_COUNT = len(GROUPS)
 
 	def __init__(self, name, rates):
 		self.name = name
 		self.rates = list(map(lambda x: float(x) if x != '~' and str(x).lower() != 'nan' else 0, rates))
 		self.total = sum(self.rates)
 
 	# Identifies proper age group given an age
	def id_group(self, age):
		if age < 1:
			return 0
		elif age >= 85:
			return self.GR_COUNT
		else:
			for i, group in enumerate(self.GROUPS[1: self.GR_COUNT-1], 1):
				if (age >= group[0] and age <= group[1]) or (age >= group[1] and age < self.GROUPS[i+1][0]):
					return i

 	# Empirical cdf
 	def ecdf(self, age):
 		index = self.id_group(age)
 		return sum(self.rates[:index+1])*1e-5/self.total

 		

