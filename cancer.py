import Tkinter
import pandas as pd
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
from random import sample as sample
import statsmodels.api as sm
import scipy.integrate as integrate
from scipy import optimize as op 

cell_count = 2e6 # Number of retina cells in a single eye
rb_rate = 1./15000.0 # Incidence rate of retinoblastoma
div = '~'*80 # A divider for printing outputs

# Reads observational data into python objects
def read_data(file):
	data = []
	csv = pd.read_csv(file)
	for val in csv.columns.values:
		data.append(list(csv.loc[0:,val]))
	return data

# Plots observational data and a best-fit curve
def fit_plot(obs, fit, limits, xlabel, ylabel, img_name):
	fig, ax = plt.subplots(figsize=(8,5))
	x = np.linspace(*limits)
	y = [obs(t) for t in x]
	ax.plot(x, y, color = 'blue')
	y = [fit(t) for t in x]
	ax.plot(x, y, color = 'red')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.savefig(img_name + '.png')

# Calculates relative error in l2 norm
def rel_err(exact, approx, interval):
	a, b = interval
	numerator = integrate.quad(lambda x: (exact(x)-approx(x))**2, a, b)[0]
	denominator = integrate.quad(lambda x: exact(x)**2, a, b)[0]
	return 100*(numerator/denominator)**0.5
