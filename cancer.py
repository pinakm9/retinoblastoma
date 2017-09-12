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
	ax.plot(x, obs(x), color = 'blue')
	ax.plot(x, fit(x), color = 'red')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.savefig(img_name + '.png')