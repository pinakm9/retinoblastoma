from cancer import *

unilateral = [] # Stores unilateral retinoblastoma data set
p1, p2 = 0., 0. # Random hit probabilities
ecdf = 0 # Empirical cdf of age at the time of diagnosis of unilateral retinoblastoma 

def assignment_1_i():
	global unilateral, ecdf
	print('Problem 1(i)\n{}'.format(div))
	unilateral = read_data("./../data/03_cancer_unilateralretinoblastoma.csv")[0]
	ecdf = lambda t: sm.distributions.ECDF(unilateral)(t)*rb_rate

	def model(t, p): 
		return (1-((1-p)**t + t*p*(1-p)**(t-1))**(2*cell_count))*rb_rate

	def loss(p): 
		return integrate.quad(lambda t: (ecdf(t)-model(t, p))**2, 0, 73)[0]

	p = op.minimize_scalar(loss, bounds=(1e-10, 1e-2), method='bounded').x
	result = lambda t: model(t, p)
	print('Best fit p = {}, relative error = {}%\n{}'.format(p, rel_err(ecdf, result, [0, 73]), div))
	fit_plot(ecdf, result, [0, 73], 'age in months at the time of diagnosis',\
	 'probability of being diagnosed with retinoblastoma', 'single random factor (unilateral)')

def assignment_1_ii():
	global p1, p2
	print('Problem 1(ii)\n{}'.format(div))
	
	def model(t, p):
		i, T =  1, int(t)
		_p1, _p2 = p
		total = (1-_p1)**T
		while i <= T:
			total = total + _p1*(1-_p1)**(i-1)*(1-_p2)**(T-i)
			i = i+1
		return (1 - total**(2*cell_count))

	def loss(p):
		return integrate.quad(lambda t: (ecdf(t)-model(t, p))**2, 0, 36)[0]

	p1, p2 = op.minimize(loss, [40e-8, 1e-8]).x
	result = lambda t: model(t, [p1, p2])
	print('p1 = {}, p2 = {}, relative error =  {}%\n{}'.format(p1, p2, rel_err(ecdf, result, [0, 36]), div))
	fit_plot(ecdf, result, [0, 73], 'age in months at the time of diagnosis',\
	 'probability of being diagnosed with retinoblastoma', 'two random factors (unilateral)')

def assignment_1_iv():
	print('Problem 1(iii)\n{}'.format(div))

	def model(t, a):
		i, T =  1, int(t)
		total = (1-p1)**T
		while i <= T:
			j, prod = i+1, 1
			while j <= T:
				if j <= 36:
					prod = prod*(1-p2)
				else:
					prod = prod*(1-p2*np.exp(-a*(j-36)))
				j = j+1
			total = total + p1*(1-p1)**(i-1)*prod
			i = i+1
		return (1 - total**(2*cell_count))

	def loss(a):
		return integrate.quad(lambda t: (ecdf(t)-model(t, a))**2, 0, 73)[0]

	a = op.minimize_scalar(loss, bounds=(0, 5), method='bounded').x
	result = lambda t: model(t, a)
	print('a = {}, relative error =  {}%\n{}'.format(a, rel_err(ecdf, result, [0, 73]), div))
	fit_plot(ecdf, result, [0, 73], 'age in months at the time of diagnosis',\
	 'probability of being diagnosed with retinoblastoma', 'decaying p2 (unilateral)')
				

def assignment_1_iv():
	print('Problem 1(iii)\n{}'.format(div))

	def model(t, a):
		i, T =  1, int(t)
		total = (1-p1)**T
		while i <= T:
			j, prod = 1, 1
			while j <= i-1:
				if j <= 36:
					prod = prod*(1-p1)
				else:
					prod = prod*(1-p1*np.exp(-a*(j-36)))
				j = j+1
			total = total + p1*(1-p2)**(T-i)*prod
			i = i+1
		return (1 - total**(2*cell_count))

	def loss(a):
		return integrate.quad(lambda t: (ecdf(t)-model(t, a))**2, 0, 73)[0]

	a = op.minimize_scalar(loss, bounds=(0, 10), method='bounded').x
	result = lambda t: model(t, a)
	print('a = {}, relative error =  {}%\n{}'.format(a, rel_err(ecdf, result, [0, 73]), div))
	fit_plot(ecdf, result, [0, 73], 'age in months at the time of diagnosis',\
	 'probability of being diagnosed with retinoblastoma', 'decaying p1 (unilateral)')
				