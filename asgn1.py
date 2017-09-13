from cancer import *

unilateral = [] # Stores unilateral retinoblastoma data set
p1, p2 = 0., 0. # Random hit probabilities
ecdf = lambda t: sm.distributions.ECDF(unilateral)(t)*rb_rate # Empirical cdf of age at the time of diagnosis 
# of unilateral retinoblastoma 

def assignment_1_i():
	global unilateral, ecdf_norm
	print('Problem 1(i)\n{}'.format(div))
	unilateral = read_data("./../data/03_cancer_unilateralretinoblastoma.csv")[0]
	
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
		return (1 - total**(2*cell_count))*rb_rate

	def loss(p):
		return sum([(model(t, p)-ecdf(t))**2 for t in range(36)])

	p1, p2 = op.minimize(loss, [1e-6, 40e-6]).x
	result = lambda t: model(t, [p1, p2])
	print('p1 = {}, p2 = {}, relative error =  {}%\n{}'.format(p1, p2, rel_err(ecdf, result, [0, 73]), div))
	fit_plot(ecdf, result, [0, 73], 'age in months at the time of diagnosis',\
	 'probability of being diagnosed with retinoblastoma', 'two random factors (unilateral)')

def assignment_1_iii():
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
		return (1 - total**(2*cell_count))*rb_rate

	def loss(a):
		return sum([(model(t, a)-ecdf(t))**2 for t in range(73)])

	a = op.minimize_scalar(loss, bounds=(0,1), method='bounded').x
	result = lambda t: model(t, a)
	print('a = {}, relative error =  {}%'.format(a, rel_err(ecdf, result, [0, 73])))
	fit_plot(ecdf, result, [0, 73], 'age in months at the time of diagnosis',\
	 'probability of being diagnosed with retinoblastoma', 'decaying p2 (unilateral)')
				