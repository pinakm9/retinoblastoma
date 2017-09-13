from cancer import *

unilateral = [] # Stores unilateral retinoblastoma data set
p1, p2 = 0., 0. # Random hit probabilities
ecdf = lambda t: sm.distributions.ECDF(unilateral)(t)*rb_rate # Empirical cdf of age at the time of diagnosis 
# of unilateral retinoblastoma 
ecdf_norm = 0 

def assignment_1_i():
	global unilateral, ecdf_norm
	print('Problem 1(i)\n{}'.format(div))
	unilateral = read_data("./../data/03_cancer_unilateralretinoblastoma.csv")[0]
	ecdf_norm = integrate.quad(lambda t: ecdf(t)**2, min(unilateral), 73)[0]**0.5
	fit  = lambda t, p: (1-((1-p)**t + t*p*(1-p)**(t-1))**(2*cell_count))*rb_rate
	loss = lambda p: integrate.quad(lambda t: (ecdf(t)-fit(t, p))**2, 0, 73)[0]
	p = op.minimize_scalar(loss, bounds=(1e-10, 1e-2), method='bounded').x
	print('Best fit p = {}, relative error = {}%\n{}'.format(p, 100*loss(p)**0.5/ecdf_norm, div))
	#fit_plot(ecdf, lambda t: fit(t, p), [0, 73], 'age in months at the time of diagnosis',\
	# 'probability of being diagnosed with retinoblastoma', 'single random factor')

def assignment_1_ii():
	global p1, p2
	print('Problem 1(ii)\n{}'.format(div))
	
	def model(t, p):
		i, T =  1, int(t)
		p1, p2 = p
		total = (1-p1)**T
		while i <= T:
			total = total + p1*(1-p1)**(i-1)*(1-p2)**(T-i)
			i = i+1
		return (1 - total**(2*cell_count))*rb_rate

	def residual(p):
		return sum([(model(t, p)-ecdf(t))**2 for t in range(36)]) 
	fig, ax = plt.subplots(figsize=(8,5))
	x = np.linspace(0,73, 20)
	for i in range(10):
		p = op.minimize(residual, [i*1e-7, 1e-6]).x
		print(p)
		y = [model(t, p) for t in x]
		#ax.plot(x, y)
	#plt.show()
	fit_plot(ecdf, lambda t: model(t, p), [0, 73], 'age in months at the time of diagnosis',\
	 'probability of being diagnosed with retinoblastoma', 'single random factor')

def assignment_1_iii():
	print('Problem 1(iii)\n{}'.format(div))
	def fit(t, a):
		_p2, t = p2, int(t)
		total = (1-p1)**t
		for i in range(1, t):
			if i > 36:
				_p2 = p2*np.exp(-a*(i-36))	
			total = total + p1*(1-p1)**(i-1)*(1-_p2)**(t-i)
		return (1 - total**(2*cell_count))*rb_rate
	loss = lambda a: sum([(ecdf(t)-fit(t, a))**2 for t in range(1, 73)])
	norm = sum([ecdf(t)**2 for t in range(1, 73)])**0.5
	a = op.minimize_scalar(loss).x
	print('a = {}, relative error = {}%\n{}'.format(a, loss(a), div))
	fit_plot(ecdf, np.vectorize(lambda t: fit(t, a)), [0, 73], 'age in months at the times of diagnosis',\
	 'probability of being diagnosed with retinoblastoma', 'decaying p2')
				
def assignment_1_iv():
	print('Problem 1(iv)\n{}'.format(div))
	def fit(t, a):
		_p1 = p1*np.exp(-a*(t-36)) if t > 36 else p1
		p1_, p2_ = 1-_p1, 1-p2
		r = p1_/p2_
		return (1- (p1_**t + p2_**t*_p1*(r**t - 1)/(p2-_p1))**(2*cell_count))*rb_rate
	loss = lambda a: integrate.quad(lambda t: (ecdf(t)-fit(t, a))**2, 0, 73)[0]
	a = op.minimize_scalar(loss, bounds=(0, 1), method='bounded').x
	print('a = {}, relative error = {}%\n{}'\
		.format(a, 100*loss(a)**0.5/ecdf_norm, div))
