from cancer import *

cancers = []

def assignment_2_a():
	print('\n\nProblem 2(a)\n{}'.format(div))
	global cancers
	names, columns = read_data('./../data/03_cancer_agerelated.csv')
	for j in range(1, len(names)):
		cancers.append(Cancer(names[j], columns[j]))
		cancers[j-1].fit('2(a)')

def assignment_2_b():
	print('\n\nProblem 2(b)\n{}'.format(div))
	for j in range(len(cancers)):
		cancers[j].fit('2(b)')
