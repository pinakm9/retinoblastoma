from cancer import *

cancers = []

def assignment_2_a():
	names, columns = read_data('./../data/03_cancer_agerelated.csv')
	parse_groups(read_data('./../data/03_cancer_agerelated.csv')[0])
	for j in range(1, len(names)):
		cancers.append(Cancer(names[j], columns[j]))
		fit_plot(cancers[j-1].ecdf, cancers[j-1].ecdf, [0, 100], img_name = cancers[j-1].name)


