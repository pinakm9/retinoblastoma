import asgn1, asgn2, warnings, os
from cancer import *

warnings.filterwarnings('ignore')
try: 
    os.makedirs('plots')
except:
    pass
print('\n'*5)
asgn1.assignment_1_a()
asgn1.assignment_1_b()
asgn1.assignment_1_c()
asgn1.assignment_1_d()
asgn2.assignment_2_a()
asgn2.assignment_2_b()