#!/usr/bin/env python3

"""
test for the helper `optimize.ga_min`
"""

from pyrimidine import optimize

"""
min x1^2 + x2
x1, x2 in [-1, 1]
"""
solution = optimize.ga_minimize(lambda x:x[0]**2+x[1], (-1,1), (-1,1))
print(solution)


solution = optimize.ga_minimize(lambda x:x[0]**2+x[1], (-1,1), (-1,1), init_x=[0.5,0])
print(solution)
