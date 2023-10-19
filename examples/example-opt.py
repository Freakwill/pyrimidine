#!/usr/bin/env python3

from pyrimidine import optimize


solution = optimize.ga_min(lambda x:x[0]**2+x[1], (-1,1), (-1,1))
print(solution)
