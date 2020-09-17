#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as LA


def lsq(X, A, B, alpha=0.1):
    return LA.norm(np.dot(A,X)-B) + alpha * LA.norm(X)


fun = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
bnds = ((-2, 2), (-1, 3))

