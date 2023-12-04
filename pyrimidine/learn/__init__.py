#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BaseEstimator

import warnings, os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'