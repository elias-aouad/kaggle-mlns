# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:47:37 2021

@author: elias
"""

import pandas as pd

csv = ['node_information.csv', 'random_predictions.csv']

df = pd.read_csv(csv[0], header=None)
cols = ['id', 'year', 'title', 'authors', 'journal', 'abstract']
df.columns = cols

pred = pd.read_csv(csv[1])
