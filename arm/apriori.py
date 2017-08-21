# Apriori - Association Rule Mining (ARM) in Python

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
# support = (set of transactions containing item X)/ totaltransactions 
# ex: (3x per day * 7 days per week) / total transactions = 21/7500 = 0.003
# confidence = rules have to be correct _% of the time
# lift should be > 3
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2,
                min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
