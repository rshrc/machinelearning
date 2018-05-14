# Apriori | Optimizing the sales of a Store

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
# Because the first transaction is not the title
transactions = []
# Making a list of lists
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.3, min_lift = 3, min_length = 2)

# Visualising the Results
results = list(rules)

