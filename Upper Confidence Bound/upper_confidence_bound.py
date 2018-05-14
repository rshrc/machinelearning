# Upper Confidence Bound
# CTR = Click Through Rate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB Algorithm
d = 10 # Number of Ads
N = 10000 # Number of Users
ads_selected = []

# Creating a vector of size d, all values initialized to 0
import math
number_of_selections  = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(0,N):
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        if number_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i]) # n + 1 because, of indexing
            # Computing Upper Bound
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 # 10^400
        
        if upper_bound > max_upper_bound:   #
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
            
# Visualizing the results
plt.hist(ads_selected)
plt.title("Histogram of Ads Selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each add was selected")
plt.show()
        
        
        
        