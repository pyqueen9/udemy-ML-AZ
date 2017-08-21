# Upper Confidence Bound (UCB) in Python
# Reinforcement learning algorithm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB algorithm for advertisements data
N = 10000
d = 10
ads_selected = []
num_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

# compute average reward and confidence interval at each round N
# see which ad is selected as N gets close to 10000
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):    
        if (num_selections[i] > 0):
            avg_reward = sums_of_rewards[i] / num_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / num_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    num_selections[ad] = num_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
        
# Visualizing the histogram for ads selected results
plt.hist(ads_selected)
plt.title('Histogram of Ads Selections - UCB')
plt.xlabel('Ads')
plt.ylabel('Number of times selected')
plt.show()