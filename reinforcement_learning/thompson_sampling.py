# Thompson Sampling in Python
# Reinforcement learning algorithm

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
 
# Implementing Thomson Sampling for advertisements data

N = 10000
d = 10
ads_selected = []
num_selections = [0] * d
num_of_rewards0 = [0] * d
num_of_rewards1 = [0] * d
total_reward = 0

# at each round N, we consider two items for each ad i :
    # # times the ad i got reward 1 up to round N;
    # # times the ad i got reward 0 up to round N
# for each ad i we take a random draw from beta distribution
# select ad with highest posterior probability
for n in range(0, N):
    ad = 0
    max_rand_draw = 0
    for i in range(0, d):    
        rand_beta = random.betavariate(num_of_rewards1[i] + 1, num_of_rewards0[i] + 1)
        if rand_beta > max_rand_draw:
            max_rand_draw = rand_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if (reward == 1):
        num_of_rewards1[ad] = num_of_rewards1[ad] = num_of_rewards1[ad] + 1
    else:
        num_of_rewards0[ad] = num_of_rewards0[ad] + 1
    total_reward = total_reward + reward
          
# Visualizing the histogram for ads selected results
plt.hist(ads_selected)
plt.title('Histogram of Ads Selections - TS')
plt.xlabel('Ads')
plt.ylabel('Number of times selected')
plt.show()