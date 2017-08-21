# Upper Confidence Bound in R
# Reinforcement learning algorithm

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing the UCB algorithm for advertisements data
N = 10000
d = 10
total_reward = 0
ads_selected = integer(0)
num_selections = integer(d)
sums_of_rewards = integer(d)

for (n in 1:N) { 
  ad = 0
  max_upper_bound = 0
  
  for (i in 1:d) {
    
    if (num_selections[i] > 0) { 
      avg_reward = sums_of_rewards[i] / num_selections[i]
      delta_i = sqrt(3/2 * log(n) / num_selections[i])
      upper_bound = avg_reward + delta_i
      
    } else {
      upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound) {
      max_upper_bound = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  num_selections[ad] = num_selections[ad] + 1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Visualizing the UCB results - histogram
hist(ads_selected, col = 'blue',
     main = 'Histogram of Ads Selections - UCB',
     xlab = 'Ads',
     ylab = 'Number of times selected')
     