# Thompson Sampling in R
# Reinforcement learning algorithm

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling for advertisements data 
N = 10000
d = 10
total_reward = 0
ads_selected = integer(0)
num_of_rewards0 = integer(d)
num_of_rewards1 = integer(d)

# at each round N, we consider two items for each ad i :
# # times the ad i got reward 1 up to round N;
# # times the ad i got reward 0 up to round N
# for each ad i we take a random draw from beta distribution
# select ad with highest posterior probability
 
for (n in 1:N) { 
  ad = 0
  max_rand_draw = 0
  
  for (i in 1:d) {
    rand_beta = rbeta(n = 1, shape1 = num_of_rewards1[i] + 1, 
                      shape2 = num_of_rewards0[i] + 1)
    
    if (rand_beta > max_rand_draw) {
      max_rand_draw = rand_beta
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  if ( reward == 1){
    num_of_rewards1[ad] = num_of_rewards1[ad] + 1
  }
  else {
    num_of_rewards0[ad] = num_of_rewards0[ad] + 1
  }
  total_reward = total_reward + reward
}

# Visualizing the UCB results - histogram
hist(ads_selected, col = 'blue',
     main = 'Histogram of Ads Selections - TS',
     xlab = 'Ads',
     ylab = 'Number of times selected')
