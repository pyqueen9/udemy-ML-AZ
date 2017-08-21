# Apriori - Association Rule Mining in R

# Data Preprocessing
#install.packages('arules') 
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)

# create sparse matrix with transactions in baskets
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', 
                            rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
# support = (set of transactions containing item X)/ totaltransactions 
# ex: (3x per day * 7 days per week) / total transactions = 21/7500 = 0.003
# confidence = rules have to be correct _% of the time
# lift should be > 3

rules = apriori(data = dataset, parameter = list(support = 0.003 , 
                                                 confidence = 0.2))

# Visualizing the results - look at first 10 rules; sort by lift values
inspect(sort(rules, by = 'lift')[1:10])