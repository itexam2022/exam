import pandas as pd
import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#1Data Preprocessing Generating list of transactions 
dataset = []
with open('Market_Basket_Optimisation.csv') as file:
    reader = csv.reader(file ,delimiter=",")
    for row in reader:
        dataset += [row]

dataset[0:10]

#structuring unsture dataset
te = TransactionEncoder()
x = te.fit_transform(dataset)
x

df = pd.DataFrame(x, columns=te.columns_)
df.head()

#2Train Apriori algorithm
#find frequent items to make rule
freq_itemset = apriori(df, min_support=0.01, use_colnames=True)
freq_itemset 

#4Generating rules
rules = association_rules(freq_itemset, metric="confidence", min_threshold=0.25)
rules = rules[['antecedents','consequents','support','confidence']]
rules.head()
rules[rules['antecedents'] == {'cake'}]
