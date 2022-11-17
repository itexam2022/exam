import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\wisha\Desktop\LABcode\ML") 
print(df)

print(df.shape) 

print(df.notnull())
print(df.isnull())

print(df.dtypes)

count = (df==0).sum(axis=1) 
print(count) 
print(df[df == 0].count())

print(df['Age'].mean())

new_df = df.filter(['Age','Sex','ChestPain','RestBP','Chol'])
print(new_df)
print(df[['Age','Sex','ChestPain','RestBP','Chol']])

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, rando_state = 0, test_size = 0.25)
print(train.shape)
print(test.shape)

from sklearn.matrics import ConfusionMatrixDisplay, classification_report
actual = list(np.ones(45)) + list(np.zeros(55))
print(actual)
predicted = list(np.ones(40)) + list(np.zeros(52)) + list(np.ones(8))
print(predicted)
ConfusionMatrixDisplay.from_predictions(actual, predicted)
print(classification_report(actual, predicted))




        
# pandas as pd
# shape
# dtypes
# head()
# sum()
# isnull()
# notnull()
# count()
# mean()
# filter()

# from sklearn.matrics import ConfusionMatrixDisplay, classification_report
# ConfusionMatrixDisplay.from_predictions(actual, predicted)
# print(classification_report(actual, predicted))

# from sklearn.model_selection import train_test_split
# train, test = train_test_split(df, random_state=0)
