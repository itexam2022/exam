# import os
# os.getcwd()
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\wisha\Desktop\LABcode\ML") 
print(df)

#1Find Shape of Data
print(df.shape)

#2Find Missing Values
print(df.notnull())
print(df.isnull())

#3Find data type of each column
print(df.dtypes)

#4Finding out Zero's
count = (df==0).sum(axis=1) 
print(count) 
print(df[df == 0].count())

#5Find Mean age of patients
print(df['Age'].mean())

#6
new_df = df.filter(['Age','Sex','ChestPain','RestBP','Chol'])
new_df
# df[['Age','Sex','ChestPain','RestBP','Chol']]



from sklearn.model_selection import train_test_split
train, test = train_test_split(df, random_state = 0, test_size = 0.25)
print(train.shape)
print(test.shape)

from sklearn.metrics import ConfusionMatrixDisplay, classification_report
actual = list(np.ones(45)) + list(np.zeros(55))
print(actual)
predicted = list(np.ones(40)) + list(np.zeros(52)) + list(np.ones(8))
print(predicted)
#confusion matrix
ConfusionMatrixDisplay.from_predictions(actual, predicted)
#I. Accuracy II. Precision III. Recall IV. F-1 score
print(classification_report(actual, predicted))