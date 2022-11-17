import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Admission.csv")

from sklearn.prerocessing import Binarizer 
bi = Binarizer(threshold = 0.75)
df['Chance of Admit '] = bi.fi_trasnform(df[['Chance of Admit ']])

x = df.drop('Chance of Admit ', axis = 1)
y = df['Chance of Admit ']
y.astype('int') #changing y datatype to int from float
y.value_counts()

#1data-preparation
from sklearn.model_selection import train_test_split 
x_train , x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)

#2Classification Algorithm
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fir(x_train, y_train) 
y_pred = classifier.predict(x_test)

result = pd.DataFrame({
    'actual':y_test,
    'predicted':y_pred
})
result

#3Evaluate Model
from sklearn.matrics import ConfusionMatrixDisplay, accuracy_score, classification_report
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))