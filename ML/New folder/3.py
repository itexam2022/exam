import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv()

from sklearn.prerocessing import Binarizer 
bi = Binarizer(threshold = 0.75)
df['Chance of Admit '] = bi.fi_trasnform(df[['Chance of Admit ']])

x = df.drop('Chance of Admit ', axis = 1)
y = df['Chance of Admit ']
y.astype('int')

y.value_counts()
sns.countplot(x = y) #here xis keyword argument not a input

from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)

x_train.shape

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fir(x_train, y_train)

y_pred = classifier.predict(x_test)
result = pd.DataFrame({
    'actual':y_test,
    'predicted':y_pred
})
result

from sklearn.matrics import ConfusionMatrixDisplay, accuracy_score, classification_report
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

# new = [[136,314,109,4,3.5,4.0,8.77,1]]
# classifier.predict(new)[0]

# from sklearn.tree import plot_tree
# plt.figure(12,12)
# plot_tree(classifier,);