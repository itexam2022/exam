import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

df = pd.read_csv("/content/temperatures.csv") 
print(df)

x = df["YEAR"]
y = df["ANNUAL"]

plt.figure(figsize=(16,9))
plt.title('temp plot of INDIA')
plt.xlabel('year')
plt.ylabel('Annual')
plt.scatter(x, y)

x.shape
x.reshape(117,1)
y.shape
y.reshape(117,1)

# either use following way or above to get train and test data
# from sklearn.model_selection import train_test_split 
# x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=0)


#1Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
regressor.intercept_ 
regressor.coef_ 
regressor.predict([[2024]])

#2MSE, MAE and R-Square
predicted = regressor.predict(x)

# MAE
np.mean(abs(y- predicted))
from sklearn.matrics import mean_absolute_error
mean_absolute_error(y, predicted)
# MSE
np.mean((y- predicted) ** 2)
from sklearn.matrics import mean_squared_error
mean_squared_error(y, predicted)
#rsquared
from sklearn.matrics import r2_score
r2_score(y, predicted)

#3plot
plt.figure(16, 9)
plt.title("temp plot of india")
plt.xlabel('year')
plt.ylabel('annual')
plt.scatter(x, y, label = 'actual', color = 'g')
plt.plot(x, predicted, label = 'predicted', color = 'r')
plt.legend()
plt.show()

#shortcut
sns.regplot(x="YEAR", y="ANNUAL", data=df)