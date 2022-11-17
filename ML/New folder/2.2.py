import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

df = pd.read_csv("")
df.head()

# input data
x = df['YEAR']

# output data
y = df['ANNUAL']

# 1)
x.shape
x = x.values #converting in to array
x.reshape(117, 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
regressor.coef_
regressor.intercept_
regressor.predict([[2024]])
regressor.predict([[2034]])

# 2)
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

# 3)
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