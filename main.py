import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_csv("student-por.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print('Accuracy: \n',acc)
# print('Coefficient: \n', linear.coef_)
# print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

plt.figure(figsize=(10,10))
plt.scatter(y_test, predictions, c='crimson')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
p1 = max(max(predictions), max(y_test))
p2 = min(min(predictions), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.show()


for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])