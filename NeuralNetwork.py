from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = pd.read_csv('C:/Users/asynk/Desktop/Data/heart.csv')
x = data.drop(['target'], axis = 1)
y = data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 100)
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(x_train, y_train)
val_acc = accuracy_score(y_test, clf.predict(x_test))
print('Точность равна', val_acc)