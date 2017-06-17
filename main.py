import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from adspy_shared_utilities import plot_fruit_knn


fruits = pd.readtable('fruit_data_with_colors.txt')

lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

X = fruits[['mass', 'width', 'height']]
Y = fruits[['fruit_label']]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(x_train, y_train)
knn.score(x_test,y_test)

plot_fruit_knn(x_train, y_train, 5, 'uniform')