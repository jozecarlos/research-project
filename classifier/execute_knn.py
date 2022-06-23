from knn import KNN
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 30)

nearest_neighbors = KNN(3).fit(X_train, y_train)

y_pred = nearest_neighbors.predict(X_test)

nearest_neighbors.evaluate(y_pred, y_test)
