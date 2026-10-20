from pyod.utils.data import generate_data
from matplotlib import pyplot

X_train, X_test, y_train, y_test = generate_data(
    n_train=400, 
    n_test=100, 
    n_features=2, 
    contamination=0.1
    )

pyplot.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')

pyplot.show()