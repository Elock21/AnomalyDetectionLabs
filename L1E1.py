from pyod.utils.data import generate_data
from matplotlib import pyplot

X_train, X_test, Y_train, Y_test = generate_data(
    training_samples=400, 
    test_samples=100, 
    n_features=2, 
    contamination=0.1
    )

pyplot.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='coolwarm')

pyplot.show()