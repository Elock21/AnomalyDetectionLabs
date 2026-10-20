from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot

contamination = 0.1

X_train, X_test, Y_train, Y_test = generate_data(
    n_train=1000,
    n_test=0,
    n_features=1,
    contamination=contamination,
    random_state=42
    )

# Z-score for each point (absolute value, anomalies can be at both ends)
mean = np.mean(X_train)
std = np.std(X_train)
z_scores = np.abs(X_train - mean) / std

# threshold: the value below which 90% of the Z-scores fall
threshold = np.quantile(z_scores, 1 - contamination)
print("Z-score threshold =", threshold)

#predictions (1 = anomalies, 0 = normal)
Y_train_pred = (z_scores > threshold).astype(int).ravel()

# confusion matrix -> [[TN, FP], [FN, TP]]
tn, fp, fn, tp = confusion_matrix(Y_train, Y_train_pred).ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
print("Train: TN =", tn, "FP =", fp, "FN =", fn, "TP =", tp)
print("Train balanced accuracy =", (tpr + tnr) / 2)


pyplot.scatter(X_train[:, 0], np.zeros(len(X_train)), c=Y_train_pred, cmap='coolwarm')
pyplot.title('Z-score anomaly detection')
pyplot.xlabel('Value')
pyplot.show()