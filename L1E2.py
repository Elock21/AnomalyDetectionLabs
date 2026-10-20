from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve
from matplotlib import pyplot


# data generation, same as in E1
X_train, X_test, Y_train, Y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1,
    random_state=42
    )

# model with default parameters, except contamination
clf = KNN(contamination=0.1)
clf.fit(X_train)

# predictions (pyod: 1 = anomaly, 0 = normal)
Y_train_pred = clf.predict(X_train)
Y_test_pred = clf.predict(X_test)

# confusion matrix -> [[True Negative, False Positive], [False Negative, True Positive]]
tn, fp, fn, tp = confusion_matrix(Y_train, Y_train_pred).ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
print("Train: TN =", tn, "FP =", fp, "FN =", fn, "TP =", tp)
print("Train balanced accuracy =", (tpr + tnr) / 2)

tn, fp, fn, tp = confusion_matrix(Y_test, Y_test_pred).ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
print("Test: TN =", tn, "FP =", fp, "FN =", fn, "TP =", tp)
print("Test balanced accuracy =", (tpr + tnr) / 2)

# ROC curve (uses the anomaly scores, not the 0/1 predictions)
Y_test_scores = clf.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(Y_test, Y_test_scores)

pyplot.plot(fpr, tpr, label='KNN')
pyplot.plot([0, 1], [0, 1], 'k--', label='Random')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('ROC curve')
pyplot.legend()
pyplot.show()

# changing the contamination rate 
for c in [0.01, 0.05, 0.1, 0.2, 0.3]:
    clf = KNN(contamination=c)
    clf.fit(X_train)
    Y_test_pred = clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(Y_test, Y_test_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    print("contamination =", c,
          "| TN =", tn, "FP =", fp, "FN =", fn, "TP =", tp,
          "| balanced accuracy =", (tpr + tnr) / 2)