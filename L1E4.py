import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot

np.random.seed(42)

contamination = 0.1
n_samples = 1000
n_outliers = int(n_samples * contamination)   # 100
n_inliers = n_samples - n_outliers            # 900

# mean array and covariance matrix for the normal points
mu = np.array([2, 3])
sigma = np.array([[4, 2],
                  [2, 3]])

# Cholesky decomposition: sigma = L @ L.T
L = np.linalg.cholesky(sigma)

# normal points: x ~ N(0, I), then y = L x + mu
# (points are in rows, so we write x @ L.T instead of L @ x)
x_in = np.random.randn(n_inliers, 2)
y_in = x_in @ L.T + mu

# anomalies: same procedure, but with a larger covariance (5 * sigma)
L_out = np.linalg.cholesky(5 * sigma)
x_out = np.random.randn(n_outliers, 2)
y_out = x_out @ L_out.T + mu

# final dataset and labels (0 = normal, 1 = anomaly)
X_train = np.vstack((y_in, y_out))
Y_train = np.concatenate((np.zeros(n_inliers), np.ones(n_outliers)))

# estimate the mean and covariance from the data (in practice we don't know them)
mean_est = np.mean(X_train, axis=0)
cov_est = np.cov(X_train, rowvar=False)

# Z-score multidimensional: z = ||L^-1 (y - mean)||
L_est = np.linalg.cholesky(cov_est)
x_back = np.linalg.solve(L_est, (X_train - mean_est).T)
z_scores = np.linalg.norm(x_back, axis=0)

# threshold: the value below which 90% of the Z-scores fall
threshold = np.quantile(z_scores, 1 - contamination)
print("Z-score threshold =", threshold)

# predictions (1 = anomaly, 0 = normal)
Y_train_pred = (z_scores > threshold).astype(int)

# confusion matrix -> [[TN, FP], [FN, TP]]
tn, fp, fn, tp = confusion_matrix(Y_train, Y_train_pred).ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
print("Train: TN =", tn, "FP =", fp, "FN =", fn, "TP =", tp)
print("Train balanced accuracy =", (tpr + tnr) / 2)

# plot: colors show the predictions
pyplot.scatter(X_train[:, 0], X_train[:, 1], c=Y_train_pred, cmap='coolwarm')
pyplot.title('Z-score anomaly detection (2D)')
pyplot.xlabel('Feature 1')
pyplot.ylabel('Feature 2')
pyplot.show()