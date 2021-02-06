import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from sklearn.datasets import make_classification
from sklearn.svm import SVC


def projection_simplex(v, z=1):
    """
    w^* = argmin_w 0.5 ||w-v||^2 s.t. \sum_i w_i = z, w_i >= 0
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


class SVM:

    def __init__(self, C=1, max_iter=100, eps=0.01, random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        self.verbose = verbose

    def partial_gradient(self, X, y, i):
        g = np.dot(X[i], self.coef.T) + 1
        g[y[i]] -= 1
        return g

    def violation(self, g, y, i):
        # Optimality violation
        smallest = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dual_coef[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_coef[k, i] >= 0:
                continue

            smallest = min(smallest, g[k])

        return g.max() - smallest

    def solver(self, g, y, norms, i):
        # Prepare inputs to the projection
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.dual_coef[:, i]) + g / norms[i]
        z = self.C * norms[i]

        # Compute projection the simplex
        beta = projection_simplex(beta_hat, z)

        return Ci - self.dual_coef[:, i] - beta / norms[i]

    def fit(self, X, y):
        n_samples, n_features = X.shape

        n_classes = 4
        self.dual_coef = np.zeros((n_classes, n_samples), dtype=np.float64)
        self.coef = np.zeros((n_classes, n_features))

        # Pre-compute norms
        norms = np.sqrt(np.sum(X ** 2, axis=1))

        # Shuffle sample indices
        rs = check_random_state(self.random_state)
        ind = np.arange(n_samples)
        rs.shuffle(ind)

        violation_init = None
        for it in range(self.max_iter):
            violation_sum = 0

            for idx in range(n_samples):
                i = ind[idx]

                if norms[i] == 0:
                    continue

                g = self.partial_gradient(X, y, i)
                v = self.violation(g, y, i)
                violation_sum += v

                if v < 1e-12:
                    continue

                delta = self.solver(g, y, norms, i)

                # Update coefficients
                self.coef += (delta * X[i][:, np.newaxis]).T
                self.dual_coef[:, i] += delta

            if it == 0:
                violation_init = violation_sum

            vratio = violation_sum / violation_init

            if self.verbose >= 1:
                print("iter", it + 1, "violation", vratio)

            if vratio < self.eps:
                if self.verbose >= 1:
                    print("Converged")
                break

        return self

    def predict(self, X):
        decision = np.dot(X, self.coef.T)
        pred = decision.argmax(axis=1)
        return pred


if __name__ == '__main__':
    mobile_data = pd.read_csv('clearDataset.csv')

    X, Y = mobile_data.drop(['price_range'], axis=1), mobile_data['price_range']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    svm = SVM()
    scores = []
    for i in range(10):
        svm.fit(x_train.values, y_train.values)
        y_pred = svm.predict(x_train.values)
        scores.append(accuracy_score(y_train.values, y_pred))

    print(f'Train accuracy score: {sum(scores) / len(scores)}')

    scores = []
    for i in range(10):
        svm.fit(x_train.values, y_train.values)
        y_pred = svm.predict(x_test.values)
        scores.append(accuracy_score(y_test.values, y_pred))

    print(f'Test accuracy score: {sum(scores) / len(scores)}')

    clf = SVC(decision_function_shape='ovr')
    clf.fit(x_train.values, y_train.values)
    y_pred = clf.predict(x_test.values)
    print(f'Sklearn accuracy score: {accuracy_score(y_test, y_pred)}')