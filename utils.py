
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def stepwise_forward_regression(X, y):
    initial_features = X.columns.tolist()
    best_features = []
    best_auc = 0

    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        auc_values = []

        for new_column in remaining_features:
            model = LogisticRegression()
            model.fit(X[best_features + [new_column]], y)
            predictions = model.predict_proba(X[best_features + [new_column]])[:,1]
            auc = roc_auc_score(y, predictions)
            auc_values.append((auc, new_column))

        if auc_values:  # Check if auc_values is not empty
            auc_values.sort(reverse=True)
            if auc_values[0][0] > best_auc:
                best_auc = auc_values[0][0]
                best_features.append(auc_values[0][1])
            else:
                break
        else:
            break

    return best_features


def compute_vip(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips

def pls_cross_validation(X, y, component_range):
    mse = []
    for i in component_range:
        pls = PLSRegression(n_components=i)
        scores = cross_val_score(pls, X, y, cv=5, scoring='roc_auc')
        mse.append(-1*np.mean(scores))

    optimal_components = component_range[np.argmin(mse)]
    pls_optimal = PLSRegression(n_components=optimal_components)
    pls_optimal.fit(X, y)

    vips = compute_vip(pls_optimal)

    return optimal_components, vips

def simulate_data(n, p, intercept, meaningful_features=5):
    
    beta = np.zeros(p)
    beta[:meaningful_features] = np.random.uniform(0.5, 2, meaningful_features)
    var_names = ['V{:02d}'.format(i) for i in range(1, 101)]
    X = pd.DataFrame(np.random.normal(size=(n, p)), columns= var_names)

    X_standardized = StandardScaler().fit_transform(X)
    y = np.random.binomial(1, 1/(1 + np.exp(-X_standardized @ beta-intercept)))
    return X, y, beta, var_names