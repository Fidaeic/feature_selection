#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import stepwise_forward_regression, pls_cross_validation
import numpy as np
import statsmodels.api as sm
import json
#%%
# Simulate a process with 100 variables and 10K observations. Only 10 variables are correlated with the dependent variable
np.random.seed(1234)
n = 10000
p = 100
X = pd.DataFrame(np.random.normal(size=(n, p)), columns=['V'+str(i) for i in range(p)])
beta = np.zeros(p)
beta[:10] = 1
# Generate the dependent using an inverse logit function and round to the closest integer
y_prob = 1 / (1 + np.exp(-X.dot(beta)))
y = np.random.binomial(1, y_prob)

real_auc = roc_auc_score(y, y_prob)
print("Real AUC: ", real_auc)
#%%
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scores of the initial model using all the variables
base_lr = LogisticRegression()
base_lr.fit(X_train, y_train)
print("AUC score", roc_auc_score(y_test, base_lr.predict_proba(X_test)[:, 1]))
print("Coefficients", base_lr.coef_)
print("Intercept", base_lr.intercept_)
#%%
# Perform stepwise forward regression
selected_features = stepwise_forward_regression(X_train, y_train)
# %%
print("Selected features", selected_features)
#%%
# Performance of the model given the selected features
stepwise_lr = LogisticRegression()
stepwise_lr.fit(X_train[selected_features], y_train)
print("AUC score", roc_auc_score(y_test, stepwise_lr.predict_proba(X_test[selected_features])[:, 1]))

print("Coefficients", stepwise_lr.coef_)
print("Intercept", stepwise_lr.intercept_)
#%%
# Fit the model with L1 regularization
model = LogisticRegression(penalty='l1', l1_ratio=.7,solver='liblinear')
model.fit(X_train, y_train)

# Print the coefficients
print(model.coef_)
print(model.intercept_)
# %%
comps, vips = pls_cross_validation(X_train, y_train, range(1, 10))

# %%
X_train_const = sm.add_constant(X_train.loc[:, vips > 1])
model = sm.Logit(y_train, X_train_const)
result = model.fit()

# Print the summary
result.summary()
# %%
pls_lr = LogisticRegression()
pls_lr.fit(X_train.loc[:, vips > 1], y_train)
print(roc_auc_score(y_test, pls_lr.predict_proba(X_test.loc[:, vips > 1])[:, 1]))

print("Coefficients", pls_lr.coef_)
print("Intercept", pls_lr.intercept_)
# %%
# Compare the results of the different models and save them in a json file. Save the the coefficients, the AUC score, the selected features, the number of features

results = {
    "simulated":{
        "real_auc": real_auc,
        "n_variables": p,
        "n_observations": n,
    },
    "base_lr": {
        "coefficients": base_lr.coef_.tolist(),
        "intercept": base_lr.intercept_.tolist(),
        "auc": roc_auc_score(y_test, base_lr.predict_proba(X_test)[:, 1]),
        "selected_features": X.columns.tolist(),
        "n_features": int(p)  # Convert to Python int
    },
    "stepwise_lr": {
        "coefficients": stepwise_lr.coef_.tolist(),
        "intercept": stepwise_lr.intercept_.tolist(),
        "auc": roc_auc_score(y_test, stepwise_lr.predict_proba(X_test[selected_features])[:, 1]),
        "selected_features": selected_features,
        "n_features": int(len(selected_features))  # Convert to Python int
    },
    "pls_lr": {
        "coefficients": pls_lr.coef_.tolist(),
        "intercept": pls_lr.intercept_.tolist(),
        "auc": roc_auc_score(y_test, pls_lr.predict_proba(X_test.loc[:, vips > 1])[:, 1]),
        "selected_features": X.columns[vips > 1].tolist(),
        "n_features": int(sum(vips > 1))  # Convert to Python int
    }
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)
# %%
