#%%
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#%%
# Load the results
with open('results.json') as f:
    results = json.load(f)  
#%%
# Create a dataframe based on the results, using the keys as rows
df = pd.DataFrame(results).T
#%%
sns.barplot(x=df.index, y='auc', data=df)
# %%
df
# %%
