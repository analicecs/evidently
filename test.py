import pandas as pd
from aif360.sklearn.metrics import conditional_demographic_disparity

y_true = pd.Series([0, 1, 1, 0, 1, 0, 1, 1])
prot_attr = pd.Series(["M", "F", "M", "F", "M", "F", "M", "F"])

cdd_value = conditional_demographic_disparity(y_true=y_true, prot_attr=prot_attr)

print("Conditional Demographic Disparity:", cdd_value)
