import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ================================
# 1) LOAD DATA
# ================================
data_path = "states_datasets/data_subset_Uttar Pradesh 09.csv"
df = pd.read_csv(data_path, low_memory=False)

# ================================
# 2) VARIABLE SETUP (as before)
# ================================
binary_health = [
    "SM4","SM5","SM6","SM7","SM8","SM9","SM10","SM12",
    "MB3","MB4","MB5","MB6","MB7","MB8","MB9","MB10",
    "MB11","MB12","MB13","MB14","MB15","MB16","MB17"
]

severity_vars = [
    "SM3", "SM11", "MB18", "MB24"
]

cost_vars = [
    "SM18","SM20","SM21","SM22",
    "MB25","MB26","MB27","MB28","MB29"
]

# -----------------------------
# Gender & Education (for regression)
# -----------------------------
# Change these to match actual dataset vars
gender_var = "RO6"   # example: sex
education_vars = ["ED2","ED3","ED4"]  # replace as needed

# District variable (if present)
district_var = "DIST2011"  # change if needed

# ================================
# 3) CLEAN & RECODE
# ================================
def recode_binary(series):
    return series.replace({1:1, 2:0}).fillna(0)

for col in binary_health:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = recode_binary(df[col])

for col in severity_vars + cost_vars:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in [gender_var] + education_vars:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ================================
# 4) BUILD SUB-INDICES
# ================================
df["binary_burden"] = df[binary_health].sum(axis=1) / len(binary_health)
df["severity_sum"]  = df[severity_vars].fillna(0).sum(axis=1)
df["cost_sum"]      = df[cost_vars].fillna(0).sum(axis=1)

df["severity_norm"] = MinMaxScaler().fit_transform(df[["severity_sum"]])
df["cost_log"]      = np.log1p(df["cost_sum"])
df["cost_norm"]     = MinMaxScaler().fit_transform(df[["cost_log"]])

# ================================
# 5) PCA-BASED INDEX
# ================================
features = df[["binary_burden","severity_norm","cost_norm"]].fillna(0)
features_std = StandardScaler().fit_transform(features)

pca = PCA(n_components=1)
df["health_pca"] = pca.fit_transform(features_std)

print("PCA weights:", pca.components_)

# Normalize PCA index 0–1
df["health_pca_norm"] = MinMaxScaler().fit_transform(df[["health_pca"]])

# ================================
# 6) REGRESSION ADJUSTMENT
# ================================
# Ensure age exists
df["RO5"] = pd.to_numeric(df["RO5"], errors="coerce")

# Build regression for health_pca_norm ~ age + gender + education
reg_vars = ["RO5", gender_var] + education_vars
X = df[reg_vars].fillna(0)
X = sm.add_constant(X)

y = df["health_pca_norm"].fillna(0)
model = sm.OLS(y, X).fit()
print(model.summary())

# Predicted adjusted health
df["health_adjusted"] = model.predict(X)

# Normalize adjusted health
df["health_adjusted_norm"] = MinMaxScaler().fit_transform(
    df[["health_adjusted"]]
)

# ================================
# 7) HOUSEHOLD & DISTRICT AGGREGATES
# ================================
agg_cols = ["health_pca_norm", "health_adjusted_norm"]

if "HHID" in df.columns:
    hh_agg = df.groupby("HHID")[agg_cols].mean().reset_index()
    hh_agg.to_csv("health_index_household_avg.csv", index=False)
    print("Saved household average health")

if district_var in df.columns:
    dist_agg = df.groupby(district_var)[agg_cols].mean().reset_index()
    dist_agg.to_csv("health_index_district_avg.csv", index=False)
    print("Saved district average health")

# ================================
# 8) OUTPUT & PLOTS
# ================================
out_cols = [
    "RO5","binary_burden","severity_norm","cost_norm",
    "health_pca_norm","health_adjusted_norm"
]
df[out_cols].to_csv("health_index_full_output.csv", index=False)

print("All outputs saved.")

# Plot distributions
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(df["health_pca_norm"].dropna(), bins=50)
plt.title("PCA Health Index Distribution")

plt.subplot(1,2,2)
plt.hist(df["health_adjusted_norm"].dropna(), bins=50)
plt.title("Regression-Adjusted Health Index")

plt.savefig("health_index_distributions.png")
plt.show()
